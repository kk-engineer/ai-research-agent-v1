import asyncio
import os
import signal
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from agent.cache import AsyncCache
from agent.cli import InteractiveShell
from agent.config import load_config
from agent.logger import FileLogWriter, fmt_ms
from agent.pipeline import Pipeline
from agent.startup import validate_connectivity

app = typer.Typer(
    name="agent",
    help="AI Research Agent — search, extract, synthesise.",
    add_completion=False,
)
console = Console()

_shutting_down = False


def _setup_signal_handlers():
    global _shutting_down

    def handle_sigint(signum, frame):
        global _shutting_down
        if _shutting_down:
            console.print("\n[red]Forced exit[/red]")
            os._exit(1)
        _shutting_down = True
        console.print("\n[yellow]Shutting down gracefully...[/yellow]")

    signal.signal(signal.SIGINT, handle_sigint)


@app.command()
def run(
    query: str = typer.Argument(..., help="Research query in natural language"),
    mode: str = typer.Option(
        "auto", "--mode", "-m", help="Force mode: auto | academic | general | hybrid"
    ),
    top_k: int = typer.Option(0, "--top-k", "-k", help="Override top-K chunks (0 = use config)"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass cache"),
    export_json: bool = typer.Option(False, "--export-json", "-j", help="Also save JSON report"),
    config_path: str = typer.Option("config.toml", "--config", "-c"),
):
    """Run the full research pipeline for QUERY with real-time logging.

    Pipeline progress is streamed to stderr. The final report and statistics
    are displayed on stdout upon completion.
    """
    cfg = load_config(Path(config_path))
    if top_k > 0:
        cfg.reranker.top_k = top_k
    if no_cache:
        cfg.cache.enabled = False

    console.print(Panel.fit(
        "[bold blue]AI Research Agent[/bold blue]",
        subtitle=f"[dim]Query:[/dim] {query}  [dim]Mode:[/dim] {mode}",
    ))

    file_logger = FileLogWriter()

    async def _run():
        await file_logger.start()
        cache: AsyncCache | None = None
        try:
            await validate_connectivity(cfg)
            pipeline = Pipeline(cfg)
            cache = pipeline.cache
            report = await pipeline.run(query)

            console.print()
            console.print(Panel.fit(
                "[bold green]Research Report[/bold green]",
                border_style="green",
            ))
            console.print()
            console.print(Markdown(report.markdown))

            stats = report.stats
            stats_dict = stats.model_dump() if hasattr(stats, "model_dump") else dict(stats)

            table = Table(title="Pipeline Statistics", box=None)
            table.add_column("Metric", style="bold cyan")
            table.add_column("Value", style="white")

            skip_keys = {"retriever_latencies"}
            for key, value in stats_dict.items():
                if key in skip_keys or isinstance(value, dict):
                    continue
                label = key.replace("_", " ").title()
                if key.endswith("_ms") and isinstance(value, (int, float)):
                    seconds = value / 1000
                    table.add_row(label.replace(" Ms", ""), f"{seconds:.2f}s")
                elif isinstance(value, float):
                    table.add_row(label, f"{value:,.2f}")
                else:
                    table.add_row(label, str(value))
            console.print(table)

            if "retriever_latencies" in stats_dict and stats_dict["retriever_latencies"]:
                rl_table = Table(title="Retriever Latencies", box=None)
                rl_table.add_column("Retriever", style="bold")
                rl_table.add_column("Latency", style="white")
                for name, lat in stats_dict["retriever_latencies"].items():
                    rl_table.add_row(name.title(), fmt_ms(lat))
                console.print(rl_table)

            report_path = report.save(Path(cfg.output.reports_dir))
            console.print(f"\n  [green]✔ Report saved →[/green] {report_path}")

            if export_json:
                json_path = report.to_json(Path(cfg.output.reports_dir))
                console.print(f"  [green]✔ JSON exported →[/green] {json_path}")

            console.print()

        except asyncio.CancelledError:
            console.print("\n[yellow]Operation cancelled[/yellow]")
        finally:
            if cache is not None:
                await cache.close()
            await file_logger.stop()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutdown complete[/yellow]")


@app.command()
def config_show():
    """Print resolved configuration (masks API keys)."""
    cfg = load_config(Path("config.toml"))
    dump = cfg.model_dump()
    if "api_keys" in dump:
        masked = {}
        for k, v in dump["api_keys"].items():
            masked[k] = v[:4] + "****" if v and len(v) > 4 else "****" if v else ""
        dump["api_keys"] = masked

    import json

    console.print_json(json.dumps(dump, indent=2, default=str))


@app.command()
def cache_clear():
    """Clear all cached content."""
    cfg = load_config(Path("config.toml"))

    async def _clear():
        cache = AsyncCache(cfg)
        await cache.clear()
        await cache.close()
        console.print("[green]✔ Cache cleared[/green]")

    asyncio.run(_clear())


@app.command()
def cache_stats():
    """Show cache statistics."""
    cfg = load_config(Path("config.toml"))

    async def _stats():
        cache = AsyncCache(cfg)
        try:
            stats = await cache.stats()
            console.print(Panel.fit("[bold]Cache Statistics[/bold]"))
            console.print(f"  L1 entries (chunks): {stats['l1_entries']}")
            console.print(f"  L2 entries (reports): {stats['l2_entries']}")
        finally:
            await cache.close()

    asyncio.run(_stats())


@app.command()
def health():
    """Run health checks on all retrievers and the LLM."""
    cfg = load_config(Path("config.toml"))
    console.print(Panel.fit("[bold]Running Health Checks[/bold]"))

    from agent.retrievers import (
        ArxivRetriever,
        DuckDuckGoRetriever,
        HackerNewsRetriever,
        SemanticScholarRetriever,
        WikipediaRetriever,
    )

    retrievers = [
        ("Arxiv", ArxivRetriever()),
        ("Semantic Scholar", SemanticScholarRetriever()),
        ("Wikipedia", WikipediaRetriever()),
        ("HackerNews", HackerNewsRetriever()),
        ("DuckDuckGo", DuckDuckGoRetriever()),
    ]

    async def _check_retrievers():
        import time

        for name, retriever in retrievers:
            t0 = time.perf_counter()
            try:
                ok = await retriever.health_check()
                latency = (time.perf_counter() - t0) * 1000
                status = "[green]✔ OK[/green]" if ok else "[red]✖ FAIL[/red]"
                console.print(f"  {status} {name} ({fmt_ms(latency)})")
            except Exception as e:
                latency = (time.perf_counter() - t0) * 1000
                console.print(f"  [red]✖ {name}[/red] — {e} ({fmt_ms(latency)})")

        console.print("[bold]LLM check:[/bold]")
        try:
            llm = create_llm(cfg)
            response = await llm.complete("Respond with exactly 'OK'.")
            if "ok" in response.lower():
                console.print(
                    f"  [green]✔ LLM ({cfg.llm.mode})[/green] — responded: {response.strip()}"
                )
            else:
                console.print(
                    f"  [yellow]⚠ LLM ({cfg.llm.mode})[/yellow] — "
                    f"unexpected response: {response.strip()}"
                )
        except Exception as e:
            console.print(f"  [red]✖ LLM ({cfg.llm.mode})[/red] — {e}")

    from agent.llm import create_llm

    asyncio.run(_check_retrievers())


@app.command()
def shell(
    config_path: str = typer.Option("config.toml", "--config", "-c"),
):
    """Start interactive shell mode with rich formatting.

    Type a research query in natural language, or use /commands.
    Run /help within the shell for available commands.
    """
    _setup_signal_handlers()
    shell = InteractiveShell(config_path=config_path)
    try:
        asyncio.run(shell.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting...[/yellow]")
    except SystemExit:
        pass


if __name__ == "__main__":
    app()
