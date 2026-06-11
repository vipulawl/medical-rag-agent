#!/usr/bin/env python3
"""Run the research agent on a schedule to keep the knowledge base up to date."""
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import schedule
import time
from rich.console import Console

console = Console()
RESEARCH_SCRIPT = ROOT / "scripts" / "research_agent.py"
PYTHON = sys.executable


def run_research():
    console.print(f"\n[cyan][{datetime.now().strftime('%Y-%m-%d %H:%M')}] Running research agent...[/cyan]")
    result = subprocess.run(
        [PYTHON, str(RESEARCH_SCRIPT)],
        capture_output=False,
    )
    if result.returncode != 0:
        console.print("[red]Research agent exited with errors.[/red]")
    else:
        console.print("[green]Research agent completed successfully.[/green]")


def parse_args():
    parser = argparse.ArgumentParser(description="Schedule periodic PubMed research ingestion")
    parser.add_argument("--interval-hours", type=float, default=24, help="Run interval in hours (default: 24)")
    parser.add_argument("--run-now", action="store_true", help="Also run immediately on start")
    return parser.parse_args()


def main():
    args = parse_args()

    console.print(f"[bold cyan]Medical Research Scheduler[/bold cyan]")
    console.print(f"Running research agent every [green]{args.interval_hours}h[/green]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    if args.run_now:
        run_research()

    schedule.every(args.interval_hours).hours.do(run_research)

    while True:
        schedule.run_pending()
        next_run = schedule.next_run()
        try:
            time.sleep(60)
        except KeyboardInterrupt:
            console.print("\n[dim]Scheduler stopped.[/dim]")
            break


if __name__ == "__main__":
    main()
