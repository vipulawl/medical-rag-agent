#!/usr/bin/env python3
"""Interactive CLI chat with the medical RAG agent."""
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from src.rag import query as rag_query

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with your medical RAG assistant")
    parser.add_argument("--model", default=None, help="Override LLM model (default: from .env)")
    parser.add_argument("--top-k", type=int, default=5, help="Chunks to retrieve per query (default: 5)")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    return parser.parse_args()


def main():
    args = parse_args()
    stream = not args.no_stream

    console.print(Panel.fit(
        "[bold cyan]Medical RAG Assistant[/bold cyan]\n"
        "Powered by your documents + PubMed research\n"
        "[dim]Type 'exit' or Ctrl+C to quit[/dim]",
        border_style="cyan"
    ))

    SYSTEM_PREAMBLE = (
        "You are a specialized medical research assistant. "
        "Answer based on the provided context from medical documents and research papers. "
        "Always note when information comes from the research context vs general knowledge. "
        "If you are unsure, say so. This is for informational purposes only, not medical advice."
    )

    while True:
        try:
            question = console.input("\n[bold green]You:[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Goodbye.[/dim]")
            break

        full_question = f"{SYSTEM_PREAMBLE}\n\nQuestion: {question}"

        console.print("\n[bold blue]Assistant:[/bold blue]")
        try:
            response = rag_query(
                full_question,
                top_k=args.top_k,
                stream=stream,
                llm_model=args.model,
            )
            if stream:
                response.print_response_stream()
            else:
                console.print(Markdown(str(response)))

            if hasattr(response, "source_nodes") and response.source_nodes:
                console.print("\n[dim]Sources:[/dim]")
                seen = set()
                for node in response.source_nodes:
                    src = node.metadata.get("file_name") or node.metadata.get("source", "unknown")
                    if src not in seen:
                        console.print(f"  [dim]• {src}[/dim]")
                        seen.add(src)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            if "connection" in str(e).lower():
                console.print("[yellow]Is Ollama running? Try: ollama serve[/yellow]")


if __name__ == "__main__":
    main()
