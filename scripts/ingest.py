#!/usr/bin/env python3
"""Ingest documents from the documents/ folder into the vector DB."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from llama_index.core import SimpleDirectoryReader
from rich.console import Console
from rich.progress import Progress
from src.rag import add_documents

console = Console()
DOCS_DIR = ROOT / "documents"


def main():
    if not DOCS_DIR.exists() or not any(DOCS_DIR.iterdir()):
        console.print("[yellow]No documents found in documents/ — drop your PDFs, DOCX, or TXT files there first.[/yellow]")
        return

    files = list(DOCS_DIR.rglob("*"))
    supported = [f for f in files if f.suffix.lower() in {".pdf", ".docx", ".txt", ".md"} and f.is_file()]

    if not supported:
        console.print(f"[yellow]No supported files found. Supported: .pdf .docx .txt .md[/yellow]")
        return

    console.print(f"[cyan]Found {len(supported)} document(s) to ingest:[/cyan]")
    for f in supported:
        console.print(f"  • {f.name}")

    console.print("\n[cyan]Loading documents...[/cyan]")
    reader = SimpleDirectoryReader(
        input_dir=str(DOCS_DIR),
        recursive=True,
        required_exts=[".pdf", ".docx", ".txt", ".md"],
    )
    documents = reader.load_data()
    console.print(f"[green]Loaded {len(documents)} chunk(s) from {len(supported)} file(s)[/green]")

    console.print("\n[cyan]Embedding and storing in ChromaDB...[/cyan]")
    add_documents(documents)
    console.print("[bold green]Done! Documents are now in the vector DB.[/bold green]")


if __name__ == "__main__":
    main()
