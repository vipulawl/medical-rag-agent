#!/usr/bin/env python3
"""Fetch new research from PubMed and ingest into the vector DB."""
import sys
import argparse
import os
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from llama_index.core import Document
from rich.console import Console
from rich.table import Table
from src.pubmed import fetch_new_research
from src.rag import add_documents

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch new PubMed research and add to knowledge base")
    parser.add_argument("--query", default=None, help="Override PubMed search query")
    parser.add_argument("--limit", type=int, default=None, help="Max results to fetch (default: from .env)")
    parser.add_argument("--days", type=int, default=None, help="Only fetch papers from last N days (default: from .env)")
    return parser.parse_args()


def main():
    args = parse_args()

    search_query = args.query or os.getenv("PUBMED_SEARCH_QUERY", "")
    max_results = args.limit or int(os.getenv("PUBMED_MAX_RESULTS", "10"))
    days_back = args.days or int(os.getenv("PUBMED_DAYS_BACK", "90"))

    if not search_query:
        console.print("[red]No search query set. Add PUBMED_SEARCH_QUERY to .env or use --query[/red]")
        sys.exit(1)

    console.print(f"[cyan]Searching PubMed for:[/cyan] {search_query}")
    console.print(f"[cyan]Limit:[/cyan] {max_results} papers | [cyan]Window:[/cyan] last {days_back} days\n")

    articles = fetch_new_research(search_query, max_results=max_results, days_back=days_back)

    if not articles:
        console.print("[yellow]No new articles found for this query and time window.[/yellow]")
        return

    table = Table(title=f"Found {len(articles)} article(s)", show_lines=True)
    table.add_column("PMID", style="dim", width=10)
    table.add_column("Title", max_width=60)
    table.add_column("Journal", max_width=30)
    table.add_column("Year", width=6)

    for a in articles:
        table.add_row(a["pmid"], a["title"], a["journal"], a["year"])

    console.print(table)

    documents = []
    for a in articles:
        text = (
            f"Title: {a['title']}\n"
            f"Authors: {a['authors']}\n"
            f"Journal: {a['journal']} ({a['year']})\n"
            f"PubMed ID: {a['pmid']}\n"
            f"URL: {a['url']}\n\n"
            f"Abstract:\n{a['abstract']}"
        )
        doc = Document(
            text=text,
            metadata={
                "source": a["source"],
                "file_name": f"PubMed_{a['pmid']}_{a['year']}.txt",
                "title": a["title"],
                "journal": a["journal"],
                "year": a["year"],
                "url": a["url"],
                "ingested_at": datetime.now().isoformat(),
            },
        )
        documents.append(doc)

    console.print(f"\n[cyan]Embedding and storing {len(documents)} article(s)...[/cyan]")
    add_documents(documents)
    console.print(f"[bold green]Done! {len(documents)} article(s) added to the knowledge base.[/bold green]")


if __name__ == "__main__":
    main()
