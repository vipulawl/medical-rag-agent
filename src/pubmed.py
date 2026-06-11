import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict


PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


def search_pubmed(query: str, max_results: int = 10, days_back: int = 90) -> List[str]:
    min_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y/%m/%d")
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "mindate": min_date,
        "datetype": "pdat",
        "sort": "relevance",
    }
    resp = requests.get(PUBMED_SEARCH_URL, params=params, timeout=15)
    resp.raise_for_status()
    ids = resp.json().get("esearchresult", {}).get("idlist", [])
    return ids


def fetch_abstracts(pmids: List[str]) -> List[Dict]:
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    resp = requests.get(PUBMED_FETCH_URL, params=params, timeout=30)
    resp.raise_for_status()

    import xml.etree.ElementTree as ET
    root = ET.fromstring(resp.text)
    articles = []

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        abstract_el = article.find(".//AbstractText")
        journal_el = article.find(".//Journal/Title")
        year_el = article.find(".//PubDate/Year")
        authors = article.findall(".//Author/LastName")

        pmid = pmid_el.text if pmid_el is not None else "unknown"
        title = title_el.text if title_el is not None else "No title"
        abstract = abstract_el.text if abstract_el is not None else ""
        journal = journal_el.text if journal_el is not None else "Unknown journal"
        year = year_el.text if year_el is not None else "Unknown year"
        author_list = ", ".join(a.text for a in authors[:3] if a.text)

        if not abstract:
            continue

        articles.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "year": year,
            "authors": author_list,
            "source": f"PubMed:{pmid}",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })

    return articles


def fetch_new_research(query: str, max_results: int = 10, days_back: int = 90) -> List[Dict]:
    pmids = search_pubmed(query, max_results=max_results, days_back=days_back)
    if not pmids:
        return []
    return fetch_abstracts(pmids)
