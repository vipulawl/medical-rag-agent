import os
import chromadb
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    Document,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

DB_PATH = Path(__file__).parent.parent / "db"
COLLECTION_NAME = "medical_knowledge"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


def _configure_settings(llm_model: str = None, embed_model: str = None):
    Settings.llm = Ollama(
        model=llm_model or LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=120.0,
    )
    Settings.embed_model = OllamaEmbedding(
        model_name=embed_model or EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    Settings.chunk_size = 512
    Settings.chunk_overlap = 64


def get_vector_store():
    DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    return ChromaVectorStore(chroma_collection=collection), client


def get_index(llm_model: str = None, embed_model: str = None) -> VectorStoreIndex:
    _configure_settings(llm_model, embed_model)
    vector_store, _ = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )


def add_documents(documents: list[Document], llm_model: str = None, embed_model: str = None):
    _configure_settings(llm_model, embed_model)
    vector_store, _ = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )


def query(question: str, top_k: int = 5, stream: bool = True, llm_model: str = None):
    index = get_index(llm_model=llm_model)
    engine = index.as_query_engine(
        similarity_top_k=top_k,
        streaming=stream,
    )
    return engine.query(question)


def query_stream(question: str, top_k: int = 5, llm_model: str = None):
    """Yields (token, sources) where sources is populated on the final chunk."""
    index = get_index(llm_model=llm_model)
    engine = index.as_query_engine(similarity_top_k=top_k, streaming=True)
    response = engine.query(question)

    sources = []
    for node in getattr(response, "source_nodes", []):
        src = node.metadata.get("file_name") or node.metadata.get("source", "unknown")
        title = node.metadata.get("title", "")
        url = node.metadata.get("url", "")
        sources.append({"name": src, "title": title, "url": url})

    for token in response.response_gen:
        yield token, None

    yield "", sources


def get_collection_count() -> int:
    try:
        _, client = get_vector_store()
        collection = client.get_or_create_collection(COLLECTION_NAME)
        return collection.count()
    except Exception:
        return 0
