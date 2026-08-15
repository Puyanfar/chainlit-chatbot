"""
RAG subsystem: everything related to embedding, storing, and retrieving FAQ
data in Qdrant. Re-exports the functions other modules actually need, so
callers can do `from rag import retrieve` instead of reaching into
`rag.retrieval` directly.
"""

from .retrieval import retrieve, build_augmented_question
from .ingestion import add_faq_pairs, ensure_collections, faq_id_for
from .clients import embed_texts, qdrant_client, embedding_client

__all__ = ["retrieve", "build_augmented_question", "add_faq_pairs", "ensure_collections", "faq_id_for", "embed_texts", "qdrant_client", "embedding_client"]