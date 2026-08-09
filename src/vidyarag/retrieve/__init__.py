"""Retrieval strategies.

Phase 2 ships dense retrieval only, which is the frozen baseline every later
enhancement is measured against. Hybrid fusion, reranking and decomposition
arrive in Phase 4 and are ablated independently against it.
"""

from vidyarag.retrieve.dense import RetrievedChunk, retrieve_dense

__all__ = ["RetrievedChunk", "retrieve_dense"]
