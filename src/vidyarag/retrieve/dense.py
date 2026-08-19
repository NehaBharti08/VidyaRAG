"""Dense retrieval -- the frozen baseline.

Deliberately the simplest thing that works: embed the query with the same local
model used at ingest, search one collection, return the top k with their
payloads intact. Every enhancement in Phase 4 is measured as a delta against
this, so it must stay boring and must not change after Phase 2 ships.

Payloads are carried through rather than reduced to text, because the citation
a reader sees is assembled from that metadata. Dropping it here would make
citations unverifiable two layers further on.
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from vidyarag.llm.provider import embed_texts
from vidyarag.store.collection import DENSE_VECTOR


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One search hit, with everything needed to cite it."""

    chunk_id: str
    score: float
    text: str
    citation: str
    book_slug: str
    book_title: str
    chapter: str | None
    section: str | None
    page_start: int
    page_end: int
    printed_page: str | None
    license_name: str
    source_url: str

    prior_score: float | None = None
    """Score from the previous stage, when a later stage rescored this chunk.

    Set by reranking so a rank change can be attributed to the reranker rather
    than inferred. ``None`` means ``score`` is the only score there has been.
    """

    @classmethod
    def from_payload(cls, payload: dict[str, object], score: float) -> RetrievedChunk:
        """Build from a Qdrant payload, tolerating absent optional fields."""

        def text_of(key: str) -> str:
            value = payload.get(key)
            return "" if value is None else str(value)

        def optional(key: str) -> str | None:
            value = payload.get(key)
            return None if value is None else str(value)

        def number(key: str) -> int:
            value = payload.get(key)
            return int(value) if isinstance(value, int | float | str) else 0

        return cls(
            chunk_id=text_of("chunk_id"),
            score=score,
            text=text_of("text"),
            citation=text_of("citation"),
            book_slug=text_of("book_slug"),
            book_title=text_of("book_title"),
            chapter=optional("chapter"),
            section=optional("section"),
            page_start=number("page_start"),
            page_end=number("page_end"),
            printed_page=optional("printed_page"),
            license_name=text_of("license"),
            source_url=text_of("source_url"),
        )


def retrieve_dense(
    client: QdrantClient,
    query: str,
    *,
    collection: str,
    embedding_model: str,
    limit: int = 20,
    book_slug: str | None = None,
) -> list[RetrievedChunk]:
    """Embed the query and return the closest chunks.

    Args:
        client: Connected Qdrant client.
        query: The user's question.
        collection: Collection to search.
        embedding_model: Must match the model the index was built with -- a
            different model produces vectors in an unrelated space, and the
            search silently returns nonsense rather than failing.
        limit: How many candidates to return.
        book_slug: Optionally restrict the search to one book.

    Returns:
        Hits ordered by descending similarity.
    """
    vector = embed_texts([query], embedding_model)[0]

    query_filter = None
    if book_slug is not None:
        query_filter = models.Filter(
            must=[models.FieldCondition(key="book_slug", match=models.MatchValue(value=book_slug))]
        )

    response = client.query_points(
        collection_name=collection,
        query=vector,
        using=DENSE_VECTOR,
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    )
    return [
        RetrievedChunk.from_payload(point.payload or {}, float(point.score))
        for point in response.points
    ]
