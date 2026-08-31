"""API tests.

Run against a real in-process Qdrant with a stubbed LLM, so routing, schema and
error handling are exercised without a key or a network call.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from vidyarag.api.routes_v1 import get_pipeline, router
from vidyarag.ingest.chunk import Chunk
from vidyarag.pipeline import Pipeline
from vidyarag.settings import IN_MEMORY, PipelineConfig, QdrantMode, Settings
from vidyarag.store.collection import ensure_collection, make_points, upsert_points

DIM = 8
COLLECTION = "test_collection"


class FakeResponse:
    text = "Glucose cannot cross the bilayer [1]."
    usage_metadata = type("Usage", (), {"prompt_token_count": 420, "candidates_token_count": 18})()


class FakeModels:
    def generate_content(self, **kwargs: Any) -> Any:
        return FakeResponse()


class FakeLLM:
    models = FakeModels()


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        GOOGLE_API_KEY="test-key-not-real",
        QDRANT_MODE=QdrantMode.EMBEDDED,
        QDRANT_PATH=IN_MEMORY,
        QDRANT_COLLECTION=COLLECTION,
        VIDYARAG_PROFILE="baseline",
    )


@pytest.fixture
def pipeline(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Iterator[Pipeline]:
    client = QdrantClient(location=IN_MEMORY)
    ensure_collection(client, COLLECTION, DIM)

    chunk = Chunk(
        chunk_id="biology-p0133-05",
        book_slug="biology",
        book_title="Biology",
        chapter="Chapter 5. Structure and Function",
        section="5.2. Passive Transport",
        page_start=133,
        page_end=133,
        printed_page="121",
        text="Glucose is large and polar, so it cannot cross the lipid bilayer.",
        token_count=14,
    )
    upsert_points(
        client,
        COLLECTION,
        list(
            make_points(
                [(chunk, [0.1] * DIM)],
                license_name="CC BY 4.0",
                source_url="https://openstax.org/details/books/biology",
            )
        ),
    )

    # Local embedding is stubbed: the model is not what these tests exercise.
    monkeypatch.setattr(
        "vidyarag.retrieve.dense.embed_texts",
        lambda texts, _model, **_kw: [[0.1] * DIM for _ in texts],
    )

    config = PipelineConfig(name="baseline", embedding_dim=DIM)
    built = Pipeline(settings, config, client=client, llm=FakeLLM())
    yield built
    built.close()


@pytest.fixture
def api(pipeline: Pipeline) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    return TestClient(app)


class TestQuery:
    def test_answers_with_validated_citations(self, api: TestClient) -> None:
        response = api.post("/v1/query", json={"question": "Why can't glucose diffuse?"})
        assert response.status_code == 200

        body = response.json()
        assert body["grounded"] is True
        assert body["citations"][0]["chunk_id"] == "biology-p0133-05"
        assert body["citations"][0]["printed_page"] == "121"
        assert body["citations"][0]["license"] == "CC BY 4.0"

    def test_returns_the_context_it_used(self, api: TestClient) -> None:
        """A service returning only prose asks to be trusted; this lets a caller check."""
        body = api.post("/v1/query", json={"question": "q"}).json()
        assert body["context"]
        assert body["context"][0]["chunk_id"] == "biology-p0133-05"

    def test_returns_a_trace(self, api: TestClient) -> None:
        trace = api.post("/v1/query", json={"question": "q"}).json()["trace"]
        assert trace["profile"] == "baseline"
        assert trace["prompt_version"] == "answer-v1"
        assert trace["input_tokens"] == 420
        assert trace["output_tokens"] == 18
        assert trace["list_price_usd"] > 0
        assert {s["name"] for s in trace["stages"]} == {"retrieve", "generate"}

    def test_rejects_an_empty_question(self, api: TestClient) -> None:
        assert api.post("/v1/query", json={"question": ""}).status_code == 422

    def test_rejects_a_missing_question(self, api: TestClient) -> None:
        assert api.post("/v1/query", json={}).status_code == 422

    def test_missing_api_key_is_a_503_not_a_500(
        self, settings: Settings, pipeline: Pipeline, api: TestClient
    ) -> None:
        """The caller should be told the cause, not left to guess."""
        pipeline._llm = None
        pipeline.settings = settings.model_copy(update={"google_api_key": settings.google_api_key})
        object.__setattr__(pipeline.settings, "google_api_key", type(settings.google_api_key)(""))

        response = api.post("/v1/query", json={"question": "q"})
        assert response.status_code == 503
        assert "aistudio.google.com" in response.json()["detail"]


class TestHealth:
    def test_reports_what_is_indexed(self, api: TestClient) -> None:
        """A service pointed at an empty collection is up and useless."""
        body = api.get("/v1/health").json()
        assert body["status"] == "ok"
        assert body["points"] == 1
        assert body["collection"] == COLLECTION


class TestConfig:
    def test_exposes_the_active_profile(self, api: TestClient) -> None:
        body = api.get("/v1/config").json()
        assert body["profile"] == "baseline"
        assert body["use_hybrid"] is False
        assert body["corrective_enabled"] is False
        assert body["top_k_retrieve"] > body["top_k_context"]


class TestSearch:
    """Retrieval only.

    /search exists so a caller can reason over the evidence itself. The
    contract it satisfies is consumed by the agent in the sibling project, so
    these tests pin the field names -- a rename here breaks a separate
    repository silently, and nothing in this one would notice.
    """

    def test_returns_passages(self, api: TestClient) -> None:
        response = api.post("/v1/search", json={"query": "glucose bilayer", "top_k": 3})

        assert response.status_code == 200
        hits = response.json()["results"]
        assert hits
        assert "Glucose" in hits[0]["text"]

    def test_carries_everything_a_citation_needs(self, api: TestClient) -> None:
        """A passage without a checkable locator is unusable for a grounded
        answer, which is the whole point of retrieving it."""
        hit = api.post("/v1/search", json={"query": "glucose"}).json()["results"][0]

        for field in (
            "chunk_id",
            "text",
            "citation",
            "book_slug",
            "book_title",
            "chapter",
            "section",
            "printed_page",
            "source_url",
            "score",
        ):
            assert field in hit, field
        assert hit["printed_page"] == "121"
        assert "5.2. Passive Transport" in hit["citation"]

    def test_makes_no_llm_call(self, api: TestClient, pipeline: Pipeline) -> None:
        """Retrieval must cost nothing and never be rate limited.

        /query spends generation quota; /search must not, or a caller polling
        it would exhaust the same budget the answer path needs.
        """
        calls = []
        pipeline._llm = None
        api.post("/v1/search", json={"query": "glucose"})

        assert pipeline._llm is None
        assert calls == []

    def test_top_k_is_respected(self, api: TestClient) -> None:
        hits = api.post("/v1/search", json={"query": "glucose", "top_k": 1}).json()["results"]

        assert len(hits) <= 1

    def test_book_filter(self, api: TestClient) -> None:
        present = api.post("/v1/search", json={"query": "glucose", "book_slug": "biology"}).json()[
            "results"
        ]
        absent = api.post(
            "/v1/search", json={"query": "glucose", "book_slug": "anatomy-and-physiology"}
        ).json()["results"]

        assert present
        assert absent == []

    def test_no_match_is_an_empty_result_not_an_error(self, api: TestClient) -> None:
        """'The corpus does not cover this' is the finding a caller needs in
        order to say so rather than guess. A 404 would read as this service
        being broken."""
        response = api.post("/v1/search", json={"query": "glucose", "book_slug": "does-not-exist"})

        assert response.status_code == 200
        assert response.json()["results"] == []

    def test_rejects_unknown_fields(self, api: TestClient) -> None:
        """extra=forbid: a caller sending `topk` should be told, not silently
        served the default."""
        response = api.post("/v1/search", json={"query": "glucose", "topk": 3})

        assert response.status_code == 422

    def test_rejects_an_empty_query(self, api: TestClient) -> None:
        assert api.post("/v1/search", json={"query": ""}).status_code == 422
