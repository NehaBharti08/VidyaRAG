"""Generation tests.

Gemini is stubbed: what needs testing is the behaviour around the model call --
that empty retrieval never reaches it, that invented citations are stripped,
and that usage lands in the trace.
"""

from __future__ import annotations

from typing import Any

import pytest

from vidyarag.generate.answer import generate_answer
from vidyarag.generate.prompts import NO_CONTEXT_ANSWER, build_answer_prompt
from vidyarag.observe.trace import QueryTrace
from vidyarag.retrieve.dense import RetrievedChunk


def _chunk(n: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"biology-p{n:04d}-00",
        score=0.9,
        text=f"Passage {n} body.",
        citation=f"Biology, {n}.1. Section, p.{100 + n}",
        book_slug="biology",
        book_title="Biology",
        chapter=f"Chapter {n}",
        section=f"{n}.1. Section",
        page_start=100 + n,
        page_end=100 + n,
        printed_page=str(88 + n),
        license_name="CC BY 4.0",
        source_url="https://openstax.org/details/books/biology",
    )


CHUNKS = [_chunk(i) for i in range(1, 4)]


class FakeResponse:
    def __init__(self, text: str, prompt_tokens: int = 500, output_tokens: int = 40) -> None:
        self.text = text
        self.usage_metadata = type(
            "Usage",
            (),
            {"prompt_token_count": prompt_tokens, "candidates_token_count": output_tokens},
        )()


class FakeModels:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._response


class FakeClient:
    def __init__(self, text: str = "Glucose is polar [1].") -> None:
        self.models = FakeModels(FakeResponse(text))


class TestPrompt:
    def test_carries_a_version(self) -> None:
        """An unversioned prompt makes an improvement unattributable."""
        assert build_answer_prompt("q", CHUNKS).version == "answer-v1"

    def test_includes_numbered_context_and_question(self) -> None:
        prompt = build_answer_prompt("Why is glucose polar?", CHUNKS)
        assert "[1] Biology, 1.1. Section, p.101" in prompt.user
        assert "Why is glucose polar?" in prompt.user

    def test_system_prompt_forbids_outside_knowledge(self) -> None:
        prompt = build_answer_prompt("q", CHUNKS)
        assert "ONLY" in prompt.system


class TestGenerateAnswer:
    def test_returns_text_and_citations(self) -> None:
        answer = generate_answer(FakeClient(), "q", CHUNKS, model="gemini-3.5-flash")
        assert "[1]" in answer.text
        assert [c.marker for c in answer.citations] == [1]
        assert answer.grounded

    def test_empty_context_never_reaches_the_model(self) -> None:
        """Asking a model to answer with no context is asking it to hallucinate."""
        client = FakeClient()
        answer = generate_answer(client, "q", [], model="gemini-3.5-flash")
        assert answer.text == NO_CONTEXT_ANSWER
        assert answer.citations == []
        assert not answer.grounded
        assert client.models.calls == []

    def test_strips_citations_the_model_invented(self) -> None:
        client = FakeClient("Real [1] and invented [9].")
        answer = generate_answer(client, "q", CHUNKS, model="gemini-3.5-flash")
        assert "[9]" not in answer.text
        assert [c.marker for c in answer.citations] == [1]

    def test_grouped_citations_all_resolve(self) -> None:
        client = FakeClient("Supported by [1, 3].")
        answer = generate_answer(client, "q", CHUNKS, model="gemini-3.5-flash")
        assert [c.marker for c in answer.citations] == [1, 3]

    def test_uncited_answer_is_not_grounded(self) -> None:
        client = FakeClient("A confident answer with no citation at all.")
        answer = generate_answer(client, "q", CHUNKS, model="gemini-3.5-flash")
        assert not answer.grounded

    def test_empty_completion_is_a_failure_not_an_answer(self) -> None:
        client = FakeClient("")
        answer = generate_answer(client, "q", CHUNKS, model="gemini-3.5-flash")
        assert answer.text == NO_CONTEXT_ANSWER
        assert not answer.grounded

    def test_records_usage_and_timing_in_the_trace(self) -> None:
        trace = QueryTrace(query="q")
        generate_answer(FakeClient(), "q", CHUNKS, model="gemini-3.5-flash", trace=trace)
        assert trace.input_tokens == 500
        assert trace.output_tokens == 40
        assert trace.stage_ms("generate") >= 0
        assert trace.prompt_version == "answer-v1"
        assert trace.cited_chunk_ids == ["biology-p0001-00"]

    @pytest.mark.parametrize("temperature", [0.0, 0.7])
    def test_passes_generation_config_through(self, temperature: float) -> None:
        client = FakeClient()
        generate_answer(client, "q", CHUNKS, model="gemini-3.5-flash", temperature=temperature)
        config = client.models.calls[0]["config"]
        assert config["temperature"] == temperature
        assert config["system_instruction"]
