"""Re-judging a committed run.

The command that rebuilds abstention verdicts in place. It exists because the
alternative to recomputing a wrong number is disowning it, and a result that
cannot be recomputed is an anecdote.

What has to hold: a refusal the judge recognises is counted, a refusal the
pipeline signalled structurally is not re-judged at all, a judgement that could
not be read is reported as unmeasured rather than as "answered", and a question
newly recognised as a refusal stops contributing a faithfulness score it should
never have had.
"""

from __future__ import annotations

from typing import Any

import pytest

from vidyarag.correct.loop import ABSTENTION_TEXT
from vidyarag.evaluation.goldset import GoldQuestion, Provenance, QuestionType
from vidyarag.evaluation.rejudge import rejudge_run
from vidyarag.evaluation.runner import EvalRun, SampleResult


class FakeChoice:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.message = type("Msg", (), {"content": content})()
        self.finish_reason = finish_reason


class ScriptedClient:
    """Replays one label per answer text, and counts the calls it receives."""

    def __init__(self, labels: dict[str, str], finish_reason: str = "stop") -> None:
        self.labels = labels
        self.finish_reason = finish_reason
        self.calls: list[str] = []
        outer = self

        class _Completions:
            async def create(self, **kwargs: Any) -> Any:
                prompt = kwargs["messages"][0]["content"]
                outer.calls.append(prompt)
                label = next(
                    (v for k, v in outer.labels.items() if k in prompt),
                    "ANSWERED",
                )
                return type("Resp", (), {"choices": [FakeChoice(label, outer.finish_reason)]})()

        self.chat = type("Chat", (), {"completions": _Completions()})()


def _question(qid: str, answerable: bool) -> GoldQuestion:
    return GoldQuestion(
        id=qid,
        question=f"question {qid}",
        type=QuestionType.FACTUAL if answerable else QuestionType.UNANSWERABLE,
        provenance=Provenance.LLM_DRAFTED_HUMAN_VERIFIED,
        reference="a reference" if answerable else None,
        gold_chunk_ids=["c1"] if answerable else [],
    )


def _sample(qid: str, answer: str, **kwargs: Any) -> SampleResult:
    data: dict[str, Any] = {
        "id": qid,
        "question": f"question {qid}",
        "type": kwargs.pop("type", QuestionType.FACTUAL),
        "answer": answer,
        "abstained": False,
    }
    data.update(kwargs)
    return SampleResult.model_validate(data)


def _run(samples: list[SampleResult]) -> EvalRun:
    return EvalRun(
        run_id="test",
        created_at="2026-09-20T00:00:00+00:00",
        profile="baseline",
        config={},
        goldset_path="goldset_v1.jsonl",
        goldset_sha256="deadbeef",
        goldset_counts={},
        generation_model="gemini-3.5-flash-lite",
        grader_model="gemini-3.1-flash-lite",
        embedding_model="BAAI/bge-base-en-v1.5",
        python_version="3.11",
        samples=samples,
    )


GOLDSET = [_question("fact-001", True), _question("unans-001", False)]


async def _rejudge(run: EvalRun, client: Any) -> EvalRun:
    return await rejudge_run(
        run,
        client=client,
        judge_model="gemini-3.1-flash-lite",
        goldset=GOLDSET,
        rate=6000.0,
        use_cache=False,
    )


class TestRejudging:
    async def test_a_recognised_refusal_becomes_an_abstention(self) -> None:
        run = _run(
            [
                _sample("fact-001", "Glucose is polar [1]."),
                _sample(
                    "unans-001",
                    "The provided passages do not contain that information.",
                    type=QuestionType.UNANSWERABLE,
                ),
            ]
        )
        client = ScriptedClient({"unans-001": "REFUSED", "fact-001": "ANSWERED"})
        await _rejudge(run, client)

        assert run.abstention["unanswerable_abstained"] == 1
        assert run.abstention["recall"] == pytest.approx(1.0)
        assert run.abstention["answerable_abstained"] == 0

    async def test_a_structural_abstention_costs_no_judge_call(self) -> None:
        """The pipeline already said so. Asking a model to confirm it is waste."""
        run = _run([_sample("unans-001", ABSTENTION_TEXT, type=QuestionType.UNANSWERABLE)])
        client = ScriptedClient({})
        await _rejudge(run, client)

        assert client.calls == []
        assert run.samples[0].abstained is True
        assert run.samples[0].abstention_judge["source"] == "structural"

    async def test_a_truncated_judgement_is_unmeasured(self) -> None:
        run = _run([_sample("fact-001", "Glucose is polar [1].")])
        client = ScriptedClient({"fact-001": "REF"}, finish_reason="length")
        await _rejudge(run, client)

        assert run.abstention["unmeasured"] == 1
        assert run.samples[0].abstention_judge["measured"] is False

    async def test_a_new_refusal_drops_its_faithfulness_score(self) -> None:
        """A refusal has no faithfulness. Keeping the old score would average a
        refusal into a metric that excludes refusals everywhere else."""
        run = _run(
            [
                _sample(
                    "fact-001",
                    "The passages do not say.",
                    ragas={"faithfulness": 0.95, "answer_relevancy": 0.4},
                ),
                _sample(
                    "unans-001",
                    "Glucose is polar [1].",
                    type=QuestionType.UNANSWERABLE,
                    ragas={"faithfulness": 0.85, "answer_relevancy": 0.8},
                ),
            ]
        )
        run.aggregates = {"faithfulness": 0.90, "graded_samples": 2.0}
        client = ScriptedClient({"fact-001": "REFUSED", "unans-001": "ANSWERED"})
        await _rejudge(run, client)

        assert run.samples[0].ragas == {}
        assert run.aggregates["graded_samples"] == 1.0
        assert run.aggregates["faithfulness"] == pytest.approx(0.85)

    async def test_an_errored_sample_is_left_alone(self) -> None:
        run = _run([_sample("fact-001", "", error="429 quota")])
        client = ScriptedClient({})
        await _rejudge(run, client)

        assert client.calls == []
        assert run.samples[0].abstention_judge == {}
