"""The query pipeline.

One config-driven orchestrator wiring every stage: retrieve, generate, cite.
Later phases add hybrid fusion, reranking, a corrective loop and guardrails --
each switched on by profile flags rather than by a different code path, so the
thing being measured in an ablation is the flag and nothing else.

Phase 2 is deliberately the shortest version of this that produces a grounded,
cited answer. It is the frozen baseline; every later number is a delta from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient

from vidyarag.correct.loop import run_corrective_loop
from vidyarag.correct.policy import CorrectivePolicy
from vidyarag.generate.answer import GeneratedAnswer, generate_answer
from vidyarag.generate.citations import Citation
from vidyarag.guard import screen_context, screen_input
from vidyarag.guard.input_guard import REFUSAL as INPUT_REFUSAL
from vidyarag.llm.provider import get_gemini_client
from vidyarag.observe.trace import QueryTrace
from vidyarag.retrieve.decompose import decompose, retrieve_decomposed
from vidyarag.retrieve.dense import RetrievedChunk, retrieve_dense
from vidyarag.retrieve.rerank import rank_movement, rerank
from vidyarag.settings import PipelineConfig, Settings
from vidyarag.store.client import build_client


@dataclass(frozen=True, slots=True)
class Answer:
    """The complete result of one query."""

    question: str
    text: str
    citations: list[Citation]
    retrieved: list[RetrievedChunk]
    trace: QueryTrace
    grounded: bool

    @property
    def context_used(self) -> list[RetrievedChunk]:
        """The chunks actually placed in the prompt."""
        return self.retrieved


class Pipeline:
    """Answers questions against the indexed corpus.

    Holds its clients rather than rebuilding them per query: the Qdrant client
    opens the on-disk index, which is expensive to reopen and, in embedded
    mode, cannot be opened twice concurrently.
    """

    def __init__(
        self,
        settings: Settings,
        config: PipelineConfig,
        *,
        client: QdrantClient | None = None,
        llm: Any | None = None,
    ) -> None:
        self.settings = settings
        self.config = config
        self.client = client if client is not None else build_client(settings)
        self._llm = llm

    @property
    def llm(self) -> Any:
        """The Gemini client, constructed on first use.

        Lazy so that retrieval-only work -- and the test suite -- never needs a
        key. Ingestion and search are genuinely free of credentials; only
        answering costs something.
        """
        if self._llm is None:
            self._llm = get_gemini_client(self.settings.google_api_key.get_secret_value())
        return self._llm

    def retrieve(self, question: str, trace: QueryTrace) -> list[RetrievedChunk]:
        """Fetch candidates and narrow them to the context budget.

        Every stage after the first is switched on by a profile flag rather than
        by a different code path, so an ablation changes exactly one thing and
        the measurement can be attributed to it.
        """
        sub_questions: list[str] = []
        if self.config.retrieval.use_decomposition:
            with trace.stage("decompose"):
                split = decompose(self.llm, question, model=self.config.generation_model)
            sub_questions = split.sub_questions
            trace.sub_questions = list(sub_questions)

        with trace.stage("retrieve"):
            if sub_questions:
                candidates = retrieve_decomposed(
                    self.client,
                    sub_questions,
                    collection=self.settings.qdrant_collection,
                    embedding_model=self.config.embedding_model,
                    limit=self.config.retrieval.top_k_retrieve,
                )[: self.config.retrieval.top_k_retrieve]
            else:
                candidates = retrieve_dense(
                    self.client,
                    question,
                    collection=self.settings.qdrant_collection,
                    embedding_model=self.config.embedding_model,
                    limit=self.config.retrieval.top_k_retrieve,
                )
        # Recorded before narrowing: retrieval metrics score the whole candidate
        # pool, and the gap between that and what reaches the prompt is the
        # thing reranking exists to close.
        trace.retrieved_chunk_ids = [c.chunk_id for c in candidates]

        if self.config.retrieval.use_reranker and candidates:
            with trace.stage("rerank"):
                reordered = rerank(
                    question,
                    candidates,
                    model_name=self.config.retrieval.reranker_model,
                )
            trace.rerank = rank_movement(candidates, reordered)
            candidates = reordered

        trace.ranked_chunk_ids = [c.chunk_id for c in candidates]
        context = candidates[: self.config.retrieval.top_k_context]

        # Screened after narrowing rather than over the whole pool: only what
        # reaches the prompt can influence the model, and scanning 20 passages
        # to protect 5 is work spent on chunks that were never going to be read.
        if self.config.guardrails.check_retrieved_context and context:
            with trace.stage("guard_context"):
                screened = screen_context(context)
            trace.guard_context = screened.as_dict()
            context = screened.kept

        return context

    def answer(self, question: str) -> Answer:
        """Answer one question end to end.

        Args:
            question: The user's question.

        Returns:
            An :class:`Answer` with validated citations and a full trace.
        """
        trace = QueryTrace(query=question, profile=self.config.name)

        # Screened before retrieval on purpose: a blocked question should cost
        # nothing. Embedding and searching first would spend the work anyway,
        # and on a rate-limited free tier that is quota an attacker burns for
        # free.
        if self.config.guardrails.check_user_input:
            with trace.stage("guard_input"):
                verdict = screen_input(question)
            if verdict.blocked:
                trace.guard_input = {"blocked": True, "categories": verdict.categories}
                return Answer(
                    question=question,
                    text=INPUT_REFUSAL,
                    citations=[],
                    retrieved=[],
                    trace=trace,
                    grounded=True,
                )

        if self.config.corrective.enabled:
            return self._answer_corrective(question, trace)

        context = self.retrieve(question, trace)
        generated: GeneratedAnswer = generate_answer(
            self.llm,
            question,
            context,
            model=self.config.generation_model,
            temperature=self.config.temperature,
            trace=trace,
        )

        return Answer(
            question=question,
            text=generated.text,
            citations=generated.citations,
            retrieved=context,
            trace=trace,
            grounded=generated.grounded,
        )

    def _answer_corrective(self, question: str, trace: QueryTrace) -> Answer:
        """Answer through the bounded self-check loop.

        The loop owns control flow; this method supplies the two operations it
        needs and keeps hold of the artefacts it does not care about -- the
        retrieved chunks and the resolved citations belonging to whichever
        attempt was finally accepted.
        """
        cfg = self.config.corrective
        policy = CorrectivePolicy(
            accept_threshold=cfg.accept_threshold,
            abstain_threshold=cfg.abstain_threshold,
            max_attempts=cfg.max_attempts,
        )
        # Populated by the closures below; the loop returns text and a verdict,
        # not the objects needed to build an Answer.
        last: dict[str, Any] = {"context": [], "generated": None}

        def retrieve(query: str) -> list[RetrievedChunk]:
            context = self.retrieve(query, trace)
            last["context"] = context
            return context

        def generate(_question: str, context: list[RetrievedChunk]) -> tuple[str, list[str]]:
            generated = generate_answer(
                self.llm,
                question,
                context,
                model=self.config.generation_model,
                temperature=self.config.temperature,
                trace=trace,
            )
            last["generated"] = generated
            return generated.text, [c.text for c in context]

        with trace.stage("corrective"):
            outcome = run_corrective_loop(
                question=question,
                generate=generate,
                retrieve=retrieve,
                llm=self.llm,
                grader_model=self.config.grader_model,
                policy=policy,
            )

        trace.attempts = outcome.attempt_count
        trace.abstained = outcome.abstained
        trace.corrective = outcome.as_dict()

        generated_answer: GeneratedAnswer | None = last["generated"]
        context_used: list[RetrievedChunk] = last["context"]

        # An abstention cites nothing. Carrying the rejected draft's citations
        # would attach page references to a refusal, implying the corpus
        # supports something the system just said it does not.
        if outcome.abstained:
            return Answer(
                question=question,
                text=outcome.answer,
                citations=[],
                retrieved=context_used,
                trace=trace,
                grounded=False,
            )

        return Answer(
            question=question,
            text=outcome.answer,
            citations=generated_answer.citations if generated_answer else [],
            retrieved=context_used,
            trace=trace,
            grounded=generated_answer.grounded if generated_answer else False,
        )

    def close(self) -> None:
        """Release the vector store handle.

        Embedded Qdrant holds a lock on the index directory, so a long-lived
        process that never closes prevents any other process from opening it.
        """
        self.client.close()


def build_pipeline(
    settings: Settings | None = None,
    config: PipelineConfig | None = None,
) -> Pipeline:
    """Construct a pipeline from the environment and the active profile."""
    from vidyarag.settings import load_pipeline_config

    resolved_settings = settings or Settings()
    resolved_config = config or load_pipeline_config(resolved_settings.profile)
    return Pipeline(resolved_settings, resolved_config)
