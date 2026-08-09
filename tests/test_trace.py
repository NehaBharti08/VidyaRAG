"""Tracing tests.

Every phase after this one reports latency and cost from these primitives, so
an error here would silently misreport every later result.
"""

from __future__ import annotations

import time

from vidyarag.observe.trace import QueryTrace, Usage, price_call


class TestPricing:
    def test_prices_a_call_from_published_rates(self) -> None:
        # 1M in + 1M out at flash rates = 0.30 + 2.50
        assert price_call("gemini-3.5-flash", 1_000_000, 1_000_000) == 2.80

    def test_lite_is_cheaper_than_flash(self) -> None:
        assert price_call("gemini-3.5-flash-lite", 1000, 1000) < price_call(
            "gemini-3.5-flash", 1000, 1000
        )

    def test_unknown_model_is_estimated_not_free(self) -> None:
        """A silently free-looking call is worse than a rough estimate."""
        assert price_call("some-future-model", 10_000, 10_000) > 0

    def test_zero_tokens_costs_nothing(self) -> None:
        assert price_call("gemini-3.5-flash", 0, 0) == 0.0


class TestUsage:
    def test_totals_tokens(self) -> None:
        usage = Usage(model="gemini-3.5-flash", input_tokens=100, output_tokens=25)
        assert usage.total_tokens == 125

    def test_exposes_list_price(self) -> None:
        usage = Usage(model="gemini-3.5-flash", input_tokens=1_000_000, output_tokens=0)
        assert usage.list_price_usd == 0.30


class TestQueryTrace:
    def test_stage_context_manager_records_time(self) -> None:
        trace = QueryTrace(query="q")
        with trace.stage("retrieve"):
            time.sleep(0.01)
        assert trace.stage_ms("retrieve") >= 5

    def test_stage_records_even_when_the_block_raises(self) -> None:
        """A failed query is exactly when knowing where time went matters."""
        trace = QueryTrace(query="q")
        try:
            with trace.stage("generate"):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert trace.stage_ms("generate") > 0

    def test_totals_across_stages(self) -> None:
        trace = QueryTrace(query="q")
        trace.record("retrieve", 10.0)
        trace.record("generate", 30.0)
        assert trace.total_ms == 40.0

    def test_accumulates_usage_across_calls(self) -> None:
        trace = QueryTrace(query="q")
        trace.add_usage("gemini-3.5-flash", 100, 20)
        trace.add_usage("gemini-3.5-flash-lite", 50, 10, purpose="grading")
        assert trace.input_tokens == 150
        assert trace.output_tokens == 30
        assert trace.list_price_usd > 0

    def test_same_stage_can_be_timed_more_than_once(self) -> None:
        """The corrective loop retries retrieval; both attempts must count."""
        trace = QueryTrace(query="q")
        trace.record("retrieve", 10.0)
        trace.record("retrieve", 15.0)
        assert trace.stage_ms("retrieve") == 25.0

    def test_summary_names_stages_and_cost(self) -> None:
        trace = QueryTrace(query="q")
        trace.record("retrieve", 12.0)
        trace.add_usage("gemini-3.5-flash", 100, 20)
        summary = trace.summary()
        assert "retrieve=12ms" in summary
        assert "100+20 tok" in summary
        assert "list price" in summary

    def test_empty_trace_is_zeroed_not_broken(self) -> None:
        trace = QueryTrace(query="q")
        assert trace.total_ms == 0
        assert trace.list_price_usd == 0.0
        assert trace.summary()
