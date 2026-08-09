"""Observability: structured logging and per-query tracing.

Built in Phase 2 rather than bolted on later. Tracing is cheap to design in and
expensive to retrofit, and every phase after this one reports its latency and
token cost for free as a result.
"""

from vidyarag.observe.logging import configure_logging, get_logger
from vidyarag.observe.trace import QueryTrace, StageTiming, Usage

__all__ = [
    "QueryTrace",
    "StageTiming",
    "Usage",
    "configure_logging",
    "get_logger",
]
