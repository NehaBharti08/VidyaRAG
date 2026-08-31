"""Grading dependencies are checked before anything expensive runs.

Regression test for a measured failure: a run invoked without the `eval`
dependency group generated all 58 answers -- roughly twenty minutes of paced,
quota-consuming calls -- and then died on `ModuleNotFoundError: No module named
'openai'` at the very first grading call.

Nothing about that failure required the answers to exist. It was knowable
before the first token.
"""

from __future__ import annotations

import builtins

import pytest

from vidyarag.evaluation.metrics import check_grading_dependencies


def _hide(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Make the named top-level modules unimportable."""
    real = builtins.__import__

    def fake(name: str, *args: object, **kwargs: object) -> object:
        if name in names:
            raise ImportError(f"No module named {name!r}")
        return real(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake)


class TestPreflight:
    def test_passes_when_dependencies_are_present(self) -> None:
        check_grading_dependencies()

    def test_names_the_missing_package(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _hide(monkeypatch, "openai")
        with pytest.raises(RuntimeError, match="openai"):
            check_grading_dependencies()

    def test_names_ragas_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _hide(monkeypatch, "ragas")
        with pytest.raises(RuntimeError, match="ragas"):
            check_grading_dependencies()

    def test_reports_both_rather_than_stopping_at_the_first(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two round trips to discover two missing packages is one too many."""
        _hide(monkeypatch, "ragas", "openai")
        with pytest.raises(RuntimeError) as exc:
            check_grading_dependencies()
        assert "ragas" in str(exc.value)
        assert "openai" in str(exc.value)

    def test_tells_the_user_how_to_fix_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An error naming the cause but not the cure just moves the search."""
        _hide(monkeypatch, "openai")
        with pytest.raises(RuntimeError, match=r"--group eval"):
            check_grading_dependencies()


class TestRunnerCallsItFirst:
    def test_run_evaluation_checks_before_touching_the_pipeline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The check must precede generation, or it saves nothing."""
        from vidyarag.evaluation import runner

        built = False

        def explode(*_a: object, **_k: object) -> object:
            nonlocal built
            built = True
            raise AssertionError("pipeline was built before the dependency check")

        monkeypatch.setattr(runner, "build_pipeline", explode)
        _hide(monkeypatch, "openai")
        with pytest.raises(RuntimeError, match="eval"):
            runner.run_evaluation(profile="baseline")
        assert built is False
