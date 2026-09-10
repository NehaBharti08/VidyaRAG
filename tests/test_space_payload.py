"""Guards on what actually reaches the Hugging Face Space.

Every assertion here corresponds to a deployment that failed in a way the rest
of the suite could not see, because the Space runs code this repository never
executes locally: a generated `app.py`, on an interpreter the platform chooses.
Both failures below cost a build cycle each to diagnose from remote logs.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_SOURCE = REPO_ROOT / "src" / "vidyarag"


def _load_deploy_script() -> ModuleType:
    """Import scripts/deploy_space.py, which is not part of the package."""
    path = REPO_ROOT / "scripts" / "deploy_space.py"
    spec = importlib.util.spec_from_file_location("deploy_space", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_staged_app_is_syntactically_valid() -> None:
    """The generated app.py must compile.

    `app.py` is not shipped verbatim: a sys.path shim is spliced in so the
    package imports without an install step. Prepending it put the shim ahead of
    `from __future__ import annotations`, which Python requires to be the first
    statement in a module. The result was a SyntaxError at startup that the
    Space surfaced only as RUNTIME_ERROR.
    """
    deploy = _load_deploy_script()
    source = (REPO_ROOT / "app" / "app.py").read_text(encoding="utf-8")
    staged = deploy.bootstrapped_app(source)

    compile(staged, "app.py", "exec")

    # The shim is useless if it lands after the import it exists to enable.
    assert staged.index("sys.path.insert") < staged.index("from vidyarag")


# Spellings that exist only on Python 3.11+. The Space runs 3.10, so each of
# these is an AttributeError or ImportError at import time in production.
FORBIDDEN = (
    (re.compile(r"\benum\.StrEnum\b"), "enum.StrEnum"),
    (re.compile(r"^from enum import .*\bStrEnum\b", re.MULTILINE), "from enum import StrEnum"),
    (re.compile(r"^from datetime import .*\bUTC\b", re.MULTILINE), "from datetime import UTC"),
)


@pytest.mark.parametrize("pattern,label", FORBIDDEN, ids=[label for _, label in FORBIDDEN])
def test_shipped_source_avoids_python_311_only_names(pattern: re.Pattern[str], label: str) -> None:
    """Nothing under src/vidyarag may use a 3.11-only standard-library name.

    Hugging Face ZeroGPU images are pinned to Python 3.10 and ignore
    `python_version` in the Space README, so the floor is not negotiable.
    Import the shims from `vidyarag._compat` instead.
    """
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in SHIPPED_SOURCE.rglob("*.py")
        if path.name != "_compat.py" and pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"{label} is unavailable on Python 3.10; found in: {offenders}"


def test_compat_strenum_stringifies_as_its_value() -> None:
    """The backport must format like the real thing.

    A plain `str, Enum` mix-in formats as `Class.MEMBER` under f-strings, not as
    the member value. That difference does not raise -- it silently writes the
    wrong text into prompts, cache keys and JSON, which is the failure mode
    worth a test rather than a comment.
    """
    from vidyarag._compat import StrEnum

    class Colour(StrEnum):
        RED = "red"

    assert f"{Colour.RED}" == "red"
    assert str(Colour.RED) == "red"
    assert Colour.RED == "red"
