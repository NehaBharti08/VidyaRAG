"""Guards on documentation that can silently disagree with itself.

The repository keeps the architecture diagram twice: inline in README.md, where
GitHub renders it, and in docs/architecture.mmd, where a Mermaid editor can open
it. Two copies of the same thing drift, and a diagram that no longer matches the
pipeline is worse than no diagram -- it is confidently wrong, and nothing fails
when it happens.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
DIAGRAM = REPO_ROOT / "docs" / "architecture.mmd"

MERMAID_BLOCK = re.compile(r"^```mermaid\n(.*?)^```", re.MULTILINE | re.DOTALL)


def _significant(source: str) -> list[str]:
    """Diagram lines, ignoring `%%` comments and blank lines.

    The standalone file carries a header comment the README block has no room
    for. That difference is intentional; anything else is drift.
    """
    lines = []
    for raw in source.splitlines():
        line = raw.strip()
        if line and not line.startswith("%%"):
            lines.append(line)
    return lines


def test_the_readme_contains_exactly_one_mermaid_diagram() -> None:
    assert len(MERMAID_BLOCK.findall(README.read_text(encoding="utf-8"))) == 1


def test_the_standalone_diagram_matches_the_one_in_the_readme() -> None:
    block = MERMAID_BLOCK.search(README.read_text(encoding="utf-8"))
    assert block is not None
    assert _significant(block.group(1)) == _significant(DIAGRAM.read_text(encoding="utf-8"))


def test_the_standalone_diagram_is_referenced_somewhere() -> None:
    """An unreferenced file is one nobody updates.

    docs/architecture.mmd existed for weeks with no link to it from any document,
    which is how the copy in the README became the only one anyone maintained.
    """
    referencing = [
        path.name
        for path in (README, REPO_ROOT / "docs" / "DESIGN.md")
        if "architecture.mmd" in path.read_text(encoding="utf-8")
    ]
    assert referencing, "docs/architecture.mmd is not linked from README.md or DESIGN.md"
