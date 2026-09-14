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


def test_no_code_or_config_claims_the_thresholds_were_tuned() -> None:
    """The abstention thresholds were deliberately left untuned.

    Four places -- two config files, the policy module and the settings model --
    said they had been tuned or swept against the gold set, and pointed to a
    sweep in docs/EVALUATION.md. No sweep was run, and DESIGN.md has a section
    explaining why not. The policy module, which calls these the most
    consequential numbers in the system, made the false claim in bold.
    """
    claim = re.compile(
        r"(tuned|swept) against the gold set|for the sweep|sweep and its results",
        re.IGNORECASE,
    )
    sources = [*(REPO_ROOT / "src").rglob("*.py"), *(REPO_ROOT / "config").rglob("*.yaml")]
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in sources
        if claim.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"claims a threshold sweep that was never run: {offenders}"

    design = (REPO_ROOT / "docs" / "DESIGN.md").read_text(encoding="utf-8")
    assert (
        "Abstention: thresholds, and why they were not swept" in design
    ), "code and config point readers at this DESIGN.md section"


def test_every_provenance_the_gold_set_uses_is_reported_verbatim() -> None:
    """`Provenance` promises its values are "reported verbatim in docs/EVALUATION.md".

    That promise is the point of the field: the honest answer to "is this
    evaluation real?" should live in the data, and the document should quote the
    data rather than paraphrase it. It named the label on 12 questions and only
    described, in prose, the label on the other 46.
    """
    import json

    goldset = REPO_ROOT / "eval" / "goldset" / "goldset_v1.jsonl"
    used = {
        json.loads(line)["provenance"]
        for line in goldset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert used, "gold set records no provenance"

    evaluation = (REPO_ROOT / "docs" / "EVALUATION.md").read_text(encoding="utf-8")
    missing = sorted(value for value in used if f"`{value}`" not in evaluation)
    assert (
        not missing
    ), f"provenance used by the gold set but not quoted in EVALUATION.md: {missing}"
