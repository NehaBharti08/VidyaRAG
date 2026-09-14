"""Guards on the container and version contracts.

Everything here corresponds to a defect that sat in the repository, green, for
weeks -- because the Docker workflow built the image and imported the package
but never started the server, and nothing compared the versions the project
reports about itself. The CI Docker job now runs the container for real, but it
only triggers when Docker-related files change; these checks run on every
commit and fail in seconds.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")


def _healthcheck_command() -> str:
    """The HEALTHCHECK instruction, with line continuations joined."""
    joined = DOCKERFILE.replace("\\\n", " ")
    match = re.search(r"^HEALTHCHECK\b(.*)$", joined, re.MULTILINE)
    assert match, "Dockerfile has no HEALTHCHECK"
    return match.group(1)


def test_healthcheck_asks_the_server_instead_of_opening_the_index() -> None:
    """The healthcheck must not compete with uvicorn for the index lock.

    It ran `vidyarag health`, which opens the embedded Qdrant index. uvicorn
    takes that index's exclusive lock at startup, so the check failed with
    "already accessed by another instance of Qdrant client" and the container
    would have reported unhealthy for its whole life while serving correctly.
    """
    command = _healthcheck_command()
    assert "vidyarag.cli" not in command, "healthcheck opens the index uvicorn holds"
    assert "/v1/health" in command, "healthcheck should probe the running server"


def test_documented_index_mount_is_not_read_only() -> None:
    """Embedded Qdrant cannot open a read-only index.

    It locks by opening `<index>/.lock` with mode "r+", which needs write
    access even when the file exists. The Dockerfile documented `:ro`, which
    fails at startup.
    """
    documented = [line for line in DOCKERFILE.splitlines() if "docker run" in line]
    assert documented, "the Dockerfile should document how to run the image"
    for line in documented:
        assert ":ro" not in line, f"read-only index mount cannot work: {line.strip()}"


def test_the_api_reports_the_package_version() -> None:
    """`vidyarag version`, /docs and the release must name the same version.

    They named three: 0.1.0, 0.3.0 and v1.2.0.
    """
    from vidyarag import __version__
    from vidyarag.api.main import create_app

    assert create_app().version == __version__


def test_pyproject_takes_its_version_from_the_package() -> None:
    """A second literal in pyproject.toml is how the versions drifted apart."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project = pyproject.split("[project]", 1)[1].split("\n[", 1)[0]
    assert re.search(r'^dynamic\s*=\s*\[[^\]]*"version"', project, re.MULTILINE)
    assert not re.search(r"^version\s*=", project, re.MULTILINE)
    assert 'path = "src/vidyarag/__init__.py"' in pyproject


def _stage(name: str) -> str:
    """Text of one build stage, from its FROM line up to the next FROM."""
    for part in re.split(r"^FROM\s", DOCKERFILE, flags=re.MULTILINE)[1:]:
        if re.search(rf"\bAS\s+{name}\b", part.splitlines()[0], re.IGNORECASE):
            return part
    raise AssertionError(f"Dockerfile has no build stage named {name!r}")


def test_the_venv_is_built_at_the_path_it_runs_from() -> None:
    """Otherwise every console script's shebang names a missing interpreter.

    The venv was built in /build and copied to /app. `uvicorn` kept the builder's
    interpreter in its shebang, so CMD died with "exec /app/.venv/bin/uvicorn: no
    such file or directory" -- the image could never serve a request. The import
    check in CI kept passing because `python -c` never reads a shebang.
    """
    workdirs = re.findall(r"^WORKDIR\s+(\S+)", _stage("builder"), re.MULTILINE)
    assert workdirs, "builder stage sets no WORKDIR"
    built = f"{workdirs[-1].rstrip('/')}/.venv"

    copy = re.search(
        r"^COPY\s+--from=builder\b.*?\s(\S+/\.venv)\s+(\S+)\s*$", DOCKERFILE, re.MULTILINE
    )
    assert copy, "runtime stage should copy the venv out of the builder"
    source, destination = copy.group(1), copy.group(2).rstrip("/")

    assert source == built, f"copies {source} but the venv was built at {built}"
    assert source == destination, (
        f"venv built at {source} but run from {destination}: console-script "
        "shebangs would name an interpreter that does not exist"
    )


def test_the_image_points_config_resolution_at_the_config_it_ships() -> None:
    """An installed package cannot find config/ by walking up from its own file.

    Installed non-editable, the package lives in site-packages, so the path it
    derived for the repo root landed inside the venv and the server refused to
    start with "Unknown profile 'guarded'. Available: (none)".
    """
    runtime = _stage("runtime")
    env = re.search(r"\bVIDYARAG_CONFIG_DIR=(\S+)", runtime)
    assert env, "runtime stage must set VIDYARAG_CONFIG_DIR"
    copy = re.search(r"^COPY\b.*\sconfig/\s+(\S+)\s*$", runtime, re.MULTILINE)
    assert copy, "runtime stage should copy config/ into the image"
    assert env.group(1).rstrip("/") == copy.group(1).rstrip(
        "/"
    ), f"config is copied to {copy.group(1)} but resolved from {env.group(1)}"
