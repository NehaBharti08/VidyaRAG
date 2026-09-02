"""Publish the demo to a Hugging Face Space.

The Space is a *separate* git repo from this one and carries things this
repository deliberately does not: the built 35 MB index, a flat `app.py` at the
root, and a `requirements.txt` instead of a lockfile.

Two properties this script exists to guarantee:

* **The index ships with the Space.** The demo must not depend on a hosted
  vector store. Free Qdrant Cloud clusters are suspended after a week idle and
  deleted after four, which would kill a portfolio link months after it was last
  touched -- exactly when someone is most likely to click it.
* **The API key never enters a file.** It is read from the local environment and
  set as a Space secret through the API, so it is not written into the Space
  repo, not printed, and not committed here.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Copied into the Space. Everything else -- tests, the eval harness, raw PDFs,
# docs -- is either irrelevant to serving a query or too large for a demo.
PAYLOAD = (
    ("app/requirements.txt", "requirements.txt"),
    ("app/README_space.md", "README.md"),
    ("src/vidyarag", "src/vidyarag"),
    ("config", "config"),
    ("data/index", "data/index"),
    ("LICENSE", "LICENSE"),
    ("ATTRIBUTION.md", "ATTRIBUTION.md"),
)

# The Space runs app.py from its root, so the package must import without an
# install step.
BOOTSTRAP = (
    "import sys\n"
    "from pathlib import Path\n"
    "\n"
    "sys.path.insert(0, str(Path(__file__).parent / 'src'))\n"
    "\n"
)


def stage(destination: Path) -> None:
    """Assemble the Space payload in a temporary directory."""
    for source_rel, target_rel in PAYLOAD:
        source = REPO_ROOT / source_rel
        target = destination / target_rel
        if not source.exists():
            raise FileNotFoundError(f"missing from payload: {source_rel}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(
                source,
                target,
                dirs_exist_ok=True,
                # .lock is Qdrant's exclusive-access marker for the embedded index. It
                # must not travel: copying it fails outright while a local process
                # holds it, and a stale one shipped into the Space could stop the
                # deployed app opening the very index it just downloaded.
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".lock"),
            )
        else:
            shutil.copy2(source, target)

    app_source = (REPO_ROOT / "app" / "app.py").read_text(encoding="utf-8")
    (destination / "app.py").write_text(BOOTSTRAP + app_source, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy the VidyaRAG demo Space.")
    parser.add_argument("--repo-id", default="nehabharti0802/VidyaRAG")
    parser.add_argument(
        "--hardware",
        default="zero-a10g",
        help="ZeroGPU is the only Gradio hardware a free account may host.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Stage only; do not push.")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    try:
        whoami = api.whoami()
    except Exception as exc:  # noqa: BLE001
        print(f"not authenticated: {exc}\nrun `hf auth login` first", file=sys.stderr)
        return 1
    print(f"authenticated as {whoami.get('name')}")

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "space"
        staged.mkdir()
        stage(staged)
        files = [f for f in staged.rglob("*") if f.is_file()]
        size_mb = sum(f.stat().st_size for f in files) / 1024**2
        print(f"staged {len(files)} files, {size_mb:.1f} MB")

        if args.dry_run:
            print("dry run: not pushing")
            return 0

        api.create_repo(
            repo_id=args.repo_id,
            repo_type="space",
            space_sdk="gradio",
            space_hardware=args.hardware,
            exist_ok=True,
        )
        print(f"space ready: https://huggingface.co/spaces/{args.repo_id}")

        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="space",
            folder_path=str(staged),
            commit_message="Deploy VidyaRAG demo",
        )
        print("uploaded")

    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        print("WARNING: GOOGLE_API_KEY not found locally; set it in Space settings")
    else:
        api.add_space_secret(repo_id=args.repo_id, key="GOOGLE_API_KEY", value=key)
        print("GOOGLE_API_KEY set as a Space secret (not written to any file)")

    for name, value in (("VIDYARAG_PROFILE", "guarded"), ("QDRANT_MODE", "embedded")):
        api.add_space_variable(repo_id=args.repo_id, key=name, value=value)
    print("space variables set")
    print(f"\nlive shortly at https://huggingface.co/spaces/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
