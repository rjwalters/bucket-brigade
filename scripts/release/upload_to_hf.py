"""Operator script: upload a frozen baseline release bundle to HuggingFace.

This is the **one** entry point that pushes artifacts to a public
HuggingFace repo. It is intentionally NOT invoked by any automation in
this repo — every upload requires a human to pass ``--confirm``
explicitly.

Usage
-----

Dry-run (default, no network writes):

::

    python -m scripts.release.upload_to_hf \\
        --source-dir bucket_brigade/baselines/release/local

Real upload (requires HF token in ``$HF_TOKEN`` or
``~/.cache/huggingface/token``):

::

    python -m scripts.release.upload_to_hf \\
        --source-dir bucket_brigade/baselines/release/local \\
        --confirm

Override the destination repo:

::

    python -m scripts.release.upload_to_hf \\
        --source-dir <staging> \\
        --repo-id my-org/my-baselines \\
        --confirm

See ``docs/RELEASE.md`` for the full release workflow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bucket_brigade.baselines.release import (
    DEFAULT_HUGGINGFACE_REPO,
    MANIFEST_FILENAME,
)
from bucket_brigade.baselines.release.hub import upload_release


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description=(
            "Upload a Bucket Brigade frozen-baseline release bundle to "
            "HuggingFace. Dry-run by default; pass --confirm to push."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help=(
            f"Directory containing {MANIFEST_FILENAME} plus all referenced "
            "artifact files. See bucket_brigade/baselines/release/local/README.md."
        ),
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_HUGGINGFACE_REPO,
        help=(f"HuggingFace repo ID to upload to. Default: {DEFAULT_HUGGINGFACE_REPO}"),
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help=(
            "Commit message for the HF repo. Defaults to "
            '"Upload bucket-brigade baselines <release_version>".'
        ),
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help=(
            "REQUIRED to actually push. Without this flag the script "
            "runs in dry-run mode (no network writes)."
        ),
    )
    args = parser.parse_args(argv)

    if not args.source_dir.exists():
        print(
            f"ERROR: source-dir {args.source_dir} does not exist.",
            file=sys.stderr,
        )
        return 2

    try:
        result = upload_release(
            source_dir=args.source_dir,
            repo_id=args.repo_id,
            commit_message=args.commit_message,
            dry_run=not args.confirm,
        )
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.confirm:
        print(f"Upload complete: {result}")
    else:
        print(
            "\nDry-run complete. Re-run with --confirm to actually push.\n"
            "Be sure $HF_TOKEN (or ~/.cache/huggingface/token) is set "
            "with write access to the target repo."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
