"""
Disk-space precheck helper for long-running experiment scripts.

These scripts (e.g., ``compute_nash.py``, ``compute_nash_v2.py``) can run for
tens of minutes performing CPU-expensive simulations, and only at the very end
write a small JSON artifact (~1.5–2 KB) to the output directory. When the
filesystem is full at write time, all of that compute is wasted with an
``OSError: [Errno 28] No space left on device`` traceback.

This helper provides a small startup-time check: resolve the output directory
to its existing parent, ``os.statvfs`` the filesystem, and abort early with a
clear error message if the free space is below a configurable threshold.

The default threshold is 100 MiB — overkill for the artifact itself (KB-scale)
but enough headroom for ``tee``-style log files and other co-located output
that accumulates during a sweep. See issue #269.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Default minimum free space (in MiB) required to proceed. Conservative to
# leave headroom for log files via `tee` and other co-located outputs.
DEFAULT_MIN_FREE_MIB = 100

# Conversion constant.
BYTES_PER_MIB = 1024 * 1024


def _resolve_existing_parent(path: Path) -> Path:
    """Return the nearest existing ancestor of ``path``.

    ``os.statvfs`` requires an existing path. Output directories are typically
    created lazily by the calling script, so we walk up to the first existing
    parent (which lives on the same filesystem as the eventual target).
    """
    candidate = path.resolve()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            # Reached the filesystem root and it does not exist (extremely
            # unlikely on a sane system, but guard anyway).
            return Path("/")
        candidate = parent
    return candidate


def check_free_space(
    output_dir: Path,
    min_free_mib: int = DEFAULT_MIN_FREE_MIB,
) -> None:
    """Abort the process if free space on ``output_dir``'s filesystem is low.

    The function uses ``os.statvfs`` against the nearest existing ancestor of
    ``output_dir`` (so it works before the directory is created). On
    insufficient space it writes a clear error message to stderr and calls
    ``sys.exit(1)`` — BEFORE any compute starts in the caller.

    Args:
        output_dir: The directory where the script intends to write outputs.
            May or may not exist yet.
        min_free_mib: Minimum required free space, in MiB. Defaults to 100.

    Exits:
        With status 1 if free bytes are below ``min_free_mib`` MiB, or if
        the filesystem cannot be statted for some reason.
    """
    output_dir = Path(output_dir)
    target = _resolve_existing_parent(output_dir)

    try:
        stats = os.statvfs(target)
    except OSError as exc:
        print(
            f"ERROR: could not stat filesystem for '{target}': {exc}. "
            "Aborting before compute.",
            file=sys.stderr,
        )
        sys.exit(1)

    free_bytes = stats.f_bavail * stats.f_frsize
    required_bytes = min_free_mib * BYTES_PER_MIB

    if free_bytes < required_bytes:
        free_mib = free_bytes / BYTES_PER_MIB
        print(
            f"ERROR: only {free_mib:.1f} MiB free on '{target}'; "
            f"need at least {min_free_mib} MiB. Aborting before compute.",
            file=sys.stderr,
        )
        sys.exit(1)
