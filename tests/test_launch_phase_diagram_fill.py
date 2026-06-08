"""Tests for ``experiments/scripts/launch_phase_diagram_fill.sh``.

The script is the operator-launch wrapper added for issue #390 (fill the
missing β=0.1 row + c=1.0/2.0 cells of the heterogeneous Nash phase-diagram
preview). Because it shells out to ``ssh`` to bootstrap a remote tmux
session, we cannot exercise the live-run path in unit tests. The
``--dry-run`` flag exists exactly so the wiring (arg parsing, host
resolution, driver-command synthesis, session-name compaction) can be
asserted without touching the network.

These tests cover the three operator-facing failure modes plus the canonical
gap-fill invocations from ``experiments/nash/phase_diagram/LAUNCH_RUNBOOK.md``.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "experiments" / "scripts" / "launch_phase_diagram_fill.sh"


def _run(args: list[str], env: dict | None = None, cwd: Path | None = None):
    """Invoke the launch script with the given args; return CompletedProcess."""
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env=run_env,
        cwd=cwd or REPO_ROOT,
        check=False,
    )


def test_script_exists_and_is_executable() -> None:
    """The launch script must be present and have the +x bit set."""
    assert SCRIPT.exists(), f"launch script missing: {SCRIPT}"
    assert os.access(SCRIPT, os.X_OK), (
        f"launch script not executable: {SCRIPT} — chmod +x it before commit"
    )


def test_help_flag_prints_usage_and_exits_zero() -> None:
    """``--help`` must succeed without trying to ssh anywhere."""
    result = _run(["--help"])
    assert result.returncode == 0, result.stderr
    assert "Usage" in result.stdout
    # Spot-check that the canonical flags appear in the doc string.
    for flag in (
        "--host",
        "--c-values",
        "--beta-values",
        "--kappa-values",
        "--dry-run",
    ):
        assert flag in result.stdout, f"--help is missing documentation for {flag}"


def test_missing_host_with_no_env_errors_cleanly() -> None:
    """Without --host and without a .env, the script must fail with exit 3."""
    # Run in a temp dir with no .env so resolve_host_from_env returns empty.
    # The script computes REPO_ROOT relative to its own location, so to force
    # the "no .env" branch we hand it a host-free environment by pointing at
    # a tmp cwd... but REPO_ROOT is hard-coded to SCRIPT_DIR/../.. so the .env
    # lookup goes through the real repo root regardless of cwd. We instead
    # invoke a fake script that wraps the real one with REPO_ROOT overridden
    # — too brittle. Simpler: just confirm "no host specified" path produces
    # the documented diagnostic when --host is missing AND .env has no
    # COMPUTE_HOST_* entries. We do that by temporarily renaming any existing
    # .env, running, and restoring.
    env_path = REPO_ROOT / ".env"
    backup = None
    if env_path.exists():
        backup = env_path.read_bytes()
        env_path.unlink()
    try:
        result = _run(["--c-values", "2.0", "--dry-run"])
    finally:
        if backup is not None:
            env_path.write_bytes(backup)

    assert result.returncode == 3, (
        f"expected exit 3 (no host), got {result.returncode}: {result.stderr}"
    )
    assert "no host specified" in result.stderr.lower()


def test_dry_run_with_explicit_host_emits_driver_command(tmp_path: Path) -> None:
    """``--dry-run`` must print the synthesized driver invocation."""
    result = _run(
        [
            "--host",
            "fake-host",
            "--beta-values",
            "0.1",
            "--kappa-values",
            "0.5,0.9",
            "--c-values",
            "0.5",
            "--num-workers",
            "16",
            "--dry-run",
        ]
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    # Argument plumbing through to the driver:
    assert "compute_nash_phase_diagram.py" in out
    assert "--beta-values '0.1'" in out
    assert "--kappa-values '0.5,0.9'" in out
    assert "--c-values '0.5'" in out
    assert "--num-workers 16" in out
    # Header echoes the resolved host.
    assert "fake-host" in out
    # Dry-run banner must be present so an operator never confuses it
    # with a real launch.
    assert "dry-run" in out.lower()


def test_dry_run_synthesizes_compact_session_name() -> None:
    """Session name compaction must include β/κ/c so concurrent fills don't collide."""
    result = _run(
        [
            "--host",
            "fake-host",
            "--c-values",
            "1.0,2.0",
            "--dry-run",
        ]
    )
    assert result.returncode == 0, result.stderr
    # Expect "nash-fill-c1.0-2.0" — c-values joined with dashes inside the
    # session-name tag, prefixed by nash-fill. (β/κ omitted because they
    # weren't passed; the driver will use its full-grid defaults.)
    assert "nash-fill-c1.0-2.0" in result.stdout, result.stdout


def test_dry_run_resolves_host_from_env(tmp_path: Path) -> None:
    """When --host is omitted, the script must pick the first non-empty
    COMPUTE_HOST_* from .env using the documented priority (PRIMARY first)."""
    env_path = REPO_ROOT / ".env"
    backup = env_path.read_bytes() if env_path.exists() else None
    try:
        env_path.write_text(
            textwrap.dedent(
                """\
                COMPUTE_HOST_PRIMARY=resolved-primary
                COMPUTE_HOST_CLUSTER=resolved-cluster
                """
            )
        )
        result = _run(["--c-values", "2.0", "--dry-run"])
    finally:
        if backup is not None:
            env_path.write_bytes(backup)
        elif env_path.exists():
            env_path.unlink()

    assert result.returncode == 0, result.stderr
    assert "resolved-primary" in result.stdout
    # CLUSTER should NOT win when PRIMARY is set.
    assert "resolved-cluster" not in result.stdout


def test_unknown_flag_exits_nonzero_with_message() -> None:
    """Operator typos must fail loudly, not silently launch the wrong thing."""
    result = _run(["--host", "x", "--not-a-flag", "y", "--dry-run"])
    assert result.returncode != 0
    assert "unknown argument" in result.stderr.lower()


@pytest.mark.parametrize(
    "args,expected_in_cmd",
    [
        # Issue #390 canonical fills (mirroring LAUNCH_RUNBOOK.md):
        # 1. β=0.1 row gap at c=0.5 (κ ∈ {0.5, 0.9}) — 2 cells
        (
            ["--beta-values", "0.1", "--kappa-values", "0.5,0.9", "--c-values", "0.5"],
            ["--beta-values '0.1'", "--kappa-values '0.5,0.9'", "--c-values '0.5'"],
        ),
        # 2. All of c=2.0 (the alc-10 gap) — 9 cells
        (
            ["--c-values", "2.0"],
            ["--c-values '2.0'"],
        ),
        # 3. All of c=1.0 (the missing 3rd plane, if extending preview to 3×3×3)
        (
            ["--c-values", "1.0"],
            ["--c-values '1.0'"],
        ),
    ],
    ids=["beta01_row_gap", "c20_plane", "c10_plane"],
)
def test_runbook_canonical_invocations(
    args: list[str], expected_in_cmd: list[str]
) -> None:
    """Lock in the runbook commands so a refactor to the script cannot
    silently break the documented operator instructions."""
    result = _run(["--host", "fake-host", *args, "--dry-run"])
    assert result.returncode == 0, result.stderr
    for fragment in expected_in_cmd:
        assert fragment in result.stdout, (
            f"missing expected driver flag '{fragment}' in:\n{result.stdout}"
        )
