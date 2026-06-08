"""Tests for ``experiments/scripts/launch_rest_trap_rerun.sh``.

The script is the operator-launch wrapper added for issue #349 (fill the
single missing `rest_trap` cell of the V1 post-#240 Nash sweep — the
2026-05-16 `nash256` sweep crashed at the `equilibrium.json` write step
with ENOSPC, now permanently guarded by the df-precheck from PR #315).
Because it shells out to ``ssh`` to bootstrap a remote tmux session, we
cannot exercise the live-run path in unit tests. The ``--dry-run`` flag
exists exactly so the wiring (arg parsing, host resolution, driver-command
synthesis, canonical-parameter defaults) can be asserted without touching
the network.

These tests cover the operator-facing failure modes plus the canonical
runbook invocation from
``experiments/nash/v1_results_python_post240/RERUN_RUNBOOK.md``.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "experiments" / "scripts" / "launch_rest_trap_rerun.sh"
RUNBOOK = (
    REPO_ROOT
    / "experiments"
    / "nash"
    / "v1_results_python_post240"
    / "RERUN_RUNBOOK.md"
)


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


# ---------------------------------------------------------------------------
# Existence + executability
# ---------------------------------------------------------------------------


def test_script_exists_and_is_executable() -> None:
    """The launch script must be present and have the +x bit set."""
    assert SCRIPT.exists(), f"launch script missing: {SCRIPT}"
    assert os.access(SCRIPT, os.X_OK), (
        f"launch script not executable: {SCRIPT} — chmod +x it before commit"
    )


def test_runbook_exists() -> None:
    """The companion runbook must be present so the launcher's --help
    pointer to it resolves."""
    assert RUNBOOK.exists(), f"runbook missing: {RUNBOOK}"


# ---------------------------------------------------------------------------
# --help / arg parsing
# ---------------------------------------------------------------------------


def test_help_flag_prints_usage_and_exits_zero() -> None:
    """``--help`` must succeed without trying to ssh anywhere."""
    result = _run(["--help"])
    assert result.returncode == 0, result.stderr
    assert "Usage" in result.stdout
    # Spot-check that the canonical flags appear in the doc string.
    for flag in (
        "--host",
        "--dry-run",
    ):
        assert flag in result.stdout, f"--help is missing documentation for {flag}"
    # The help text must not leak past the header into the bash body.
    assert "set -euo pipefail" not in result.stdout, (
        "help text bleeds past the header comment — tighten the sed range"
    )


def test_unknown_flag_exits_nonzero_with_message() -> None:
    """Operator typos must fail loudly, not silently launch the wrong thing."""
    result = _run(["--host", "x", "--not-a-flag", "y", "--dry-run"])
    assert result.returncode != 0
    assert "unknown argument" in result.stderr.lower()


# ---------------------------------------------------------------------------
# Host resolution
# ---------------------------------------------------------------------------


def test_missing_host_with_no_env_errors_cleanly() -> None:
    """Without --host and without a .env, the script must fail with exit 3."""
    env_path = REPO_ROOT / ".env"
    backup = None
    if env_path.exists():
        backup = env_path.read_bytes()
        env_path.unlink()
    try:
        result = _run(["--dry-run"])
    finally:
        if backup is not None:
            env_path.write_bytes(backup)

    assert result.returncode == 3, (
        f"expected exit 3 (no host), got {result.returncode}: {result.stderr}"
    )
    assert "no host specified" in result.stderr.lower()


def test_dry_run_resolves_host_from_env() -> None:
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
        result = _run(["--dry-run"])
    finally:
        if backup is not None:
            env_path.write_bytes(backup)
        elif env_path.exists():
            env_path.unlink()

    assert result.returncode == 0, result.stderr
    assert "resolved-primary" in result.stdout
    # CLUSTER should NOT win when PRIMARY is set.
    assert "resolved-cluster" not in result.stdout


# ---------------------------------------------------------------------------
# Driver command synthesis
# ---------------------------------------------------------------------------


def test_dry_run_emits_canonical_driver_command() -> None:
    """The default invocation must produce exactly the canonical
    `compute_nash.py` command line — positional `rest_trap` (NOT
    --scenario, the curator-caught bug in the original issue body) and
    the locked-in sweep parameters."""
    result = _run(["--host", "fake-host", "--dry-run"])
    assert result.returncode == 0, result.stderr
    out = result.stdout

    # The driver must be called positionally with the scenario name. The
    # script wraps it in single quotes for shell safety; we accept either
    # quoted or bare in case the script is later refactored to drop the
    # unnecessary quoting around a non-special token.
    assert "compute_nash.py 'rest_trap'" in out or "compute_nash.py rest_trap" in out
    # Critical regression guard: the original issue body had the wrong
    # form (--scenario rest_trap) that would have failed at argparse.
    # The launcher must NEVER reintroduce it.
    assert "--scenario" not in out, (
        "compute_nash.py expects `scenario` positionally, not --scenario "
        "(see compute_nash.py:291 and the curator note on issue #349)"
    )

    # Canonical sweep parameters must match the sibling 11 scenarios so
    # that the resulting equilibrium.json is schema-comparable. These
    # values appear in the launcher's defaults; any drift breaks the
    # issue #349 acceptance criterion.
    assert "--simulations 200" in out
    assert "--max-iterations 50" in out
    assert "--epsilon 0.01" in out
    assert "--seed 42" in out

    # Output directory must land in the post-240 sweep tree, NOT in the
    # driver's default fallback at experiments/scenarios/rest_trap/nash/.
    assert "--output-dir 'experiments/nash/v1_results_python_post240/rest_trap'" in out

    # Header echoes the resolved host so the operator can sanity-check.
    assert "fake-host" in out
    # Dry-run banner so an operator never confuses it with a real launch.
    assert "dry-run" in out.lower()


def test_dry_run_default_session_name() -> None:
    """The default tmux session name must be ``nash-rest-trap`` — used by
    the runbook's monitoring snippets."""
    result = _run(["--host", "fake-host", "--dry-run"])
    assert result.returncode == 0, result.stderr
    assert "nash-rest-trap" in result.stdout


def test_dry_run_overrides_propagate_to_driver() -> None:
    """Operator overrides for the canonical sweep params must appear in
    the synthesized driver command. (The runbook says don't use these
    for the #349 re-run, but the launcher exposes them for adjacent
    experimentation — verify the wiring works.)"""
    result = _run(
        [
            "--host",
            "fake-host",
            "--seed",
            "7",
            "--simulations",
            "500",
            "--max-iterations",
            "25",
            "--epsilon",
            "0.005",
            "--output-dir",
            "/tmp/custom-out",
            "--session-name",
            "custom-session",
            "--dry-run",
        ]
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "--seed 7" in out
    assert "--simulations 500" in out
    assert "--max-iterations 25" in out
    assert "--epsilon 0.005" in out
    assert "--output-dir '/tmp/custom-out'" in out
    assert "custom-session" in out


def test_dry_run_default_scenario_is_rest_trap() -> None:
    """The launcher exists to re-run `rest_trap`. The default scenario
    arg must therefore be `rest_trap` (and the header echoes it for the
    operator's confirmation)."""
    result = _run(["--host", "fake-host", "--dry-run"])
    assert result.returncode == 0, result.stderr
    # The plan header shows "Scenario:       rest_trap".
    assert "Scenario:       rest_trap" in result.stdout


# ---------------------------------------------------------------------------
# Canonical runbook invocation
# ---------------------------------------------------------------------------


def test_runbook_canonical_invocation_is_dry_runnable() -> None:
    """Lock in the runbook's primary copy-paste command so a refactor to
    the script cannot silently break the documented operator
    instructions. The runbook says the single-line, default-everything
    invocation should Just Work."""
    # Equivalent to: ./experiments/scripts/launch_rest_trap_rerun.sh
    # with .env supplying the host. We pass --host explicitly here to
    # decouple the test from the local .env contents.
    result = _run(["--host", "fake-host", "--dry-run"])
    assert result.returncode == 0, result.stderr
    out = result.stdout
    # All four canonical sweep params + the post-240 output dir must
    # appear in the synthesized command — this is the schema-match
    # contract with the 11 sibling scenarios.
    for fragment in (
        "--simulations 200",
        "--max-iterations 50",
        "--epsilon 0.01",
        "--seed 42",
        "experiments/nash/v1_results_python_post240/rest_trap",
    ):
        assert fragment in out, (
            f"missing expected fragment '{fragment}' from canonical run:\n{out}"
        )
