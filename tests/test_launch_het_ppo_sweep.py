"""Tests for ``experiments/scripts/launch_het_ppo_sweep.sh``.

The script is the operator-launch wrapper added for issue #386 (the
asymmetry-aware HetGPPO-style PPO arm sweep on ``rest_trap`` and the
``asymmetric_only`` phase-diagram cells). It shells out to ``ssh`` to
bootstrap a remote tmux session, so the live-run path is not exercisable in
unit tests. The ``--dry-run`` flag exists exactly so the wiring (arg parsing,
host resolution, driver-command synthesis, session-name compaction) can be
asserted without touching the network.

These tests mirror the contract style established for the phase-diagram
launcher (``tests/test_launch_phase_diagram_fill.py``): each documented
operator invocation in ``experiments/p3_specialization/het_ppo_runbook.md``
is locked in here so a future refactor to the script cannot silently break
the runbook commands.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "experiments" / "scripts" / "launch_het_ppo_sweep.sh"


def _run(args: list[str], env: dict | None = None, cwd: Path | None = None):
    """Invoke the launch script with the given args; return CompletedProcess."""
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    return subprocess.run(  # nosec B603 B607 (cmd is list, no shell)
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
        "--scenarios",
        "--seeds",
        "--num-iterations",
        "--rollout-steps",
        "--dry-run",
    ):
        assert flag in result.stdout, f"--help is missing documentation for {flag}"


def test_missing_scenarios_errors_cleanly() -> None:
    """Without --scenarios, the script must fail with a clear message.

    --scenarios is required because there is no sensible default sweep —
    Phase 1 is rest_trap only, Phase 2 depends on #358's per-cell scenario
    file mapping. We force the operator to be explicit.
    """
    result = _run(["--host", "fake-host", "--dry-run"])
    assert result.returncode == 2, (
        f"expected exit 2 (missing --scenarios), got {result.returncode}: "
        f"{result.stderr}"
    )
    assert "--scenarios" in result.stderr


def test_missing_host_with_no_env_errors_cleanly() -> None:
    """Without --host and without a .env, the script must fail with exit 3."""
    env_path = REPO_ROOT / ".env"
    backup = env_path.read_bytes() if env_path.exists() else None
    if env_path.exists():
        env_path.unlink()
    try:
        result = _run(["--scenarios", "rest_trap", "--dry-run"])
    finally:
        if backup is not None:
            env_path.write_bytes(backup)

    assert result.returncode == 3, (
        f"expected exit 3 (no host), got {result.returncode}: {result.stderr}"
    )
    assert "no host specified" in result.stderr.lower()


def test_dry_run_with_explicit_host_emits_driver_command() -> None:
    """``--dry-run`` must print the synthesized driver invocation."""
    result = _run(
        [
            "--host",
            "fake-host",
            "--scenarios",
            "rest_trap",
            "--seeds",
            "42,43,44",
            "--num-iterations",
            "25",
            "--rollout-steps",
            "1024",
            "--dry-run",
        ]
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    # Argument plumbing through to the driver:
    assert "run_tier1_cell.py" in out
    assert "--trainer het_ppo" in out
    assert "--scenario 'rest_trap'" in out
    # Seeds are passed space-separated (not comma) to run_tier1_cell.py.
    assert "--seeds 42 43 44" in out
    assert "--num-iterations 25" in out
    assert "--rollout-steps 1024" in out
    # Header echoes the resolved host.
    assert "fake-host" in out
    # Dry-run banner must be present so an operator never confuses it
    # with a real launch.
    assert "dry-run" in out.lower()


def test_dry_run_synthesizes_session_name() -> None:
    """Session name must include the scenario tag so concurrent launches
    on the same host don't collide."""
    result = _run(
        [
            "--host",
            "fake-host",
            "--scenarios",
            "rest_trap",
            "--dry-run",
        ]
    )
    assert result.returncode == 0, result.stderr
    assert "het-ppo-rest_trap" in result.stdout, result.stdout


def test_dry_run_session_name_with_multiple_scenarios() -> None:
    """Multi-scenario launches must produce a distinguishable session name."""
    result = _run(
        [
            "--host",
            "fake-host",
            "--scenarios",
            "rest_trap,asym_b05_k05_c09,asym_b05_k09_c09",
            "--dry-run",
        ]
    )
    assert result.returncode == 0, result.stderr
    # First scenario in the tag, plus a count marker so the operator can tell
    # at a glance this was a multi-scenario launch.
    assert "het-ppo-rest_trap-and3more" in result.stdout, result.stdout


def test_dry_run_loops_over_scenarios() -> None:
    """Multi-scenario launches must produce one run_tier1_cell.py invocation
    per scenario inside the synthesized driver command."""
    result = _run(
        [
            "--host",
            "fake-host",
            "--scenarios",
            "rest_trap,asym_b05_k05_c09",
            "--dry-run",
        ]
    )
    assert result.returncode == 0, result.stderr
    # One --scenario clause per scenario in the driver invocation.
    assert "--scenario 'rest_trap'" in result.stdout
    assert "--scenario 'asym_b05_k05_c09'" in result.stdout
    # The loop is chained with &&; verify we have at least one chain so an
    # early failure aborts the rest (set -e + && in the wrapper).
    driver_section = result.stdout.split("Driver command")[-1]
    assert " && " in driver_section, (
        f"multi-scenario driver should chain with &&: {driver_section}"
    )


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
        result = _run(["--scenarios", "rest_trap", "--dry-run"])
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
    result = _run(
        [
            "--host",
            "x",
            "--scenarios",
            "rest_trap",
            "--not-a-flag",
            "y",
            "--dry-run",
        ]
    )
    assert result.returncode != 0
    assert "unknown argument" in result.stderr.lower()


@pytest.mark.parametrize(
    "args,expected_in_cmd",
    [
        # Runbook Plan A: Phase 1 anchor (default 20 seeds × default budget)
        (
            ["--scenarios", "rest_trap"],
            [
                "--trainer het_ppo",
                "--scenario 'rest_trap'",
                # Default seed list expands to 20 space-separated ints.
                "42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61",
            ],
        ),
        # Runbook Plan B: fast turnaround (3 seeds × 25 iter)
        (
            [
                "--scenarios",
                "rest_trap",
                "--seeds",
                "42,43,44",
                "--num-iterations",
                "25",
            ],
            [
                "--trainer het_ppo",
                "--scenario 'rest_trap'",
                "--seeds 42 43 44",
                "--num-iterations 25",
            ],
        ),
        # Runbook Plan C: Phase 2 multi-scenario
        (
            [
                "--scenarios",
                "rest_trap,asym_b05_k05_c09,asym_b05_k09_c09",
            ],
            [
                "--scenario 'rest_trap'",
                "--scenario 'asym_b05_k05_c09'",
                "--scenario 'asym_b05_k09_c09'",
            ],
        ),
    ],
    ids=["plan_a_anchor", "plan_b_fast", "plan_c_phase2"],
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


def test_runbook_referenced_by_script_help() -> None:
    """The runbook path must be discoverable from --help so an operator can
    find it without grepping the source tree."""
    result = _run(["--help"])
    assert result.returncode == 0
    assert "het_ppo_runbook.md" in result.stdout, (
        "--help should point operators at the runbook for the canonical scenario list"
    )
