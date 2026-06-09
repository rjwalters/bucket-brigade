"""Tests for ``experiments/scripts/launch_phase_diagram_ppo.sh``.

The launcher dispatches the phase-diagram PPO sweep (issue #360) to a
remote host. Because it shells out to ``ssh`` we cannot exercise the
live-run path in unit tests — ``--dry-run`` + ``--skip-connectivity-check``
exist exactly so the wiring (arg parsing, scenario lock, host resolution,
driver-command synthesis) can be asserted without touching the network.

The PR #410 Judge review identified one blocking issue: ``--scenario foo``
must be rejected at the CLI because the gap_closed metric written by
``run_phase_diagram_ppo.py`` is hard-coded to the MINSPEC_RANDOM /
MINSPEC_SPECIALIST baselines. These tests lock that fix in place.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "experiments" / "scripts" / "launch_phase_diagram_ppo.sh"


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
    # Spot-check canonical flags appear in the doc string.
    for flag in (
        "--host",
        "--cells-source",
        "--limit-cells",
        "--dry-run",
    ):
        assert flag in result.stdout, f"--help is missing documentation for {flag}"


def test_help_does_not_leak_set_euo_pipefail() -> None:
    """The sed range that produces --help output must stop before the
    ``set -euo pipefail`` directive (cosmetic fix from PR #410 review)."""
    result = _run(["--help"])
    assert result.returncode == 0
    assert "set -euo pipefail" not in result.stdout, (
        "print_usage sed range overshoots into shell directives — check "
        "the '2,Np' range matches the actual header length."
    )


def test_scenario_lock_rejects_non_minspec() -> None:
    """``--scenario`` other than minimal_specialization must be rejected.

    Lock the PR #410 blocking fix in place: the gap_closed metric written
    by run_phase_diagram_ppo.py is calibrated only for the MINSPEC
    baselines. Running with any other scenario produces uncalibrated
    numbers, so the launcher refuses to start the sweep without an
    explicit opt-out flag.
    """
    result = _run(
        [
            "--scenario",
            "default",
            "--dry-run",
            "--skip-connectivity-check",
        ]
    )
    assert result.returncode == 5, (
        f"expected exit 5 (scenario lock), got {result.returncode}: "
        f"stderr={result.stderr!r} stdout={result.stdout!r}"
    )
    # The error message must name the root cause so an operator can act on
    # it without re-reading the source.
    err = result.stderr.lower()
    assert "gap_closed" in err, (
        f"scenario lock error must reference the gap_closed metric: {result.stderr!r}"
    )
    assert "minimal_specialization" in err
    assert "--allow-non-minspec-gap" in result.stderr


def test_allow_non_minspec_gap_overrides_lock() -> None:
    """``--allow-non-minspec-gap`` is the documented opt-out for operators
    who know they want raw trajectories without a comparable gap_closed."""
    result = _run(
        [
            "--scenario",
            "default",
            "--allow-non-minspec-gap",
            "--dry-run",
            "--skip-connectivity-check",
        ]
    )
    assert result.returncode == 0, (
        f"override flag should let non-minspec scenarios through, "
        f"got exit {result.returncode}: {result.stderr!r}"
    )
    # Driver invocation must propagate the override so the driver's own
    # scenario lock also lets the run through.
    assert "--allow-non-minspec-gap" in result.stdout, (
        "launcher must forward --allow-non-minspec-gap to the driver"
    )
    assert "--scenario 'default'" in result.stdout


def test_default_scenario_passes_lock() -> None:
    """Default scenario (minimal_specialization) must not be blocked."""
    result = _run(["--dry-run", "--skip-connectivity-check"])
    assert result.returncode == 0, (
        f"default scenario should pass the lock, got exit "
        f"{result.returncode}: {result.stderr!r}"
    )
    assert "minimal_specialization" in result.stdout
    # Driver invocation should NOT include --allow-non-minspec-gap when
    # the operator hasn't explicitly opted out (it's a no-op but cleaner).
    assert "--allow-non-minspec-gap" not in result.stdout


def test_dry_run_emits_driver_command() -> None:
    """``--dry-run`` must print the synthesized driver invocation so the
    operator can confirm wiring before the real launch."""
    result = _run(["--dry-run", "--skip-connectivity-check"])
    assert result.returncode == 0, result.stderr
    out = result.stdout
    # Argument plumbing through to the driver:
    assert "run_phase_diagram_ppo.py" in out
    assert "--cells-source 'experiments/nash/phase_diagram/results.json'" in out
    assert "--seeds 42 43 44 45" in out
    assert "--num-iterations 50" in out
    assert "--rollout-steps 2048" in out
    # Dry-run banner must be present.
    assert "dry-run" in out.lower()


def test_unknown_flag_exits_nonzero_with_message() -> None:
    """Operator typos must fail loudly, not silently launch the wrong thing."""
    result = _run(["--not-a-flag", "y", "--dry-run", "--skip-connectivity-check"])
    assert result.returncode != 0
    assert "unknown argument" in result.stderr.lower()
