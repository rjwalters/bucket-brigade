"""Tests for ``experiments/scripts/launch_ppo_baselines.sh``.

The script is the operator-launch wrapper added for issue #384 (the PPO
baseline training sweep that produces the frozen checkpoints shipped by
issue #371). It shells out to ``ssh`` to bootstrap a remote tmux session,
so the live-run path is not exercisable in unit tests. The ``--dry-run``
flag exists exactly so the wiring (arg parsing, host resolution, driver-
command synthesis, session-name compaction) can be asserted without
touching the network.

These tests mirror the contract style established for the sibling
launchers (``tests/test_launch_phase_diagram_fill.py``,
``tests/test_launch_tier1_sweep.py``, ``tests/test_launch_het_ppo_sweep.py``):
each documented operator invocation in
``experiments/p3_specialization/PPO_BASELINES_RUNBOOK.md`` is locked in
here so a future refactor to the script cannot silently break the
runbook commands.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "experiments" / "scripts" / "launch_ppo_baselines.sh"
RUNBOOK = REPO_ROOT / "experiments" / "p3_specialization" / "PPO_BASELINES_RUNBOOK.md"


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
    output can reference it."""
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
        "--scenarios",
        "--seeds",
        "--num-iterations",
        "--rollout-steps",
        "--dry-run",
    ):
        assert flag in result.stdout, f"--help is missing documentation for {flag}"


def test_unknown_flag_exits_nonzero_with_message() -> None:
    """Operator typos must fail loudly, not silently launch the wrong thing."""
    result = _run(
        [
            "--host",
            "x",
            "--not-a-flag",
            "y",
            "--dry-run",
        ]
    )
    assert result.returncode != 0
    assert "unknown argument" in result.stderr.lower()


def test_empty_scenarios_errors_cleanly() -> None:
    """Explicit empty --scenarios must fail (the default is non-empty,
    but a deliberately-empty override is almost certainly a mistake)."""
    result = _run(
        [
            "--host",
            "fake-host",
            "--scenarios",
            "",
            "--dry-run",
        ]
    )
    assert result.returncode == 2, (
        f"expected exit 2 (empty --scenarios), got {result.returncode}: {result.stderr}"
    )
    assert "--scenarios" in result.stderr


# ---------------------------------------------------------------------------
# Host resolution
# ---------------------------------------------------------------------------


def test_missing_host_with_no_env_errors_cleanly() -> None:
    """Without --host and without a .env, the script must fail with exit 3."""
    env_path = REPO_ROOT / ".env"
    backup = env_path.read_bytes() if env_path.exists() else None
    if env_path.exists():
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


def test_dry_run_with_explicit_host_emits_driver_command() -> None:
    """``--dry-run`` must print the synthesized driver invocation."""
    result = _run(
        [
            "--host",
            "fake-host",
            "--scenarios",
            "minimal_specialization",
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
    assert "--trainer ippo" in out
    assert "--scenario 'minimal_specialization'" in out
    # Seeds are passed space-separated (not comma) to run_tier1_cell.py.
    assert "--seeds 42 43 44" in out
    assert "--num-iterations 25" in out
    assert "--rollout-steps 1024" in out
    # Header echoes the resolved host.
    assert "fake-host" in out
    # Dry-run banner must be present so an operator never confuses it
    # with a real launch.
    assert "dry-run" in out.lower()


def test_dry_run_default_output_dir_is_baselines() -> None:
    """The default --output-dir must point at the baselines tree so #371's
    packaging step knows where to look."""
    result = _run(
        [
            "--host",
            "fake-host",
            "--scenarios",
            "minimal_specialization",
            "--dry-run",
        ]
    )
    assert result.returncode == 0, result.stderr
    assert "experiments/p3_specialization/baselines" in result.stdout


def test_dry_run_default_scenarios_include_release_set() -> None:
    """With no --scenarios, the launcher must expand to the canonical
    #384 release scenario set (all 6 names appear in the driver plan)."""
    result = _run(
        [
            "--host",
            "fake-host",
            "--dry-run",
        ]
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    for scen in (
        "minimal_specialization",
        "default",
        "positional_default",
        "chain_reaction",
        "trivial_cooperation",
        "v2_minimal",
    ):
        assert f"--scenario '{scen}'" in out, (
            f"default scenario set missing '{scen}':\n{out}"
        )


def test_dry_run_synthesizes_session_name_single_scenario() -> None:
    """Session name must include the scenario tag so concurrent launches
    on the same host don't collide."""
    result = _run(
        [
            "--host",
            "fake-host",
            "--scenarios",
            "minimal_specialization",
            "--dry-run",
        ]
    )
    assert result.returncode == 0, result.stderr
    assert "ppo-baselines-minimal_specialization" in result.stdout, result.stdout


def test_dry_run_session_name_with_multiple_scenarios() -> None:
    """Multi-scenario launches must produce a distinguishable session name."""
    result = _run(
        [
            "--host",
            "fake-host",
            "--scenarios",
            "minimal_specialization,default,chain_reaction",
            "--dry-run",
        ]
    )
    assert result.returncode == 0, result.stderr
    # First scenario + count marker so the operator can tell at a glance
    # this was a multi-scenario launch.
    assert "ppo-baselines-minimal_specialization-and3more" in result.stdout, (
        result.stdout
    )


def test_dry_run_loops_over_scenarios_with_semicolon() -> None:
    """Multi-scenario launches must produce one run_tier1_cell.py invocation
    per scenario, chained with ';' so one failing scenario does not abort
    the rest (vs het_ppo which uses '&&' because each scenario is an
    indivisible Phase). For #384 we want partial success — if 5/6 scenarios
    train and 1 fails, #371 can still package the 5."""
    result = _run(
        [
            "--host",
            "fake-host",
            "--scenarios",
            "minimal_specialization,default",
            "--dry-run",
        ]
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    # One --scenario clause per scenario in the driver invocation.
    assert "--scenario 'minimal_specialization'" in out
    assert "--scenario 'default'" in out
    # The chain uses `;` so a single-scenario failure doesn't take down the
    # rest of the sweep. Count actual driver invocations in the plan block
    # by counting `uv run python experiments/p3_specialization/run_tier1_cell.py`
    # — the prose around the plan also mentions `run_tier1_cell.py` once but
    # the literal command string is distinctive.
    plan_section = out.split("Per-scenario driver commands")[-1].split(
        "===================================================================="
    )[0]
    n_cells = plan_section.count(
        "uv run python experiments/p3_specialization/run_tier1_cell.py"
    )
    assert n_cells == 2, (
        f"expected 2 cells in plan for 2 scenarios, got {n_cells}:\n{plan_section}"
    )


def test_unknown_flag_does_not_silently_become_scenario() -> None:
    """A typoed flag like `--scenrios` (note the typo) must error, not
    fall through to the default scenarios with mysterious extra args."""
    result = _run(
        [
            "--host",
            "fake-host",
            "--scenrios",
            "minimal_specialization",
            "--dry-run",
        ]
    )
    assert result.returncode != 0
    assert "unknown argument" in result.stderr.lower()


# ---------------------------------------------------------------------------
# Canonical runbook invocations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args,expected_in_cmd",
    [
        # Runbook Plan A: full release sweep (default scenario set)
        (
            [],
            [
                "--trainer ippo",
                "--scenario 'minimal_specialization'",
                "--scenario 'default'",
                "--scenario 'positional_default'",
                "--scenario 'chain_reaction'",
                "--scenario 'trivial_cooperation'",
                "--scenario 'v2_minimal'",
                # Default seeds = 42 43 44
                "--seeds 42 43 44",
            ],
        ),
        # Runbook Plan B: single scenario sanity rerun
        (
            [
                "--scenarios",
                "minimal_specialization",
                "--seeds",
                "42,43,44",
                "--num-iterations",
                "25",
            ],
            [
                "--trainer ippo",
                "--scenario 'minimal_specialization'",
                "--seeds 42 43 44",
                "--num-iterations 25",
            ],
        ),
        # Runbook Plan C shard 1: cheap scenarios
        (
            [
                "--scenarios",
                "minimal_specialization,trivial_cooperation,v2_minimal",
            ],
            [
                "--scenario 'minimal_specialization'",
                "--scenario 'trivial_cooperation'",
                "--scenario 'v2_minimal'",
            ],
        ),
        # Runbook Plan C shard 2: expensive scenarios
        (
            [
                "--scenarios",
                "default,positional_default,chain_reaction",
            ],
            [
                "--scenario 'default'",
                "--scenario 'positional_default'",
                "--scenario 'chain_reaction'",
            ],
        ),
    ],
    ids=["plan_a_full", "plan_b_sanity", "plan_c_shard1", "plan_c_shard2"],
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
    assert "PPO_BASELINES_RUNBOOK.md" in result.stdout, (
        "--help should point operators at the runbook for the canonical "
        "scenario list and handoff procedure to #371"
    )


def test_handoff_to_371_documented_in_followups() -> None:
    """A real ``ssh``-mode launch prints follow-up instructions to the
    operator. The handoff to #371 must be referenced so the operator
    knows the workflow does not end at "training complete".

    We assert this against the help/usage text rather than running a real
    launch (which we can't from a test)."""
    result = _run(["--help"])
    assert result.returncode == 0
    # The header in the --help block mentions #371 explicitly.
    assert "#371" in result.stdout, (
        "--help should reference #371 as the downstream consumer of the "
        "checkpoints this launcher produces"
    )


# ---------------------------------------------------------------------------
# Drift check: launcher uses 'ippo' which must be in the driver's TRAINERS
# ---------------------------------------------------------------------------


def test_launcher_trainer_name_matches_driver() -> None:
    """The launcher hardcodes ``--trainer ippo``. If the driver renames
    or removes the ippo entry from TRAINERS, the launcher silently
    produces a command that will error out on the remote host. Catch
    drift here, locally, before consuming compute."""
    out = subprocess.run(
        [
            "python",
            "-c",
            (
                "import sys; sys.path.insert(0, '.'); "
                "from experiments.p3_specialization.run_tier1_cell import TRAINERS; "
                "print('ippo' if 'ippo' in TRAINERS else 'MISSING')"
            ),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        pytest.skip(
            f"cannot import run_tier1_cell.TRAINERS in test env "
            f"(stderr: {out.stderr.strip()}); skipping drift check"
        )
    assert out.stdout.strip() == "ippo", (
        "launcher hardcodes --trainer ippo but driver does not expose it. "
        "Either restore the ippo entry in TRAINERS or update the launcher."
    )
