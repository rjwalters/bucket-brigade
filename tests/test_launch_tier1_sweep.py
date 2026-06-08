"""Tests for ``experiments/scripts/launch_tier1_sweep.sh``.

The script is the operator-launch wrapper for the Tier-1 row of the #343
trainer matrix. Because it shells out to ``ssh`` to bootstrap a remote
tmux session, we cannot exercise the live-run path in unit tests. The
``--dry-run`` flag exists exactly so the wiring (arg parsing, host
resolution, trainer validation, per-cell driver-command synthesis,
session-name compaction) can be asserted without touching the network.

These tests cover:

* operator-facing failure modes (unknown trainer, missing host)
* host resolution from .env
* default launch set (12 cells, one per trainer in the launch set)
* sharding by --trainers
* multi-scenario expansion
* --skip-aggregate
* canonical runbook invocations from TIER1_LAUNCH_RUNBOOK.md

We also assert that the launcher's KNOWN_TRAINERS list stays in sync with
the driver's TRAINERS dict in ``run_tier1_cell.py``. A drift between the
two would mean a typo in the launcher silently rejects (or accepts) a
trainer the driver knows about.
"""

from __future__ import annotations

import os
import re
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "experiments" / "scripts" / "launch_tier1_sweep.sh"
DRIVER = REPO_ROOT / "experiments" / "p3_specialization" / "run_tier1_cell.py"
RUNBOOK = REPO_ROOT / "experiments" / "p3_specialization" / "TIER1_LAUNCH_RUNBOOK.md"
MATRIX_DOC = REPO_ROOT / "experiments" / "p3_specialization" / "TIER1_SWEEP_MATRIX.md"


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


# The dry-run stdout contains the per-cell commands twice: once in the
# pretty-printed plan block (two-space-indented, one cell per line) and
# once inside the embedded REMOTE_BOOTSTRAP script (concatenated with `;`
# inside the tmux invocation). For test assertions about cell count we
# only count the pretty-printed lines because that's the operator-facing
# representation that must match the plan.
_PLAN_LINE_RE = re.compile(
    r"^\s{2}uv run python experiments/p3_specialization/run_tier1_cell\.py",
    re.MULTILINE,
)


def _count_plan_cells(stdout: str) -> int:
    """Count cells in the pretty-printed plan block (not the bootstrap)."""
    return len(_PLAN_LINE_RE.findall(stdout))


# ---------------------------------------------------------------------------
# Existence + executability
# ---------------------------------------------------------------------------


def test_script_exists_and_is_executable() -> None:
    """The launch script must be present and have the +x bit set."""
    assert SCRIPT.exists(), f"launch script missing: {SCRIPT}"
    assert os.access(SCRIPT, os.X_OK), (
        f"launch script not executable: {SCRIPT} — chmod +x it before commit"
    )


def test_runbook_and_matrix_doc_exist() -> None:
    """The companion docs must be present so the runbook references
    resolve from the launcher's --help output."""
    assert RUNBOOK.exists(), f"runbook missing: {RUNBOOK}"
    assert MATRIX_DOC.exists(), f"matrix doc missing: {MATRIX_DOC}"


# ---------------------------------------------------------------------------
# --help / arg parsing
# ---------------------------------------------------------------------------


def test_help_flag_prints_usage_and_exits_zero() -> None:
    """``--help`` must succeed without trying to ssh anywhere."""
    result = _run(["--help"])
    assert result.returncode == 0, result.stderr
    assert "Usage" in result.stdout
    # Spot-check that the canonical flags appear in the docstring usage
    # examples (matches the pattern used by launch_phase_diagram_fill.sh's
    # test_help_flag_prints_usage_and_exits_zero — only the flags actually
    # shown in usage examples are asserted, not every parsed flag).
    for flag in (
        "--host",
        "--trainers",
        "--scenarios",
        "--dry-run",
    ):
        assert flag in result.stdout, f"--help is missing documentation for {flag}"


def test_unknown_flag_exits_nonzero_with_message() -> None:
    """Operator typos must fail loudly, not silently launch the wrong thing."""
    result = _run(
        ["--host", "x", "--not-a-flag", "y", "--trainers", "mappo", "--dry-run"]
    )
    assert result.returncode != 0
    assert "unknown argument" in result.stderr.lower()


def test_unknown_trainer_exits_with_clear_error() -> None:
    """An unknown trainer name must abort BEFORE consuming any compute."""
    result = _run(
        [
            "--host",
            "fake-host",
            "--trainers",
            "definitely-not-a-trainer",
            "--dry-run",
            "--skip-connectivity-check",
        ]
    )
    assert result.returncode == 5, (
        f"expected exit 5 (unknown trainer), got {result.returncode}: {result.stderr}"
    )
    assert "unknown trainer" in result.stderr.lower()


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
        result = _run(["--trainers", "mappo", "--dry-run"])
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
        result = _run(["--trainers", "mappo", "--dry-run"])
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
# Per-cell driver command synthesis
# ---------------------------------------------------------------------------


def test_dry_run_with_single_trainer_emits_one_driver_command() -> None:
    """A single-trainer dry-run must emit exactly one run_tier1_cell.py
    invocation with the requested flags."""
    result = _run(
        [
            "--host",
            "fake-host",
            "--trainers",
            "mappo",
            "--num-iterations",
            "25",
            "--rollout-steps",
            "512",
            "--dry-run",
        ]
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    # The header echoes the resolved trainer + iteration values.
    assert "mappo" in out
    assert "Num iterations: 25" in out
    assert "Rollout steps:  512" in out
    # Exactly one cell command + aggregator. The pretty-printed block has
    # one driver line per cell:
    assert _count_plan_cells(out) == 1
    assert "--trainer 'mappo'" in out
    assert "--scenario 'minimal_specialization'" in out
    assert "--seeds 42 43 44" in out
    assert "--num-iterations 25" in out
    assert "--rollout-steps 512" in out
    # Aggregator runs by default.
    assert "aggregate_tier1.py" in out
    # Dry-run banner so an operator never confuses it with a real launch.
    assert "dry-run" in out.lower()


def test_dry_run_multi_trainer_emits_one_command_per_trainer() -> None:
    """A comma-separated trainer list must produce one driver command per
    trainer (× per scenario)."""
    result = _run(
        [
            "--host",
            "fake-host",
            "--trainers",
            "mappo,high_lambda,reinforce",
            "--dry-run",
        ]
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert _count_plan_cells(out) == 3
    for tr in ("mappo", "high_lambda", "reinforce"):
        assert f"--trainer '{tr}'" in out


def test_dry_run_multi_scenario_expands_cartesian_product() -> None:
    """trainers × scenarios is a Cartesian product, not a zip."""
    result = _run(
        [
            "--host",
            "fake-host",
            "--trainers",
            "mappo,high_lambda",
            "--scenarios",
            "minimal_specialization,default",
            "--dry-run",
        ]
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    # 2 trainers × 2 scenarios = 4 cells.
    assert _count_plan_cells(out) == 4
    for tr in ("mappo", "high_lambda"):
        for sc in ("minimal_specialization", "default"):
            assert f"--trainer '{tr}'" in out
            assert f"--scenario '{sc}'" in out


def test_default_launch_set_has_twelve_cells() -> None:
    """Running with no --trainers must produce the 12-cell default launch
    set documented in TIER1_SWEEP_MATRIX.md."""
    result = _run(["--host", "fake-host", "--dry-run", "--skip-connectivity-check"])
    assert result.returncode == 0, result.stderr
    out = result.stdout
    # 12 trainers × 1 scenario = 12 cells.
    assert _count_plan_cells(out) == 12, (
        f"expected 12 default cells, got {_count_plan_cells(out)}"
    )
    # ippo and coma must be excluded by default per the matrix doc.
    assert "--trainer 'ippo'" not in out
    assert "--trainer 'coma'" not in out


def test_skip_aggregate_omits_aggregator_call() -> None:
    """--skip-aggregate must drop the aggregate_tier1.py invocation so
    sharded multi-host runs don't race on the verdict file."""
    result = _run(
        [
            "--host",
            "fake-host",
            "--trainers",
            "mappo",
            "--skip-aggregate",
            "--dry-run",
        ]
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "aggregate_tier1.py" not in out
    # The header should also confirm aggregation is off.
    assert re.search(r"Aggregate:\s+no", out)


def test_session_name_includes_trainer_count_and_first_trainer() -> None:
    """Session-name compaction must include trainer count + first trainer
    so concurrent launches on the same host don't collide."""
    result = _run(
        [
            "--host",
            "fake-host",
            "--trainers",
            "lola,hca,influence",
            "--dry-run",
        ]
    )
    assert result.returncode == 0, result.stderr
    # Expect "tier1-sweep-n3-lola".
    assert "tier1-sweep-n3-lola" in result.stdout, result.stdout


# ---------------------------------------------------------------------------
# Canonical runbook invocations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args,expected_cells,must_contain",
    [
        # Plan A — default Tier-1 launch (12 cells)
        (
            [],
            12,
            ["--trainer 'mappo'", "--trainer 'pbt'", "aggregate_tier1.py"],
        ),
        # Plan B shard 1 — cheap PPO-family arms (3 cells, --skip-aggregate)
        (
            ["--trainers", "mappo,high_lambda,reinforce", "--skip-aggregate"],
            3,
            ["--trainer 'mappo'", "--trainer 'high_lambda'", "--trainer 'reinforce'"],
        ),
        # Plan B shard 2 — credit-assignment family (4 cells, --skip-aggregate,
        # includes coma which is normally excluded)
        (
            ["--trainers", "lola,hca,influence,coma", "--skip-aggregate"],
            4,
            ["--trainer 'lola'", "--trainer 'coma'"],
        ),
        # Plan C — robustness recheck on a single trainer across 3 scenarios
        (
            [
                "--trainers",
                "bc_init_continuation",
                "--scenarios",
                "minimal_specialization,default,chain_reaction",
            ],
            3,
            [
                "--trainer 'bc_init_continuation'",
                "--scenario 'default'",
                "--scenario 'chain_reaction'",
            ],
        ),
    ],
    ids=[
        "plan_A_default",
        "plan_B_ppo_shard",
        "plan_B_credit_shard",
        "plan_C_robustness",
    ],
)
def test_runbook_canonical_invocations(
    args: list[str], expected_cells: int, must_contain: list[str]
) -> None:
    """Lock in the runbook commands so a refactor to the script cannot
    silently break the documented operator instructions."""
    result = _run(["--host", "fake-host", *args, "--dry-run"])
    assert result.returncode == 0, result.stderr
    out = result.stdout
    actual = _count_plan_cells(out)
    assert actual == expected_cells, (
        f"expected {expected_cells} cells for {args}, got {actual}:\n{out}"
    )
    for fragment in must_contain:
        assert fragment in out, f"missing expected fragment '{fragment}' in:\n{out}"


# ---------------------------------------------------------------------------
# Drift check: launcher's KNOWN_TRAINERS == driver's TRAINERS keys
# ---------------------------------------------------------------------------


def _launcher_known_trainers() -> set[str]:
    """Parse the KNOWN_TRAINERS bash array out of the launch script.

    We do this with a regex rather than execing bash because the value is
    a small literal list and we want a hermetic test that doesn't depend
    on the bash version.
    """
    text = SCRIPT.read_text()
    m = re.search(r"KNOWN_TRAINERS=\(\s*([^)]+)\s*\)", text, re.DOTALL)
    assert m is not None, "could not find KNOWN_TRAINERS array in launcher"
    block = m.group(1)
    # Strip comments and whitespace, split into tokens.
    tokens = [
        line.split("#", 1)[0].strip()
        for line in block.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    # Each non-empty token is one trainer name.
    return {t for t in tokens if t}


def _driver_trainers() -> set[str]:
    """Import the driver's TRAINERS dict via a subprocess so we don't need
    numpy / torch in the test interpreter."""
    out = subprocess.run(
        [
            "python",
            "-c",
            (
                "import sys; sys.path.insert(0, '.'); "
                "from experiments.p3_specialization.run_tier1_cell import TRAINERS; "
                "print('\\n'.join(sorted(TRAINERS)))"
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
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


def test_launcher_known_trainers_matches_driver_trainers() -> None:
    """The launcher's KNOWN_TRAINERS validation list must match the
    driver's TRAINERS dict exactly. Drift means either a typo in the
    launcher silently rejects a real trainer, or accepts a fake one."""
    launcher = _launcher_known_trainers()
    driver = _driver_trainers()
    assert launcher == driver, (
        f"KNOWN_TRAINERS drift detected:\n"
        f"  in launcher but not driver: {launcher - driver}\n"
        f"  in driver but not launcher: {driver - launcher}"
    )
