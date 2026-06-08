"""Unit tests for the Tier-1 sweep driver (#345).

Covers:

* Per-trainer CLI dispatch — verify the argv constructed for each trainer
  contains the expected selector flags. ``subprocess.run`` is mocked so no
  real training runs.
* Two-step BC-init dispatch — verify ``bc_init.py`` runs before
  ``train.py --bc-init-checkpoint-dir``.
* PBT shell-out — verify the PBT arm dispatches to ``run_issue288_pbt.py``
  not to ``train.py``.
* ``cell_summary.json`` schema — feed synthetic per-seed ``metrics.json``
  files, verify the aggregated cell summary matches the curator-defined
  schema and round-trips through JSON cleanly.
* Aggregator table — feed synthetic ``cell_summary.json`` files at the
  {0.20, 0.49, 0.88} verdict thresholds (historical 4-tier ladder from
  ``diagnostics/random_mlp_search.py``), verify the markdown table is
  well-formed and verdicts classify correctly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
P3_DIR = REPO_ROOT / "experiments" / "p3_specialization"
sys.path.insert(0, str(P3_DIR))

import aggregate_tier1  # noqa: E402
import run_tier1_cell  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _argv_contains(argv: list[str], needle: str) -> bool:
    return any(needle in token for token in argv)


def _argv_contains_pair(argv: list[str], flag: str, value: str) -> bool:
    """Return True iff ``argv`` contains ``flag`` immediately followed by ``value``."""
    for i in range(len(argv) - 1):
        if argv[i] == flag and argv[i + 1] == value:
            return True
    return False


# ---------------------------------------------------------------------------
# Per-trainer dispatch tests (parametrized over the matrix)
# ---------------------------------------------------------------------------


SINGLE_STEP_TRAINER_EXPECTATIONS = [
    # (trainer, expected_flag_substring, expected_pair_or_None)
    ("ippo", "--algorithm", ("--algorithm", "ppo")),
    (
        "het_ppo",
        "--per-agent-init-seed-offset",
        ("--per-agent-init-seed-offset", "1000"),
    ),
    ("mappo", "--centralized-critic", None),
    ("high_lambda", "--gae-lambda", ("--gae-lambda", "0.99")),
    ("lola", "--lola-dice", None),
    ("coma", "--use-coma", None),
    ("hca", "--use-hca", None),
    ("influence", "--influence-coef", ("--influence-coef", "0.5")),
    (
        "nhr",
        "--team-welfare-lambda",
        ("--team-welfare-kind", "team_welfare_closed_form"),
    ),
    ("progress", "--progress-shaping-coef", ("--progress-shaping-coef", "1.0")),
    ("macro_actions", "--macro-actions", None),
    ("reinforce", "--algorithm", ("--algorithm", "reinforce")),
]


@pytest.mark.parametrize("trainer,flag,pair", SINGLE_STEP_TRAINER_EXPECTATIONS)
def test_single_step_trainer_dispatch(trainer, flag, pair, tmp_path):
    """Each single-step trainer constructs an argv with the expected selector flags."""
    argvs = run_tier1_cell.build_argvs_for_seed(
        trainer,
        scenario="minimal_specialization",
        seed=42,
        output_dir=tmp_path,
        num_iterations=1,
        rollout_steps=64,
    )
    assert len(argvs) == 1, f"{trainer} should be single-step, got {len(argvs)} argvs"
    argv = argvs[0]
    # Common train.py invocation properties.
    assert "-m" in argv and "experiments.p3_specialization.train" in argv, (
        f"{trainer} should dispatch via -m experiments.p3_specialization.train; got {argv!r}"
    )
    assert _argv_contains_pair(argv, "--scenario", "minimal_specialization"), argv
    assert _argv_contains_pair(argv, "--seed", "42"), argv
    assert _argv_contains_pair(argv, "--num-iterations", "1"), argv
    assert _argv_contains_pair(argv, "--rollout-steps", "64"), argv
    assert _argv_contains_pair(argv, "--lambda-red", "0.0"), argv
    # Trainer-specific selector flag.
    assert _argv_contains(argv, flag), f"{trainer} argv missing {flag!r}: {argv!r}"
    if pair is not None:
        assert _argv_contains_pair(argv, pair[0], pair[1]), (
            f"{trainer} argv missing pair {pair}: {argv!r}"
        )


def test_dispatch_ippo_has_no_extra_selectors(tmp_path):
    """IPPO baseline should not pull in critic/coma/hca/lola/influence flags."""
    argv = run_tier1_cell.build_argvs_for_seed(
        "ippo",
        scenario="minimal_specialization",
        seed=42,
        output_dir=tmp_path,
        num_iterations=1,
        rollout_steps=64,
    )[0]
    for forbidden in (
        "--centralized-critic",
        "--use-coma",
        "--use-hca",
        "--lola-dice",
        "--macro-actions",
    ):
        assert not _argv_contains(argv, forbidden), (
            f"IPPO argv should not contain {forbidden!r}: {argv!r}"
        )


# ---------------------------------------------------------------------------
# BC-init two-step dispatch
# ---------------------------------------------------------------------------


def test_dispatch_bc_init_two_step(tmp_path):
    """BC-init continuation runs bc_init.py THEN train.py with --bc-init-checkpoint-dir."""
    argvs = run_tier1_cell.build_argvs_for_seed(
        "bc_init_continuation",
        scenario="minimal_specialization",
        seed=42,
        output_dir=tmp_path,
        num_iterations=1,
        rollout_steps=64,
    )
    assert len(argvs) == 2, f"bc_init_continuation should be 2-step; got {len(argvs)}"
    bc_argv, train_argv = argvs
    # Step 1: bc_init.
    assert "experiments.p3_specialization.bc_init" in bc_argv, bc_argv
    assert _argv_contains_pair(bc_argv, "--seed", "42"), bc_argv
    # Step 2: train with --bc-init-checkpoint-dir pointing at step 1's output.
    assert "experiments.p3_specialization.train" in train_argv, train_argv
    assert _argv_contains(train_argv, "--bc-init-checkpoint-dir"), train_argv
    # The checkpoint-dir value should be inside the per-seed output_dir.
    for i, tok in enumerate(train_argv):
        if tok == "--bc-init-checkpoint-dir":
            assert str(tmp_path) in train_argv[i + 1], train_argv
            break


def test_dispatch_bc_init_high_lambda(tmp_path):
    """BC-init high-lambda variant adds --gae-lambda to the train step."""
    argvs = run_tier1_cell.build_argvs_for_seed(
        "bc_init_high_lambda",
        scenario="minimal_specialization",
        seed=42,
        output_dir=tmp_path,
        num_iterations=1,
        rollout_steps=64,
    )
    assert len(argvs) == 2
    _bc, train_argv = argvs
    assert _argv_contains_pair(train_argv, "--gae-lambda", "0.99"), train_argv
    assert _argv_contains(train_argv, "--bc-init-checkpoint-dir"), train_argv


# ---------------------------------------------------------------------------
# PBT shell-out
# ---------------------------------------------------------------------------


def test_dispatch_pbt_shells_out_to_run_issue288(tmp_path):
    """PBT dispatch should target run_issue288_pbt.py, not train.py."""
    argv = run_tier1_cell.build_pbt_argv(
        scenario="minimal_specialization",
        seeds=[42, 43, 44],
        output_dir=tmp_path,
        num_iterations=2,
        rollout_steps=64,
    )
    assert any("run_issue288_pbt.py" in tok for tok in argv), argv
    # Should NOT use -m experiments.p3_specialization.train.
    assert "experiments.p3_specialization.train" not in argv, argv
    # Seeds all passed through.
    for s in ("42", "43", "44"):
        assert s in argv, argv
    assert _argv_contains_pair(argv, "--iters-per-gen", "2"), argv


def test_build_argvs_for_seed_rejects_pbt(tmp_path):
    """PBT must not be dispatched per-seed (it owns the seed loop itself)."""
    with pytest.raises(ValueError, match="PBT"):
        run_tier1_cell.build_argvs_for_seed(
            "pbt",
            scenario="minimal_specialization",
            seed=42,
            output_dir=tmp_path,
            num_iterations=1,
            rollout_steps=64,
        )


# ---------------------------------------------------------------------------
# Unknown trainer
# ---------------------------------------------------------------------------


def test_unknown_trainer_raises():
    with pytest.raises(KeyError):
        run_tier1_cell.build_argvs_for_seed(
            "nonexistent",
            scenario="minimal_specialization",
            seed=42,
            output_dir=Path("/tmp"),
            num_iterations=1,
            rollout_steps=64,
        )


# ---------------------------------------------------------------------------
# Full run_cell with mocked subprocesses
# ---------------------------------------------------------------------------


def _write_synthetic_metrics(seed_dir: Path, gap_target: float) -> None:
    """Write a synthetic ``metrics.json`` whose trailing-5 mean maps to
    approximately ``gap_target`` in gap_closed space.
    """
    # gap_closed(reward) = (reward - MINSPEC_RANDOM) / (MINSPEC_SPECIALIST - MINSPEC_RANDOM)
    # => reward = gap_target * (MINSPEC_SPECIALIST - MINSPEC_RANDOM) + MINSPEC_RANDOM
    from bucket_brigade.baselines import MINSPEC_RANDOM, MINSPEC_SPECIALIST

    target_reward = gap_target * (MINSPEC_SPECIALIST - MINSPEC_RANDOM) + MINSPEC_RANDOM
    seed_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"iter": i, "mean_step_reward_team": float(target_reward)} for i in range(10)
    ]
    (seed_dir / "metrics.json").write_text(json.dumps(rows))


def test_run_cell_writes_summary_schema(tmp_path):
    """End-to-end with mocked subprocess: cell_summary.json matches schema."""
    output_root = tmp_path / "tier1_runs"

    def fake_run(argv, cwd=None, **_kwargs):  # mimic subprocess.run signature
        # Emulate train.py by writing a synthetic metrics.json into the
        # --output-dir referenced in the argv. Skip ``git rev-parse`` calls
        # used by _git_sha — those have no ``--output-dir`` flag.
        out_dir = None
        for i, tok in enumerate(argv):
            if tok == "--output-dir" and i + 1 < len(argv):
                out_dir = Path(argv[i + 1])
                break
        if out_dir is not None:
            _write_synthetic_metrics(out_dir, gap_target=0.35)

        class _CP:
            returncode = 0
            stdout = "deadbeef\n"

        return _CP()

    with patch.object(run_tier1_cell.subprocess, "run", side_effect=fake_run):
        summary = run_tier1_cell.run_cell(
            trainer="ippo",
            scenario="minimal_specialization",
            seeds=[42, 43, 44],
            num_iterations=5,
            rollout_steps=64,
            output_root=output_root,
            skip_precheck=True,
        )

    # Required schema fields per the curator spec.
    required = {
        "trainer",
        "scenario",
        "seeds",
        "num_iterations",
        "n_seeds_completed",
        "n_seeds_failed",
        "gap_closed_mean",
        "gap_closed_std",
        "gap_closed_per_seed",
        "trailing5_team_mean",
        "iter0_gap_closed_mean",
        "min_iter_gap_closed_mean",
        "mean_traj_gap_closed",
        "verdict_tier",
        "verdict_reason",
        "command_invoked",
        "git_sha",
        "wall_clock_seconds",
    }
    missing = required - set(summary)
    assert not missing, f"cell_summary missing fields: {missing}"

    # 3 seeds completed, none failed.
    assert summary["n_seeds_completed"] == 3
    assert summary["n_seeds_failed"] == 0
    # gap_closed_mean should be near 0.35 (we wrote rewards mapping to that).
    assert summary["gap_closed_mean"] == pytest.approx(0.35, abs=1e-6)
    # Verdict should be "partial_lower" at 0.35 (0.20 <= 0.35 < 0.49).
    assert summary["verdict_tier"] == "partial_lower"

    # The artifact file exists and round-trips.
    summary_path = output_root / "ippo_minimal_specialization" / "cell_summary.json"
    assert summary_path.exists()
    parsed = json.loads(summary_path.read_text())
    assert parsed["trainer"] == "ippo"


def test_run_cell_handles_failed_seed(tmp_path):
    """If a seed's subprocess exits non-zero with no metrics, it's marked failed."""
    output_root = tmp_path / "tier1_runs"
    call_count = {"n": 0}

    def fake_run(argv, cwd=None, **_kwargs):
        # Skip ``git rev-parse`` calls (no --output-dir flag).
        out_dir = None
        for i, tok in enumerate(argv):
            if tok == "--output-dir" and i + 1 < len(argv):
                out_dir = Path(argv[i + 1])
                break

        class _CP:
            returncode = 0
            stdout = "deadbeef\n"

        if out_dir is None:
            return _CP()

        call_count["n"] += 1
        if call_count["n"] == 1:
            _CP.returncode = 1
        else:
            _write_synthetic_metrics(out_dir, gap_target=0.9)
        return _CP()

    with patch.object(run_tier1_cell.subprocess, "run", side_effect=fake_run):
        summary = run_tier1_cell.run_cell(
            trainer="ippo",
            scenario="minimal_specialization",
            seeds=[42, 43, 44],
            num_iterations=2,
            rollout_steps=64,
            output_root=output_root,
            skip_precheck=True,
        )
    assert summary["n_seeds_completed"] == 2, summary
    assert summary["n_seeds_failed"] == 1, summary
    assert 42 in summary["failed_seeds"], summary
    # 0.9 is "closed" (>= 0.88 threshold).
    assert summary["verdict_tier"] == "closed"


# ---------------------------------------------------------------------------
# Aggregator tests
# ---------------------------------------------------------------------------


def _make_cell_summary(
    cell_dir: Path, trainer: str, scenario: str, mean: float, std: float = 0.04
) -> None:
    cell_dir.mkdir(parents=True, exist_ok=True)
    from run_tier1_cell import _verdict_for  # noqa: PLC0415 - local helper import

    tier, reason = _verdict_for(mean)
    payload = {
        "trainer": trainer,
        "scenario": scenario,
        "seeds": [42, 43, 44],
        "num_iterations": 50,
        "n_seeds_completed": 3,
        "n_seeds_failed": 0,
        "failed_seeds": [],
        "gap_closed_mean": mean,
        "gap_closed_std": std,
        "gap_closed_per_seed": [mean - std, mean, mean + std],
        "trailing5_team_mean": 0.0,
        "iter0_gap_closed_mean": 0.05,
        "min_iter_gap_closed_mean": 0.05,
        "mean_traj_gap_closed": [],
        "verdict_tier": tier,
        "verdict_reason": reason,
        "command_invoked": "fake",
        "git_sha": "deadbeef",
        "wall_clock_seconds": 100.0,
    }
    (cell_dir / "cell_summary.json").write_text(json.dumps(payload))


def test_aggregator_table_structure_and_verdicts(tmp_path):
    """Verdicts and sort order are correct at the {0.20, 0.49, 0.88} thresholds."""
    root = tmp_path / "tier1_runs"
    _make_cell_summary(
        root / "ippo_minimal_specialization", "ippo", "minimal_specialization", 0.10
    )
    _make_cell_summary(
        root / "lola_minimal_specialization", "lola", "minimal_specialization", 0.31
    )
    _make_cell_summary(
        root / "coma_minimal_specialization", "coma", "minimal_specialization", 0.60
    )
    _make_cell_summary(
        root / "pbt_minimal_specialization", "pbt", "minimal_specialization", 0.95
    )
    # Boundary cells: exactly 0.20 -> partial_lower, exactly 0.49 ->
    # partial_upper, exactly 0.88 -> closed.
    _make_cell_summary(
        root / "edge_low_minimal_specialization",
        "edge_low",
        "minimal_specialization",
        0.20,
    )
    _make_cell_summary(
        root / "edge_mid_minimal_specialization",
        "edge_mid",
        "minimal_specialization",
        0.49,
    )
    _make_cell_summary(
        root / "edge_high_minimal_specialization",
        "edge_high",
        "minimal_specialization",
        0.88,
    )

    rows = aggregate_tier1.load_cells(root)
    assert len(rows) == 7

    md = aggregate_tier1.build_markdown(rows)
    assert "| Trainer |" in md
    assert "| pbt |" in md and "closed" in md
    # Sort: closed first.
    pbt_idx = md.index("| pbt |")
    ippo_idx = md.index("| ippo |")
    assert pbt_idx < ippo_idx, "closed cells should sort before insufficient cells"

    # JSON form classifies thresholds correctly.
    out = aggregate_tier1.build_json(rows)
    by_trainer = {c["trainer"]: c for c in out["cells"]}
    assert by_trainer["ippo"]["verdict_tier"] == "insufficient"  # 0.10
    assert by_trainer["lola"]["verdict_tier"] == "partial_lower"  # 0.31
    assert by_trainer["coma"]["verdict_tier"] == "partial_upper"  # 0.60
    assert by_trainer["pbt"]["verdict_tier"] == "closed"  # 0.95
    assert by_trainer["edge_low"]["verdict_tier"] == "partial_lower"  # exactly 0.20
    assert by_trainer["edge_mid"]["verdict_tier"] == "partial_upper"  # exactly 0.49
    assert by_trainer["edge_high"]["verdict_tier"] == "closed"  # exactly 0.88


def test_aggregator_handles_missing_cell_summary(tmp_path):
    """A cell directory without cell_summary.json surfaces as no_data, not silently dropped."""
    root = tmp_path / "tier1_runs"
    (root / "abandoned_minimal_specialization").mkdir(parents=True)
    _make_cell_summary(
        root / "ippo_minimal_specialization", "ippo", "minimal_specialization", 0.40
    )

    rows = aggregate_tier1.load_cells(root)
    tiers = {r.get("trainer"): r.get("verdict_tier") for r in rows}
    assert "abandoned" in tiers or any(r["verdict_tier"] == "no_data" for r in rows)


def test_aggregator_json_roundtrip(tmp_path):
    """tier1_verdict.json is valid JSON (no NaN tokens)."""
    root = tmp_path / "tier1_runs"
    _make_cell_summary(
        root / "ippo_minimal_specialization", "ippo", "minimal_specialization", 0.40
    )

    rc = aggregate_tier1.main(["--tier1-root", str(root)])
    assert rc == 0
    json_path = root / "tier1_verdict.json"
    md_path = root / "tier1_verdict.md"
    assert json_path.exists() and md_path.exists()
    # Round-trip parse.
    data = json.loads(json_path.read_text())
    assert "cells" in data and "thresholds" in data
    assert data["thresholds"]["partial_lower"] == 0.20
    assert data["thresholds"]["partial_upper"] == 0.49
    assert data["thresholds"]["closed"] == 0.88


def test_aggregator_verdict_for_nan_is_no_data():
    """A NaN mean classifies as no_data, not insufficient."""
    tier, _ = aggregate_tier1.verdict_for(float("nan"))
    assert tier == "no_data"


# ---------------------------------------------------------------------------
# Dispatch-table sanity
# ---------------------------------------------------------------------------


def test_all_required_trainers_present():
    """All 14 trainers from the #343 matrix must be in the dispatch table."""
    expected = {
        "ippo",
        "mappo",
        "high_lambda",
        "bc_init_continuation",
        "bc_init_high_lambda",
        "lola",
        "coma",
        "hca",
        "influence",
        "nhr",
        "progress",
        "macro_actions",
        "reinforce",
        "pbt",
    }
    assert expected.issubset(set(run_tier1_cell.TRAINERS))


def test_het_ppo_trainer_registered():
    """Issue #386: ``het_ppo`` is the asymmetry-aware (HetGPPO-style) PPO arm.

    Dispatches via train.py with ``--algorithm ppo --per-agent-init-seed-offset
    1000`` so each per-position policy is initialized from a maximally-distinct
    RNG stream. Designed for ``asymmetric_only`` phase-diagram cells.
    """
    assert "het_ppo" in run_tier1_cell.TRAINERS
    spec = run_tier1_cell.TRAINERS["het_ppo"]
    assert spec.is_pbt is False, "het_ppo is a single-step train.py trainer"
    assert spec.run_seed is None, "het_ppo uses the default per-seed dispatch"
    assert ("--per-agent-init-seed-offset", "1000") == (
        spec.train_extra[-2],
        spec.train_extra[-1],
    ), (
        f"het_ppo train_extra must end with --per-agent-init-seed-offset 1000; "
        f"got {spec.train_extra!r}"
    )


def test_het_ppo_dispatch_does_not_overlap_other_arms(tmp_path):
    """het_ppo must not silently pull in centralized critic, COMA, HCA, LOLA,
    macro-actions, influence, etc. — those would entangle the asymmetry-aware
    diagnostic with a different experimental arm."""
    argv = run_tier1_cell.build_argvs_for_seed(
        "het_ppo",
        scenario="rest_trap",
        seed=42,
        output_dir=tmp_path,
        num_iterations=1,
        rollout_steps=64,
    )[0]
    for forbidden in (
        "--centralized-critic",
        "--use-coma",
        "--use-hca",
        "--lola-dice",
        "--macro-actions",
        "--influence-coef",
        "--team-welfare-lambda",
        "--progress-shaping-coef",
        "--bc-init-checkpoint-dir",
    ):
        assert not _argv_contains(argv, forbidden), (
            f"het_ppo argv should not contain {forbidden!r}: {argv!r}"
        )
