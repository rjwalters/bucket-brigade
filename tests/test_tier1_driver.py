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


def test_aggregator_appends_notes_file(tmp_path):
    """tier1_verdict_notes.md (if present) is appended to the generated md."""
    root = tmp_path / "tier1_runs"
    _make_cell_summary(
        root / "ippo_minimal_specialization", "ippo", "minimal_specialization", 0.40
    )
    notes = "## Notes\n\nrest_trap gap_closed is a scale artifact.\n"
    (root / aggregate_tier1.NOTES_FILENAME).write_text(notes)

    rc = aggregate_tier1.main(["--tier1-root", str(root)])
    assert rc == 0
    md = (root / "tier1_verdict.md").read_text()
    assert "## Notes" in md
    assert "scale artifact" in md
    # Table still present, notes appended after it.
    assert md.index("| Trainer |") < md.index("## Notes")


def test_aggregator_no_notes_file_leaves_md_unchanged(tmp_path):
    """Without a notes file the markdown has no Notes section."""
    root = tmp_path / "tier1_runs"
    _make_cell_summary(
        root / "ippo_minimal_specialization", "ippo", "minimal_specialization", 0.40
    )
    rc = aggregate_tier1.main(["--tier1-root", str(root)])
    assert rc == 0
    md = (root / "tier1_verdict.md").read_text()
    assert "## Notes" not in md


# ---------------------------------------------------------------------------
# Scenario-aware gap_closed (#434): three resolution outcomes
# ---------------------------------------------------------------------------


def _write_team_metrics(seed_dir: Path, values: list[float]) -> None:
    """Write a ``metrics.json`` with explicit per-iteration team rewards."""
    seed_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"iter": i, "mean_step_reward_team": float(v)} for i, v in enumerate(values)
    ]
    (seed_dir / "metrics.json").write_text(json.dumps(rows))


def _summary_for(tmp_path: Path, scenario: str, per_seed_values) -> dict:
    """Build a cell summary from synthetic per-seed trajectories."""
    seeds = list(range(42, 42 + len(per_seed_values)))
    seed_dirs = []
    for seed, values in zip(seeds, per_seed_values):
        sdir = tmp_path / f"seed_{seed}"
        _write_team_metrics(sdir, values)
        seed_dirs.append(sdir)
    return run_tier1_cell.build_cell_summary(
        trainer="ippo",
        scenario=scenario,
        seeds=seeds,
        seed_dirs=seed_dirs,
        num_iterations=len(per_seed_values[0]),
        command_invoked="fake",
        git_sha="deadbeef",
        wall_clock_seconds=1.0,
    )


def test_rest_trap_is_degenerate_reference_not_ladder(tmp_path):
    """rest_trap must not be classified on the fraction ladder (#434)."""
    from bucket_brigade.baselines import SCENARIO_RANDOM_BASELINES

    rest_trap_random = SCENARIO_RANDOM_BASELINES["rest_trap"]  # 302.87
    # 10 iterations flat at 306.26/step (the het_ppo trailing-5 headline).
    summary = _summary_for(tmp_path, "rest_trap", [[306.26] * 10] * 3)

    assert summary["gap_closed_mean"] is None
    assert summary["gap_closed_std"] is None
    assert summary["gap_closed_per_seed"] is None
    assert summary["iter0_gap_closed_mean"] is None
    assert summary["min_iter_gap_closed_mean"] is None
    assert summary["mean_traj_gap_closed"] is None
    assert summary["gap_source"] == "degenerate_reference"
    assert summary["verdict_tier"] == "not_scored_degenerate_reference"
    assert summary["scenario_random_baseline"] == rest_trap_random
    assert summary["scenario_reference"]["value"] is None
    assert summary["scenario_reference"]["reason"] == "social_trap_ne_below_random"
    # Scenario-scale headline: uplift over uniform random, per-step.
    assert summary["uplift_over_random_mean"] == pytest.approx(
        306.26 - rest_trap_random
    )
    # Legacy MINSPEC-scale audit column still carries the historical value.
    assert summary["gap_closed_minspec_legacy_mean"] == pytest.approx(
        (306.26 - (-87.72)) / (-28.38 - (-87.72))
    )


def test_minimal_specialization_numerics_unchanged(tmp_path):
    """minspec gap values are bit-identical to the historical MINSPEC formula."""
    from bucket_brigade.baselines import MINSPEC_RANDOM, MINSPEC_SPECIALIST

    reward = 0.35 * (MINSPEC_SPECIALIST - MINSPEC_RANDOM) + MINSPEC_RANDOM
    summary = _summary_for(tmp_path, "minimal_specialization", [[reward] * 10] * 3)

    assert summary["gap_source"] == "scenario"
    assert summary["gap_closed_mean"] == pytest.approx(0.35, abs=1e-9)
    assert summary["verdict_tier"] == "partial_lower"
    assert summary["scenario_random_baseline"] == MINSPEC_RANDOM
    assert summary["scenario_reference"]["value"] == MINSPEC_SPECIALIST
    assert summary["scenario_reference"]["kind"] == "specialist_homogeneous"
    # Legacy audit column equals the scored value on minspec.
    assert summary["gap_closed_minspec_legacy_mean"] == pytest.approx(
        summary["gap_closed_mean"]
    )
    # Point-value equivalence of the scenario-aware and legacy formulas.
    assert run_tier1_cell.gap_closed(
        reward, "minimal_specialization"
    ) == run_tier1_cell.gap_closed_minspec_legacy(reward)


def test_unknown_scenario_not_scored_with_warning(tmp_path, capsys):
    """Unknown scenarios: null gap + loud stderr warning, never MINSPEC."""
    summary = _summary_for(tmp_path, "definitely_not_a_scenario", [[1.0] * 10] * 2)
    captured = capsys.readouterr()

    assert "SCENARIO_GAP_REFERENCES" in captured.err
    assert "definitely_not_a_scenario" in captured.err
    assert summary["gap_closed_mean"] is None
    assert summary["gap_source"] == "missing_reference"
    assert summary["verdict_tier"] == "not_scored"
    # No random baseline either -> no uplift.
    assert summary["scenario_random_baseline"] is None
    assert summary["uplift_over_random_mean"] is None
    # Audit column still records the (meaningless off-minspec) legacy value.
    assert summary["gap_closed_minspec_legacy_mean"] is not None


def test_gap_closed_returns_none_off_scenario_table():
    assert run_tier1_cell.gap_closed(100.0, "rest_trap") is None
    assert run_tier1_cell.gap_closed(100.0, "definitely_not_a_scenario") is None


def test_zero_denominator_reference_treated_as_degenerate(tmp_path):
    """reference == random must resolve as degenerate, not ZeroDivisionError."""
    import bucket_brigade.baselines as baselines

    with patch.dict(
        baselines.SCENARIO_GAP_REFERENCES,
        {"zero_denom": {"random": 5.0, "reference": 5.0}},
    ):
        refs = run_tier1_cell.resolve_scenario_references("zero_denom", warn=False)
        assert refs["source"] == "degenerate_reference"
        assert run_tier1_cell.gap_closed(10.0, "zero_denom") is None
        summary = _summary_for(tmp_path, "zero_denom", [[10.0] * 6] * 2)
        assert summary["verdict_tier"] == "not_scored_degenerate_reference"
        assert summary["uplift_over_random_mean"] == pytest.approx(5.0)


def test_all_seeds_failed_is_no_data_not_not_scored(tmp_path):
    """An empty cell yields no_data even on a degenerate-reference scenario."""
    summary = run_tier1_cell.build_cell_summary(
        trainer="het_ppo",
        scenario="rest_trap",
        seeds=[42, 43],
        seed_dirs=[tmp_path / "seed_42", tmp_path / "seed_43"],
        num_iterations=5,
        command_invoked="fake",
        git_sha="deadbeef",
        wall_clock_seconds=1.0,
    )
    assert summary["verdict_tier"] == "no_data"
    assert summary["n_seeds_completed"] == 0
    assert summary["uplift_over_random_mean"] is None
    # NaN-safe JSON round-trip with the new null fields.
    parsed = json.loads(run_tier1_cell._safe_json_dump(summary))
    assert parsed["gap_closed_mean"] is None
    assert parsed["gap_closed_minspec_legacy_mean"] is None


def test_summary_schema_version_bumped(tmp_path):
    summary = _summary_for(tmp_path, "minimal_specialization", [[-80.0] * 6])
    assert summary["schema_version"] == run_tier1_cell.SUMMARY_SCHEMA_VERSION
    assert run_tier1_cell.SUMMARY_SCHEMA_VERSION >= 2


# ---------------------------------------------------------------------------
# Trap-escape verdict rule (#436): four-way CI-based classification
# ---------------------------------------------------------------------------


_TRAP_RANDOM = 302.87
# Measured 95% upper bound of the random anchor itself (battery final-stage
# n=10k uniform re-measurement; PR #440 review): the escaped_trap rung must
# clear THIS, not the bare point.
_TRAP_RANDOM_CI_HI = 304.31
_TRAP_NE_BOUND = 248.67
_TRAP_SCRIPTED = {"value": 386.60, "ci95_lo": 386.17, "ci95_hi": 387.03}


def _classify(
    values,
    *,
    ne=_TRAP_NE_BOUND,
    scripted=_TRAP_SCRIPTED,
    random_ci_hi=_TRAP_RANDOM_CI_HI,
):
    return run_tier1_cell.classify_trap_verdict(
        values,
        random_baseline=_TRAP_RANDOM,
        random_ci95_hi=random_ci_hi,
        ne_per_step_bound=ne,
        scripted_best=scripted,
    )


def test_trap_verdict_trapped_at_ne():
    """CI overlapping/below the NE bound -> trapped_at_ne."""
    verdict, reason, details = _classify([240.0, 245.0, 238.0, 250.0, 242.0])
    assert verdict == "trapped_at_ne"
    assert "NE per-step bound" in reason
    lo, hi = details["trailing5_ci95"]
    assert lo <= _TRAP_NE_BOUND


def test_trap_verdict_at_random():
    """CI above the NE bound but overlapping random -> at_random."""
    verdict, reason, details = _classify([300.0, 302.0, 304.0, 301.0, 303.0])
    assert verdict == "at_random"
    lo, _hi = details["trailing5_ci95"]
    assert _TRAP_NE_BOUND < lo <= _TRAP_RANDOM


def test_trap_verdict_escaped_trap():
    """CI clearing the random anchor's own 95% upper bound but below
    scripted_best -> escaped_trap."""
    verdict, reason, details = _classify([320.0, 322.0, 318.0, 321.0, 319.0])
    assert verdict == "escaped_trap"
    lo, _hi = details["trailing5_ci95"]
    assert lo > _TRAP_RANDOM_CI_HI
    assert lo <= _TRAP_SCRIPTED["ci95_hi"]


def test_trap_verdict_marginal_above_point_below_random_ci_hi_is_at_random():
    """The PR #440 marginal case: a CI lower bound between the random *point*
    anchor and the random anchor's own measured 95% upper bound is NOT a
    statistically supportable "above random" claim -> at_random.

    This pins the het_ppo / rest_trap regression: trained lo = 302.95 clears
    the 302.87 point by 0.08/step but sits below the n=10k uniform
    re-measurement's ci95_hi = 304.31.
    """
    # Flat seeds -> the bootstrap CI degenerates to the point 303.5, strictly
    # between the random point (302.87) and its upper bound (304.31).
    verdict, reason, details = _classify([303.5, 303.5, 303.5])
    assert verdict == "at_random"
    lo, _hi = details["trailing5_ci95"]
    assert _TRAP_RANDOM < lo < _TRAP_RANDOM_CI_HI
    assert "does not clear" in reason
    assert details["anchors"]["random_ci95_hi"] == pytest.approx(_TRAP_RANDOM_CI_HI)


def test_trap_verdict_falls_back_to_point_anchor_without_random_ci_hi():
    """Without a committed random_ci95_hi the rung anchors on the point
    (documented fallback), and the reason string says so."""
    verdict, reason, details = _classify([303.5, 303.5, 303.5], random_ci_hi=None)
    assert verdict == "escaped_trap"
    assert "no measured random_ci95_hi recorded" in reason
    assert details["anchors"]["random_ci95_hi"] is None


def test_trap_verdict_above_scripted_best():
    """CI significantly above the scripted_best anchor -> above_scripted_best."""
    verdict, reason, details = _classify([400.0, 402.0, 398.0, 401.0, 399.0])
    assert verdict == "above_scripted_best"
    lo, _hi = details["trailing5_ci95"]
    assert lo > _TRAP_SCRIPTED["ci95_hi"]


def test_trap_verdict_scripted_best_below_random_not_an_anchor():
    """#436 failure mode: a battery that does not beat random is recorded
    but never anchors the top rung — the rule works on NE + random alone."""
    weak_scripted = {"value": 290.0, "ci95_lo": 289.0, "ci95_hi": 291.0}
    verdict, reason, details = _classify(
        [400.0, 402.0, 398.0, 401.0, 399.0], scripted=weak_scripted
    )
    assert verdict == "escaped_trap"  # top rung unreachable
    assert "not usable as an upper anchor" in reason
    assert details["anchors"]["scripted_best_anchor"] is None


def test_trap_verdict_no_scripted_best_recorded():
    """Missing scripted_best entirely: escaped_trap is the ceiling."""
    verdict, reason, _details = _classify(
        [400.0, 402.0, 398.0, 401.0, 399.0], scripted=None
    )
    assert verdict == "escaped_trap"
    assert "no scripted_best recorded" in reason


def test_trap_verdict_no_ne_bound_bottoms_out_at_at_random():
    """Without an NE bound the bottom two rungs collapse into at_random."""
    verdict, reason, _details = _classify([240.0, 245.0, 238.0, 250.0, 242.0], ne=None)
    assert verdict == "at_random"
    assert "no NE" in reason


def test_trap_verdict_single_seed_degenerate_ci():
    """One seed: the CI collapses to the point estimate (no between-seed
    uncertainty), classification still works."""
    verdict, _reason, details = _classify([320.0])
    lo, hi = details["trailing5_ci95"]
    assert lo == hi == 320.0
    assert verdict == "escaped_trap"


def test_trap_verdict_deterministic():
    """Fixed bootstrap seed: identical inputs give identical CIs."""
    a = _classify([300.0, 310.0, 295.0, 305.0])
    b = _classify([300.0, 310.0, 295.0, 305.0])
    assert a == b


def test_seed_bootstrap_ci_rejects_empty():
    with pytest.raises(ValueError):
        run_tier1_cell._seed_bootstrap_ci([])


def test_rest_trap_summary_carries_trap_verdict(tmp_path):
    """Degenerate-reference cells surface the trap verdict + anchors in
    cell_summary.json (integration through build_cell_summary)."""
    summary = _summary_for(tmp_path, "rest_trap", [[306.26] * 10] * 3)
    assert summary["schema_version"] >= 3
    # Flat 306.26 across seeds -> degenerate CI at 306.26, above the random
    # anchor's measured 95% upper bound (304.31), far below scripted_best.
    assert summary["trap_verdict"] == "escaped_trap"
    assert "trap_verdict = escaped_trap" in summary["verdict_reason"]
    lo, hi = summary["trailing5_team_ci95"]
    assert lo == hi == pytest.approx(306.26)
    anchors = summary["trap_anchors"]
    assert anchors["random"] == pytest.approx(302.87)
    assert anchors["random_ci95_hi"] == pytest.approx(304.3071270072002)
    assert anchors["ne_per_step_bound"] == pytest.approx(2984.043694076538 / 12.0)
    assert anchors["scripted_best_kind"] == "scripted_battery:specialist"
    assert anchors["scripted_best_value"] > anchors["random"]
    # JSON round-trip with the new fields.
    parsed = json.loads(run_tier1_cell._safe_json_dump(summary))
    assert parsed["trap_verdict"] == "escaped_trap"


def test_scored_scenario_has_null_trap_fields(tmp_path):
    """Trap fields must stay null off the degenerate-reference path."""
    summary = _summary_for(tmp_path, "minimal_specialization", [[-80.0] * 6] * 2)
    assert summary["trap_verdict"] is None
    assert summary["trap_verdict_reason"] is None
    assert summary["trap_anchors"] is None
    assert summary["trailing5_team_ci95"] is None


def test_no_data_cell_has_null_trap_fields(tmp_path):
    """An empty degenerate-reference cell has no trap verdict (no_data)."""
    summary = run_tier1_cell.build_cell_summary(
        trainer="het_ppo",
        scenario="rest_trap",
        seeds=[42],
        seed_dirs=[tmp_path / "seed_42"],
        num_iterations=5,
        command_invoked="fake",
        git_sha="deadbeef",
        wall_clock_seconds=1.0,
    )
    assert summary["verdict_tier"] == "no_data"
    assert summary["trap_verdict"] is None


def test_trap_verdict_degenerate_entry_without_anchors(tmp_path):
    """A degenerate entry lacking #436 anchors still classifies (rungs
    collapse gracefully) rather than crashing."""
    import bucket_brigade.baselines as baselines

    with patch.dict(
        baselines.SCENARIO_GAP_REFERENCES,
        {"anchorless": {"random": 5.0, "reference": None, "degenerate_reason": "x"}},
    ):
        summary = _summary_for(tmp_path, "anchorless", [[10.0] * 6] * 2)
        assert summary["verdict_tier"] == "not_scored_degenerate_reference"
        assert summary["trap_verdict"] == "escaped_trap"
        assert summary["trap_anchors"]["ne_per_step_bound"] is None
        assert summary["trap_anchors"]["scripted_best_value"] is None


def test_resolve_scenario_references_passes_trap_anchors():
    refs = run_tier1_cell.resolve_scenario_references("rest_trap", warn=False)
    assert refs["source"] == "degenerate_reference"
    assert refs["ne_per_step_bound"] == pytest.approx(2984.043694076538 / 12.0)
    assert refs["random_ci95_hi"] == pytest.approx(304.3071270072002)
    assert refs["scripted_best"]["kind"] == "scripted_battery:specialist"
    # Scored + missing scenarios expose the keys as None.
    refs = run_tier1_cell.resolve_scenario_references(
        "minimal_specialization", warn=False
    )
    assert refs["ne_per_step_bound"] is None and refs["scripted_best"] is None
    assert refs["random_ci95_hi"] is None
    refs = run_tier1_cell.resolve_scenario_references("nope_never", warn=False)
    assert refs["ne_per_step_bound"] is None and refs["scripted_best"] is None
    assert refs["random_ci95_hi"] is None


def test_rest_trap_marginal_cell_is_at_random_end_to_end(tmp_path):
    """Integration pin for the corrected rung 2 (PR #440): a rest_trap cell
    whose trained CI lower bound clears the 302.87 random point but not the
    committed random_ci95_hi (304.31) must summarize as at_random."""
    summary = _summary_for(tmp_path, "rest_trap", [[303.5] * 10] * 3)
    assert summary["trap_verdict"] == "at_random"
    assert "trap_verdict = at_random" in summary["verdict_reason"]
    lo, hi = summary["trailing5_team_ci95"]
    assert 302.87 < lo < 304.3071270072002


# ---------------------------------------------------------------------------
# --summarize-only recompute mode (#434)
# ---------------------------------------------------------------------------


def test_summarize_only_does_not_dispatch_training(tmp_path):
    """--summarize-only rebuilds cell_summary.json without any trainer run."""
    output_root = tmp_path / "tier1_runs"
    cell_dir = output_root / "het_ppo_rest_trap"
    for seed in (42, 43):
        _write_team_metrics(cell_dir / f"seed_{seed}", [300.0] * 6 + [306.26] * 5)

    def _boom(*_args, **_kwargs):
        raise AssertionError("--summarize-only must not dispatch subprocesses")

    with patch.object(run_tier1_cell, "_run_subprocess", side_effect=_boom):
        summary = run_tier1_cell.run_cell(
            trainer="het_ppo",
            scenario="rest_trap",
            seeds=[42, 43],
            num_iterations=11,
            rollout_steps=64,
            output_root=output_root,
            summarize_only=True,
        )

    assert summary["n_seeds_completed"] == 2
    assert summary["verdict_tier"] == "not_scored_degenerate_reference"
    assert summary["uplift_over_random_mean"] == pytest.approx(306.26 - 302.87)
    parsed = json.loads((cell_dir / "cell_summary.json").read_text())
    assert parsed["gap_closed_mean"] is None
    assert parsed["gap_source"] == "degenerate_reference"


def test_summarize_only_preserves_prior_provenance(tmp_path):
    """Recompute carries forward the original training-run provenance."""
    output_root = tmp_path / "tier1_runs"
    cell_dir = output_root / "ippo_minimal_specialization"
    _write_team_metrics(cell_dir / "seed_42", [-80.0] * 6)
    (cell_dir / "cell_summary.json").write_text(
        json.dumps(
            {
                "command_invoked": "original-training-command",
                "git_sha": "0ldsha00",
                "wall_clock_seconds": 1234.5,
            }
        )
    )

    with patch.object(
        run_tier1_cell,
        "_run_subprocess",
        side_effect=AssertionError("no dispatch"),
    ):
        summary = run_tier1_cell.run_cell(
            trainer="ippo",
            scenario="minimal_specialization",
            seeds=[42],
            num_iterations=6,
            rollout_steps=64,
            output_root=output_root,
            summarize_only=True,
        )

    assert summary["command_invoked"] == "original-training-command"
    assert summary["git_sha"] == "0ldsha00"
    assert summary["wall_clock_seconds"] == 1234.5


def test_summarize_only_cli_flag(tmp_path):
    """main() accepts --summarize-only and skips trainer dispatch."""
    output_root = tmp_path / "tier1_runs"
    _write_team_metrics(
        output_root / "ippo_minimal_specialization" / "seed_42", [-80.0] * 6
    )

    with patch.object(
        run_tier1_cell,
        "_run_subprocess",
        side_effect=AssertionError("no dispatch"),
    ):
        rc = run_tier1_cell.main(
            [
                "--trainer",
                "ippo",
                "--scenario",
                "minimal_specialization",
                "--seeds",
                "42",
                "--summarize-only",
                "--output-root",
                str(output_root),
            ]
        )
    assert rc == 0


# ---------------------------------------------------------------------------
# Aggregator null-gap handling (#434)
# ---------------------------------------------------------------------------


def _make_not_scored_cell_summary(cell_dir: Path) -> None:
    cell_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "trainer": "het_ppo",
        "scenario": "rest_trap",
        "seeds": [42, 43],
        "num_iterations": 50,
        "n_seeds_completed": 2,
        "n_seeds_failed": 0,
        "failed_seeds": [],
        "gap_closed_mean": None,
        "gap_closed_std": None,
        "gap_closed_per_seed": None,
        "gap_source": "degenerate_reference",
        "scenario_random_baseline": 302.87,
        "scenario_reference": {
            "value": None,
            "kind": None,
            "reason": "social_trap_ne_below_random",
            "provenance": "test",
        },
        "uplift_over_random_mean": 3.39,
        "uplift_over_random_std": 7.53,
        "uplift_over_random_per_seed": [3.0, 3.78],
        "trap_verdict": "at_random",
        "trap_verdict_reason": "CI does not clear random_ci95_hi, above NE",
        "trap_anchors": {
            "ne_per_step_bound": 248.67,
            "random": 302.87,
            "random_ci95_hi": 304.31,
            "scripted_best_anchor": 387.03,
            "scripted_best_value": 386.60,
            "scripted_best_kind": "scripted_battery:specialist",
        },
        "trailing5_team_ci95": [302.95, 309.33],
        "gap_closed_minspec_legacy_mean": 6.639,
        "trailing5_team_mean": 306.26,
        "iter0_gap_closed_mean": None,
        "min_iter_gap_closed_mean": None,
        "mean_traj_gap_closed": None,
        "verdict_tier": "not_scored_degenerate_reference",
        "verdict_reason": "reference pair degenerate",
        "command_invoked": "fake",
        "git_sha": "deadbeef",
        "wall_clock_seconds": 100.0,
    }
    (cell_dir / "cell_summary.json").write_text(json.dumps(payload))


def test_aggregator_renders_null_gap_row(tmp_path):
    """A not_scored* row renders with uplift, sorts below scored tiers, and
    never crashes on the null gap columns."""
    root = tmp_path / "tier1_runs"
    _make_cell_summary(
        root / "ippo_minimal_specialization", "ippo", "minimal_specialization", 0.10
    )
    _make_not_scored_cell_summary(root / "het_ppo_rest_trap")
    (root / "abandoned_minimal_specialization").mkdir(parents=True)  # no_data row

    rows = aggregate_tier1.load_cells(root)
    md = aggregate_tier1.build_markdown(rows)

    assert "uplift_over_random" in md
    assert "not_scored_degenerate_reference" in md
    assert "+3.390 ± 7.530" in md
    # #436: trap verdict column present; degenerate row carries its verdict,
    # scored rows render n/a.
    assert "| Trap verdict |" in md
    assert "| at_random |" in md
    ippo_line = next(line for line in md.splitlines() if "| ippo |" in line)
    assert "| n/a |" in ippo_line
    assert "nan" not in md.lower().replace("n/a", "")
    # Sort: scored (even insufficient) above not_scored, no_data last.
    ippo_idx = md.index("| ippo |")
    het_idx = md.index("| het_ppo |")
    assert ippo_idx < het_idx, "insufficient must sort above not_scored*"
    no_data_line = [line for line in md.splitlines() if "no_data" in line]
    assert no_data_line, "missing-summary cell should surface as no_data"
    assert md.index(no_data_line[0]) > het_idx, "no_data must sort last"

    # JSON side: null gap round-trips, uplift + gap_source columns present.
    out = aggregate_tier1.build_json(rows)
    by_trainer = {c["trainer"]: c for c in out["cells"]}
    het = by_trainer["het_ppo"]
    assert het["gap_closed_mean"] is None
    assert het["gap_source"] == "degenerate_reference"
    assert het["trap_verdict"] == "at_random"
    assert by_trainer["ippo"]["trap_verdict"] is None
    assert het["uplift_over_random_mean"] == pytest.approx(3.39)
    assert het["gap_closed_minspec_legacy_mean"] == pytest.approx(6.639)
    assert het["verdict_tier"] == "not_scored_degenerate_reference"
    # Whole payload dumps to valid JSON (no NaN tokens).
    json.loads(aggregate_tier1._safe_json_dump(out))


def test_aggregator_verdict_rank_ordering():
    """not_scored* tiers sit between insufficient and no_data."""
    ranks = [
        aggregate_tier1._verdict_rank(t)
        for t in (
            "closed",
            "partial_upper",
            "partial_lower",
            "insufficient",
            "not_scored_degenerate_reference",
            "not_scored",
            "no_data",
        )
    ]
    assert ranks == sorted(ranks)
    assert len(set(ranks)) == len(ranks)


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
