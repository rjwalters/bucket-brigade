"""Tests for the issue #288 PBT orchestrator and verdict classifier.

Three concerns covered:

1. **Verdict classifier (unit, no I/O)**: ``analyze_288.classify_verdict``
   maps the three success-criterion tier boundaries to the correct verdict
   strings. Pure function; the test does not touch disk.

2. **Perturbed-checkpoint pickle round-trip**: ``_perturb_checkpoint_dir``
   produces ``agent_*.pt`` files that ``torch.load`` can read back with
   ``weights_only=True`` (the same code path ``train.py``'s
   ``--bc-init-checkpoint-dir`` uses). Validates that mutation does not break
   the on-disk format and that the perturbation is non-trivial (>0 std
   difference vs the donor).

3. **End-to-end smoke**: tiny PBT run (population=4, generations=2,
   iters_per_gen=2, rollout_steps=128, num_agents=4) on
   ``minimal_specialization`` (the scenario requires exactly 4 agents — its
   ``reward_own_house_survives`` vector is length-4). Verifies:
     - directory layout under ``output_dir/seed_<S>/gen_<G>/lineage_<L>/``
     - ``metrics.json`` written per lineage cell
     - ranking.json written per generation
     - at least one lineage was replaced (bottom 25% of 4 = 1)
     - the replaced lineage has a ``perturbed_init/`` dir at gen 1

The smoke test takes <60s on a laptop CPU; ``pytest -k pbt`` is the local
verification path that the worker actually runs before pushing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # skip module when RL extras absent (issue #484)

from experiments.p3_specialization.analyze_288 import (  # noqa: E402
    VERDICT_ESCAPE_THRESHOLD,
    VERDICT_PARTIAL_THRESHOLD,
    classify_verdict,
)
from experiments.p3_specialization.run_issue288_pbt import (  # noqa: E402
    LineageState,
    _perturb_checkpoint_dir,
    run_pbt,
)


# --- Verdict classifier (no I/O) --------------------------------------------


def test_classify_verdict_population_escape():
    """At/above the escape threshold returns the escape verdict."""
    verdict, _ = classify_verdict(VERDICT_ESCAPE_THRESHOLD)
    assert verdict == "population_escape"
    verdict, _ = classify_verdict(0.85)
    assert verdict == "population_escape"


def test_classify_verdict_partial():
    """In the [partial, escape) band returns ``partial``."""
    verdict, _ = classify_verdict(VERDICT_PARTIAL_THRESHOLD)
    assert verdict == "partial"
    verdict, _ = classify_verdict(0.45)
    assert verdict == "partial"
    # Just below the escape threshold is still partial.
    verdict, _ = classify_verdict(VERDICT_ESCAPE_THRESHOLD - 1e-9)
    assert verdict == "partial"


def test_classify_verdict_basin_globally_unreachable():
    """Below the partial threshold returns the unreachable verdict."""
    verdict, _ = classify_verdict(0.10)
    assert verdict == "basin_globally_unreachable"
    verdict, _ = classify_verdict(VERDICT_PARTIAL_THRESHOLD - 1e-9)
    assert verdict == "basin_globally_unreachable"
    # Curator's stated boundary cases: 0.75 (escape), 0.45 (partial), 0.10 (unreachable).
    assert classify_verdict(0.75)[0] == "population_escape"
    assert classify_verdict(0.45)[0] == "partial"
    assert classify_verdict(0.10)[0] == "basin_globally_unreachable"


# --- Perturbation pickle round-trip -----------------------------------------


def test_perturb_checkpoint_round_trip(tmp_path: Path):
    """Donor checkpoints round-trip through perturbation + torch.load."""
    donor_dir = tmp_path / "donor_policies"
    donor_dir.mkdir()
    # Two synthetic per-agent state dicts. Mix of weight + bias keys, one with
    # zero std (constant bias) to exercise the fallback noise scale.
    torch.manual_seed(0)
    for i in range(2):
        sd = {
            "fc1.weight": torch.randn(4, 3),
            "fc1.bias": torch.zeros(4),  # zero-std -> falls back to plain sigma noise
            "fc2.weight": torch.randn(2, 4) * 0.5,
        }
        torch.save(sd, donor_dir / f"agent_{i}.pt")

    dst_dir = tmp_path / "perturbed_init"
    rng_seed = 123
    import random as pyrand

    _perturb_checkpoint_dir(donor_dir, dst_dir, sigma=0.01, rng=pyrand.Random(rng_seed))

    for i in range(2):
        src = torch.load(donor_dir / f"agent_{i}.pt", weights_only=True)
        dst = torch.load(dst_dir / f"agent_{i}.pt", weights_only=True)
        assert set(src.keys()) == set(dst.keys())
        for key in src:
            assert src[key].shape == dst[key].shape
            assert src[key].dtype == dst[key].dtype
        # Perturbation is non-trivial: at least one float tensor must differ.
        any_diff = any(
            not torch.equal(src[k], dst[k])
            for k in src
            if torch.is_floating_point(src[k])
        )
        assert any_diff, "perturbation produced an identical checkpoint"


# --- End-to-end smoke -------------------------------------------------------


@pytest.mark.slow
def test_pbt_smoke_end_to_end(tmp_path: Path):
    """Tiny PBT run validates directory layout, replacement, and metrics."""
    output_root = tmp_path / "issue288_smoke"
    summary = run_pbt(
        output_root=output_root,
        pbt_seed=42,
        population_size=4,
        generations=2,
        iters_per_gen=2,
        scenario="minimal_specialization",
        lambda_red=0.0,
        num_agents=4,
        rollout_steps=128,
        initial_lr=3e-4,
        initial_entropy_coef=0.01,
        weight_noise=0.01,
        truncation_frac=0.25,
    )

    seed_dir = output_root / "seed_42"
    assert seed_dir.is_dir()

    # gen_0 and gen_1 directories with one cell per lineage.
    for gen in (0, 1):
        gen_dir = seed_dir / f"gen_{gen}"
        assert gen_dir.is_dir(), f"missing {gen_dir}"
        for lineage in range(4):
            cell = gen_dir / f"lineage_{lineage}"
            assert cell.is_dir(), f"missing {cell}"
            assert (cell / "metrics.json").exists(), f"metrics.json missing for {cell}"
            metrics = json.loads((cell / "metrics.json").read_text())
            assert len(metrics) == 2, "expected 2 iters per generation"

        ranking_path = gen_dir / "ranking.json"
        assert ranking_path.exists()
        ranking = json.loads(ranking_path.read_text())
        assert ranking["population_size"] == 4
        # gen_0 ranking should record at least one replacement (bottom 25% of 4 = 1).
        if gen == 0:
            assert len(ranking["replacements"]) >= 1, (
                "expected at least one bottom-25% replacement after gen_0"
            )
            replaced_ids = {r["lineage_id"] for r in ranking["replacements"]}
            # Each replaced lineage should have a perturbed_init dir in gen_1.
            for lid in replaced_ids:
                perturbed = seed_dir / "gen_1" / f"lineage_{lid}" / "perturbed_init"
                assert perturbed.is_dir(), (
                    f"replaced lineage {lid} missing perturbed_init dir"
                )
                # And the perturbed_init should contain per-agent checkpoints.
                agent_ckpts = list(perturbed.glob("agent_*.pt"))
                assert len(agent_ckpts) == 4, (
                    f"expected 4 perturbed agent ckpts for lineage {lid}, "
                    f"got {len(agent_ckpts)}"
                )

    # Final per-seed state file must round-trip.
    state_path = seed_dir / "lineage_state.json"
    assert state_path.exists()
    states = json.loads(state_path.read_text())
    assert len(states) == 4
    for s in states:
        assert "lr" in s and "entropy_coef" in s
        assert len(s["trailing5_team_history"]) == 2

    assert summary["population_size"] == 4
    assert summary["generations"] == 2
    assert summary["final_best_trailing5"] != float("-inf"), (
        "all lineages crashed — smoke is broken, see train.log per cell"
    )


def test_lineage_state_defaults():
    """LineageState constructs with the expected defaults and types."""
    s = LineageState(lineage_id=3, seed=42_003, lr=3e-4, entropy_coef=0.01)
    assert s.init_checkpoint_dir is None
    assert s.trailing5_team_history == []
    assert s.donor_history == []
