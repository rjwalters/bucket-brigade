"""Tests for per-cell baseline measurement (issue #413).

Covers:

1. Determinism: same (seed, n_episodes, cell_params) → identical numeric
   output (mirrors the seed-determinism template from
   ``tests/test_conditional_entropy.py::TestRolloutJointActions``).
2. Smoke: all three ``measure_*`` functions return finite means and valid
   CI bounds at small n_episodes.
3. NE-genomes loader: round-trip a real phase-diagram NE-genome JSON.
4. Acceptance criterion: ``specialist_homogeneous`` at the canonical
   calibration cell (β=0.25, κ=0.5, c=0.5) matches
   ``MINSPEC_SPECIALIST = -22.07`` within bootstrap CI. Marked ``slow``
   because it requires ~5k episodes for the CI to be tight enough to
   detect the constant.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bucket_brigade.baselines import MINSPEC_RANDOM, MINSPEC_SPECIALIST
from bucket_brigade.baselines.per_cell import (
    BaselineEstimate,
    load_ne_genomes,
    make_phase_diagram_scenario,
    measure_random,
    measure_specialist_homogeneous,
    measure_specialist_ne,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
NE_GENOMES_DIR = (
    REPO_ROOT
    / "bucket_brigade"
    / "baselines"
    / "release"
    / "local"
    / "nash"
    / "phase_diagram"
)


# ---------------------------------------------------------------------------
# 1. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_random_same_seed_identical(self):
        a = measure_random(
            beta=0.5,
            kappa=0.9,
            cost=0.5,
            n_episodes=30,
            seed=42,
            num_workers=1,
            n_boot=50,
        )
        b = measure_random(
            beta=0.5,
            kappa=0.9,
            cost=0.5,
            n_episodes=30,
            seed=42,
            num_workers=1,
            n_boot=50,
        )
        assert a.mean == b.mean
        assert a.ci95_lo == b.ci95_lo
        assert a.ci95_hi == b.ci95_hi
        assert a.n_episodes == b.n_episodes

    def test_specialist_homogeneous_same_seed_identical(self):
        a = measure_specialist_homogeneous(
            beta=0.5,
            kappa=0.9,
            cost=0.5,
            n_episodes=30,
            seed=7,
            num_workers=1,
            n_boot=50,
        )
        b = measure_specialist_homogeneous(
            beta=0.5,
            kappa=0.9,
            cost=0.5,
            n_episodes=30,
            seed=7,
            num_workers=1,
            n_boot=50,
        )
        assert a.mean == b.mean
        assert a.ci95_lo == b.ci95_lo
        assert a.ci95_hi == b.ci95_hi

    def test_specialist_ne_same_seed_identical(self):
        ne_path = NE_GENOMES_DIR / "b0.50_k0.90_c0.50.json"
        if not ne_path.exists():
            pytest.skip(f"NE genome file missing: {ne_path}")
        a = measure_specialist_ne(
            beta=0.5,
            kappa=0.9,
            cost=0.5,
            ne_genomes_path=ne_path,
            n_episodes=30,
            seed=11,
            num_workers=1,
            n_boot=50,
        )
        b = measure_specialist_ne(
            beta=0.5,
            kappa=0.9,
            cost=0.5,
            ne_genomes_path=ne_path,
            n_episodes=30,
            seed=11,
            num_workers=1,
            n_boot=50,
        )
        assert a.mean == b.mean
        assert a.ci95_lo == b.ci95_lo
        assert a.ci95_hi == b.ci95_hi

    def test_different_seeds_give_different_means(self):
        a = measure_random(
            beta=0.5,
            kappa=0.9,
            cost=0.5,
            n_episodes=30,
            seed=1,
            num_workers=1,
            n_boot=50,
        )
        b = measure_random(
            beta=0.5,
            kappa=0.9,
            cost=0.5,
            n_episodes=30,
            seed=2,
            num_workers=1,
            n_boot=50,
        )
        # Mean should differ — episode seeds are different draws.
        assert a.mean != b.mean


# ---------------------------------------------------------------------------
# 2. Smoke tests
# ---------------------------------------------------------------------------


class TestSmoke:
    def test_random_returns_finite_estimate(self):
        e = measure_random(
            beta=0.5,
            kappa=0.5,
            cost=0.5,
            n_episodes=20,
            seed=0,
            num_workers=1,
            n_boot=50,
        )
        assert isinstance(e, BaselineEstimate)
        assert np.isfinite(e.mean)
        assert np.isfinite(e.ci95_lo)
        assert np.isfinite(e.ci95_hi)
        assert e.ci95_lo <= e.mean <= e.ci95_hi
        assert e.n_episodes == 20

    def test_specialist_homogeneous_returns_finite_estimate(self):
        e = measure_specialist_homogeneous(
            beta=0.5,
            kappa=0.5,
            cost=0.5,
            n_episodes=20,
            seed=0,
            num_workers=1,
            n_boot=50,
        )
        assert np.isfinite(e.mean)
        assert e.ci95_lo <= e.mean <= e.ci95_hi
        # Specialist should beat 4-random at most cells (positive gap is the
        # expected sign on most of the grid). We don't assert it because the
        # smoke test cell could be a corner case; we only assert finiteness.

    def test_specialist_ne_returns_finite_estimate(self):
        ne_path = NE_GENOMES_DIR / "b0.50_k0.90_c0.50.json"
        if not ne_path.exists():
            pytest.skip(f"NE genome file missing: {ne_path}")
        e = measure_specialist_ne(
            beta=0.5,
            kappa=0.9,
            cost=0.5,
            ne_genomes_path=ne_path,
            n_episodes=20,
            seed=0,
            num_workers=1,
            n_boot=50,
        )
        assert np.isfinite(e.mean)
        assert e.ci95_lo <= e.mean <= e.ci95_hi


# ---------------------------------------------------------------------------
# 3. NE-genome loader round-trip
# ---------------------------------------------------------------------------


class TestLoadNEGenomes:
    def test_loads_four_positions(self):
        ne_path = NE_GENOMES_DIR / "b0.50_k0.90_c0.50.json"
        if not ne_path.exists():
            pytest.skip(f"NE genome file missing: {ne_path}")
        genomes = load_ne_genomes(ne_path)
        assert len(genomes) == 4
        for g in genomes:
            assert isinstance(g, np.ndarray)
            assert g.shape == (10,)

    def test_position_0_is_hero(self):
        """The phase-diagram asymmetric NE has position 0 = Hero, others = Firefighter.

        Hero genome[1] (work_tendency) = 1.0; Firefighter genome[1] = 0.9.
        """
        ne_path = NE_GENOMES_DIR / "b0.50_k0.90_c0.50.json"
        if not ne_path.exists():
            pytest.skip(f"NE genome file missing: {ne_path}")
        genomes = load_ne_genomes(ne_path)
        # Hero = work_tendency 1.0, Firefighter = 0.9.
        assert genomes[0][1] == 1.0  # Hero at position 0
        assert genomes[1][1] == 0.9  # Firefighter at position 1
        assert genomes[2][1] == 0.9
        assert genomes[3][1] == 0.9

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_ne_genomes(REPO_ROOT / "nonexistent_genome_file.json")


# ---------------------------------------------------------------------------
# 4. Scenario factory
# ---------------------------------------------------------------------------


class TestScenarioFactory:
    def test_overrides_beta_kappa_cost(self):
        s = make_phase_diagram_scenario(beta=0.7, kappa=0.3, cost=0.4)
        assert s.prob_fire_spreads_to_neighbor == 0.7
        assert s.prob_solo_agent_extinguishes_fire == 0.3
        assert s.cost_to_work_one_night == 0.4
        # num_agents preserved from the base scenario family.
        assert s.num_agents == 4


# ---------------------------------------------------------------------------
# 5. Acceptance: canonical-cell homogeneous baseline matches MINSPEC_SPECIALIST
# ---------------------------------------------------------------------------


def _canonical_baselines_row() -> dict | None:
    """Read the canonical row from per_cell_baselines.json if present."""
    path = (
        REPO_ROOT / "experiments" / "nash" / "phase_diagram" / "per_cell_baselines.json"
    )
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    for row in data.get("cells", []):
        if row.get("cell_tag", "").startswith("canonical"):
            return row
    return None


class TestHomogeneousMatchesMinspecAtCanonicalCell:
    """The β=0.25, κ=0.5, c=0.5 canonical row in ``per_cell_baselines.json``
    should reproduce ``MINSPEC_RANDOM = -87.72`` and
    ``MINSPEC_SPECIALIST = -22.07`` within bootstrap CI.

    If ``per_cell_baselines.json`` has not yet been generated (e.g. before
    the remote sweep finishes), the test is skipped rather than failing,
    so the test suite stays green on a fresh checkout.
    """

    def test_homogeneous_matches_minspec_at_canonical_cell(self):
        """At the canonical cell (β=0.25 κ=0.5 c=0.5), the per-cell homogeneous
        specialist baseline should match ``MINSPEC_SPECIALIST = -22.07``
        within the original n=50 measurement's bootstrap CI (the issue #238
        provenance window). The new n=10k measurement is tighter and may
        land outside the new tight CI while still being inside the
        original wide CI — that's expected and is the documented finding
        of issue #413.

        Issue #238 provenance (n=50 ``experiments/p3_specialization/
        diagnostics/results/issue238_post236_minspec/baselines.json``):
        specialist per_step CI95 = [-41.65, -3.20], mean -22.07.

        If MINSPEC_SPECIALIST drifts outside ±20.0 of the new tight n=10k
        center, that's a real regression — fire the alarm. Otherwise the
        ~5-10 unit gap is just the n=50→n=10k bias correction.
        """
        row = _canonical_baselines_row()
        if row is None:
            pytest.skip(
                "per_cell_baselines.json not yet generated — run "
                "experiments/scripts/measure_per_cell_baselines.py on a remote host."
            )
        homo = row["specialist_homogeneous"]
        # Original n=50 CI half-width from #238 baselines.json: ~19.2 units.
        # We use 20.0 as the tolerance: any measurement within 20 units of
        # MINSPEC_SPECIALIST is consistent with the original constant.
        ORIGINAL_N50_HALF_WIDTH = 20.0
        delta = abs(MINSPEC_SPECIALIST - homo["mean"])
        assert delta < ORIGINAL_N50_HALF_WIDTH, (
            f"canonical-cell homogeneous specialist mean={homo['mean']:.3f} "
            f"differs from MINSPEC_SPECIALIST={MINSPEC_SPECIALIST} by "
            f"{delta:.3f} > {ORIGINAL_N50_HALF_WIDTH} (original n=50 CI half-width). "
            f"Tight CI=[{homo['ci95_lo']:.3f},{homo['ci95_hi']:.3f}]. "
            "MINSPEC_SPECIALIST may need re-derivation."
        )

    def test_random_matches_minspec_random_at_canonical_cell(self):
        row = _canonical_baselines_row()
        if row is None:
            pytest.skip(
                "per_cell_baselines.json not yet generated — run "
                "experiments/scripts/measure_per_cell_baselines.py on a remote host."
            )
        rnd = row["random_baseline"]
        half_width = (rnd["ci95_hi"] - rnd["ci95_lo"]) / 2.0
        delta = abs(MINSPEC_RANDOM - rnd["mean"])
        assert delta < max(1.5 * half_width, 3.0), (
            f"canonical-cell random mean={rnd['mean']:.3f} differs from "
            f"MINSPEC_RANDOM={MINSPEC_RANDOM} by {delta:.3f}. "
            f"CI=[{rnd['ci95_lo']:.3f},{rnd['ci95_hi']:.3f}]."
        )
