"""Tests for ``experiments/nash/phase_diagram/exploitability/crn_deviation_check.py``.

Issue #459. Covers the statistical/plumbing helpers without running the full
n=20000 CRN sweep:

1. ``make_seeds`` determinism (CRN precondition).
2. ``paired_stats`` mean/SE/t arithmetic, including the all-zero null case.
3. ``build_candidates`` dedupe (incumbent excluded, archetype-equal
   harness BR genomes not duplicated).
4. ``per_position_rewards`` determinism at a tiny seed count (Rust engine:
   same genomes + same seeds → bit-identical rewards), which is the property
   the script's null control relies on.

Imported by path because the script lives outside a package (same pattern
as ``tests/test_mixture_exploitability.py``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = (
    REPO_ROOT
    / "experiments"
    / "nash"
    / "phase_diagram"
    / "exploitability"
    / "crn_deviation_check.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "crn_deviation_check_under_test", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_module()


class TestMakeSeeds:
    def test_deterministic(self, mod):
        assert mod.make_seeds(10, 42) == mod.make_seeds(10, 42)

    def test_seed_changes_list(self, mod):
        assert mod.make_seeds(10, 42) != mod.make_seeds(10, 43)


class TestPairedStats:
    def test_basic(self, mod):
        stats = mod.paired_stats(np.array([1.0, 2.0, 3.0]))
        assert stats["mean"] == pytest.approx(2.0)
        assert stats["se"] == pytest.approx(1.0 / np.sqrt(3))
        assert stats["t"] == pytest.approx(2.0 * np.sqrt(3))

    def test_all_zero_null_deviation(self, mod):
        """Identical-behavior deviations must not divide by zero."""
        stats = mod.paired_stats(np.zeros(100))
        assert stats["mean"] == 0.0
        assert stats["t"] == 0.0


class TestBuildCandidates:
    def test_incumbent_archetype_excluded(self, mod):
        cands = mod.build_candidates(0, mod.ARCHETYPES["firefighter"], {})
        names = [n for n, _ in cands]
        assert "archetype:firefighter" not in names
        assert len(names) == 4  # the other 4 archetypes

    def test_harness_br_included_for_position(self, mod):
        br = {1: [0.5] * 10}
        cands = mod.build_candidates(1, mod.ARCHETYPES["hero"], br)
        names = [n for n, _ in cands]
        assert "harness_br" in names
        # Other positions don't get this BR genome.
        cands0 = mod.build_candidates(0, mod.ARCHETYPES["hero"], br)
        assert "harness_br" not in [n for n, _ in cands0]

    def test_harness_br_deduped_against_archetype(self, mod):
        """A BR genome that IS an archetype appears once, as the archetype."""
        br = {2: [float(v) for v in mod.ARCHETYPES["liar"]]}
        cands = mod.build_candidates(2, mod.ARCHETYPES["hero"], br)
        names = [n for n, _ in cands]
        assert names.count("harness_br") == 0
        assert "archetype:liar" in names


class TestPerPositionRewards:
    def test_crn_determinism(self, mod):
        """Same genomes + same seeds → bit-identical (the null-control basis)."""
        genomes = [[float(v) for v in mod.ARCHETYPES["firefighter"]]] * 4
        seeds = mod.make_seeds(3, 7)
        a = mod.per_position_rewards(genomes, "asym_b05_k09_c05", seeds)
        b = mod.per_position_rewards(genomes, "asym_b05_k09_c05", seeds)
        assert a.shape == (3, 4)
        assert np.array_equal(a, b)
