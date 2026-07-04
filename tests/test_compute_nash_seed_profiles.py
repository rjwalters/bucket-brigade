"""
Tests for the ``--seed-profiles`` seeding hook in ``compute_nash.py``.

Issue #453 (follow-up to PR #452 / issue #445): the hook lets a JSON file of
named seed genomes be appended to the default archetype pool before
``DoubleOracle.solve()`` runs. PR #452 shipped the hook without committing its
test plan; these tests replicate that coverage so regressions are caught by CI.

Coverage:

1. ``load_seed_profiles`` happy path on the committed
   ``experiments/nash/seeds/rest_trap_battery_seeds.json`` (5 named genomes,
   shape (10,), values in [0, 1], expected name order).
2. Loader validation failures (all ``ValueError``): missing file
   (``FileNotFoundError``), malformed JSON, non-list top level, empty list,
   missing keys, wrong genome length, non-numeric entries, out-of-range values.
3. Stub test with ``DoubleOracle`` monkeypatched in the module:
   - Seeded path: ``solve()`` receives a 9-strategy pool (4 archetypes in
     solver-default order + 5 seeds), the log enumerates the pool, and
     ``equilibrium.json`` carries ``algorithm.seed_profiles`` provenance.
   - No-flag path: ``solve(initial_strategies=None)`` and
     ``algorithm.seed_profiles`` is ``null`` — behavior identical to the
     pre-#445 script.

The script is imported by path because ``experiments/scripts`` is not a
Python package (same pattern as ``tests/test_compute_nash_df_precheck.py``).
No real Double Oracle solves run: the solver is stubbed, so the suite is fast.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "experiments" / "scripts" / "compute_nash.py"
SEEDS_PATH = (
    REPO_ROOT / "experiments" / "nash" / "seeds" / "rest_trap_battery_seeds.json"
)

EXPECTED_SEED_NAMES = [
    "specialist_owned_house_firefighter",
    "firefighter_any_full_work",
    "conditional_firefighter_fire_responsive",
    "always_rest",
    "uniform_random",
]


def _load_compute_nash_module():
    """Load ``compute_nash.py`` as a standalone module for testing."""
    spec = importlib.util.spec_from_file_location(
        "compute_nash_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None, (
        f"Could not load spec for {SCRIPT_PATH}"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["compute_nash_under_test"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cn():
    """Provide the loaded ``compute_nash`` module."""
    return _load_compute_nash_module()


def _write_seeds(tmp_path: Path, payload) -> Path:
    """Write ``payload`` as JSON to a temp seeds file and return its path."""
    path = tmp_path / "seeds.json"
    path.write_text(json.dumps(payload))
    return path


def _valid_entry(name: str = "seed_a", value: float = 0.5) -> dict:
    return {"name": name, "genome": [value] * 10}


# ---------------------------------------------------------------------------
# 1. Loader happy path (committed seeds file)
# ---------------------------------------------------------------------------


def test_load_committed_seeds_file(cn):
    """The committed rest_trap seeds file loads as 5 named (10,) genomes."""
    profiles = cn.load_seed_profiles(SEEDS_PATH)

    assert [name for name, _ in profiles] == EXPECTED_SEED_NAMES
    for name, genome in profiles:
        assert isinstance(name, str)
        assert isinstance(genome, np.ndarray)
        assert genome.shape == (cn.GENOME_LENGTH,)
        assert genome.dtype == np.float64
        assert np.all(genome >= 0.0) and np.all(genome <= 1.0), (
            f"seed '{name}' has genome values outside [0, 1]"
        )


def test_load_committed_seeds_genomes_match_json(cn):
    """Loaded genome values are exactly the values in the JSON file."""
    with open(SEEDS_PATH) as f:
        raw = json.load(f)
    profiles = cn.load_seed_profiles(SEEDS_PATH)

    assert len(profiles) == len(raw)
    for (name, genome), entry in zip(profiles, raw):
        assert name == entry["name"]
        np.testing.assert_array_equal(
            genome, np.asarray(entry["genome"], dtype=np.float64)
        )


def test_loader_happy_path_synthetic(cn, tmp_path):
    """A minimal well-formed seeds file loads without error."""
    path = _write_seeds(tmp_path, [_valid_entry("only_seed", 0.25)])
    profiles = cn.load_seed_profiles(path)

    assert len(profiles) == 1
    name, genome = profiles[0]
    assert name == "only_seed"
    np.testing.assert_array_equal(genome, np.full(10, 0.25))


def test_loader_accepts_duplicate_names(cn, tmp_path):
    """Current behavior: duplicate seed names are not rejected by the loader.

    Documented here so any future dedup validation is a deliberate change.
    """
    path = _write_seeds(tmp_path, [_valid_entry("dup", 0.1), _valid_entry("dup", 0.9)])
    profiles = cn.load_seed_profiles(path)

    assert [name for name, _ in profiles] == ["dup", "dup"]


# ---------------------------------------------------------------------------
# 2. Loader validation failures
# ---------------------------------------------------------------------------


def test_loader_missing_file_raises(cn, tmp_path):
    """A nonexistent seeds file fails fast with FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        cn.load_seed_profiles(tmp_path / "does_not_exist.json")


def test_loader_malformed_json_raises(cn, tmp_path):
    """Malformed JSON raises (json.JSONDecodeError is a ValueError subclass)."""
    path = tmp_path / "seeds.json"
    path.write_text("{not valid json]")
    with pytest.raises(ValueError):
        cn.load_seed_profiles(path)


def test_loader_non_list_top_level_raises(cn, tmp_path):
    """A JSON object (not a list) at top level raises ValueError."""
    path = _write_seeds(tmp_path, {"name": "x", "genome": [0.5] * 10})
    with pytest.raises(ValueError, match="non-empty JSON list"):
        cn.load_seed_profiles(path)


def test_loader_empty_list_raises(cn, tmp_path):
    """An empty profile list raises ValueError."""
    path = _write_seeds(tmp_path, [])
    with pytest.raises(ValueError, match="non-empty JSON list"):
        cn.load_seed_profiles(path)


def test_loader_missing_genome_key_raises(cn, tmp_path):
    """An entry without a 'genome' key raises ValueError."""
    path = _write_seeds(tmp_path, [{"name": "no_genome"}])
    with pytest.raises(ValueError, match="'name' and 'genome'"):
        cn.load_seed_profiles(path)


def test_loader_missing_name_key_raises(cn, tmp_path):
    """An entry without a 'name' key raises ValueError."""
    path = _write_seeds(tmp_path, [{"genome": [0.5] * 10}])
    with pytest.raises(ValueError, match="'name' and 'genome'"):
        cn.load_seed_profiles(path)


def test_loader_non_dict_entry_raises(cn, tmp_path):
    """A bare list entry (not an object) raises ValueError."""
    path = _write_seeds(tmp_path, [[0.5] * 10])
    with pytest.raises(ValueError, match="'name' and 'genome'"):
        cn.load_seed_profiles(path)


@pytest.mark.parametrize("length", [0, 9, 11])
def test_loader_wrong_genome_length_raises(cn, tmp_path, length):
    """Genomes must have exactly GENOME_LENGTH (10) entries."""
    path = _write_seeds(tmp_path, [{"name": "bad_len", "genome": [0.5] * length}])
    with pytest.raises(ValueError, match="length"):
        cn.load_seed_profiles(path)


def test_loader_non_numeric_genome_raises(cn, tmp_path):
    """Non-numeric genome entries fail float64 conversion with ValueError."""
    genome = [0.5] * 9 + ["not_a_number"]
    path = _write_seeds(tmp_path, [{"name": "bad_type", "genome": genome}])
    with pytest.raises(ValueError):
        cn.load_seed_profiles(path)


@pytest.mark.parametrize("bad_value", [-0.1, 1.1])
def test_loader_out_of_range_values_raise(cn, tmp_path, bad_value):
    """Genome values outside [0, 1] raise ValueError."""
    genome = [0.5] * 9 + [bad_value]
    path = _write_seeds(tmp_path, [{"name": "out_of_range", "genome": genome}])
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        cn.load_seed_profiles(path)


# ---------------------------------------------------------------------------
# 3. Stub tests: pool enumeration + equilibrium.json provenance
# ---------------------------------------------------------------------------


def _install_fake_double_oracle(cn, monkeypatch):
    """Replace ``DoubleOracle`` in the module with a recording stub.

    Returns a dict that captures the ``initial_strategies`` passed to
    ``solve()`` (key present only after solve() is called).
    """
    captured: dict = {}

    class _FakeEquilibrium:
        def __init__(self, strategy_pool):
            self.strategy_pool = strategy_pool
            self.distribution = {0: 1.0}
            self.payoff = 123.0
            self.converged = True
            self.iterations = 1

    class _FakeDoubleOracle:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs

        def solve(self, initial_strategies=None):
            captured["initial_strategies"] = initial_strategies
            if initial_strategies is not None:
                pool = [s.copy() for s in initial_strategies]
            else:
                # Mirror the solver's fallback: default archetype pool.
                pool = [g.copy() for _, g in cn.DEFAULT_ARCHETYPE_POOL]
            return _FakeEquilibrium(pool)

    monkeypatch.setattr(cn, "DoubleOracle", _FakeDoubleOracle)
    return captured


def test_seeded_path_pool_and_provenance(cn, monkeypatch, tmp_path, capsys):
    """--seed-profiles: 9-strategy pool passed to solve(), provenance in JSON."""
    captured = _install_fake_double_oracle(cn, monkeypatch)
    output_dir = tmp_path / "nash_out"

    cn.compute_nash_equilibrium(
        "rest_trap",
        output_dir,
        num_simulations=1,
        max_iterations=1,
        seed=42,
        seed_profiles_path=SEEDS_PATH,
        verbose=False,
    )

    # solve() received the seeded pool: 4 archetypes + 5 seeds = 9 strategies.
    pool = captured["initial_strategies"]
    assert pool is not None
    assert len(pool) == 9

    # First 4 entries byte-equal to the archetype params in solver-default
    # order (firefighter, free_rider, hero, coordinator).
    expected_archetypes = [
        cn.FIREFIGHTER_PARAMS,
        cn.FREE_RIDER_PARAMS,
        cn.HERO_PARAMS,
        cn.COORDINATOR_PARAMS,
    ]
    for i, expected in enumerate(expected_archetypes):
        assert pool[i].tobytes() == np.asarray(expected, dtype=np.float64).tobytes(), (
            f"pool[{i}] is not byte-equal to the expected archetype"
        )

    # Seeds follow in file order; the specialist genome sits at index 4.
    seeds = cn.load_seed_profiles(SEEDS_PATH)
    for i, (name, genome) in enumerate(seeds):
        np.testing.assert_array_equal(pool[4 + i], genome)
    assert seeds[0][0] == "specialist_owned_house_firefighter"
    np.testing.assert_array_equal(pool[4], seeds[0][1])

    # The log enumerates the seeded pool.
    out = capsys.readouterr().out
    assert "Seeded Initial Strategy Pool (9 strategies)" in out
    assert f"Seed profiles file: {SEEDS_PATH}" in out
    assert "[0] (archetype) firefighter" in out
    assert "[4] (SEED) specialist_owned_house_firefighter" in out
    assert "[8] (SEED) uniform_random" in out

    # equilibrium.json carries seed provenance under algorithm.seed_profiles.
    with open(output_dir / "equilibrium.json") as f:
        results = json.load(f)
    provenance = results["algorithm"]["seed_profiles"]
    assert provenance is not None
    assert provenance["file"] == str(SEEDS_PATH)
    assert provenance["names"] == EXPECTED_SEED_NAMES


def test_no_flag_path_unchanged(cn, monkeypatch, tmp_path, capsys):
    """No --seed-profiles: solve(initial_strategies=None), null provenance."""
    captured = _install_fake_double_oracle(cn, monkeypatch)
    output_dir = tmp_path / "nash_out"

    cn.compute_nash_equilibrium(
        "rest_trap",
        output_dir,
        num_simulations=1,
        max_iterations=1,
        seed=42,
        seed_profiles_path=None,
        verbose=False,
    )

    # The solver falls back to its own default archetype pool.
    assert captured["initial_strategies"] is None

    # No seeded-pool banner is printed.
    out = capsys.readouterr().out
    assert "Seeded Initial Strategy Pool" not in out

    # Provenance field is present but null.
    with open(output_dir / "equilibrium.json") as f:
        results = json.load(f)
    assert results["algorithm"]["seed_profiles"] is None


def test_default_archetype_pool_matches_solver_default(cn):
    """DEFAULT_ARCHETYPE_POOL mirrors DoubleOracle.solve()'s fallback pool.

    The solver default (bucket_brigade/equilibrium/double_oracle.py, solve())
    is [FIREFIGHTER, FREE_RIDER, HERO, COORDINATOR] built from the archetype
    constants. The script's copy must stay byte-identical and in the same
    order so the seeded pool is a strict superset of the no-flag pool.
    """
    from bucket_brigade.agents.archetypes import (
        COORDINATOR_PARAMS,
        FIREFIGHTER_PARAMS,
        FREE_RIDER_PARAMS,
        HERO_PARAMS,
    )

    expected = [
        ("firefighter", FIREFIGHTER_PARAMS),
        ("free_rider", FREE_RIDER_PARAMS),
        ("hero", HERO_PARAMS),
        ("coordinator", COORDINATOR_PARAMS),
    ]
    assert len(cn.DEFAULT_ARCHETYPE_POOL) == len(expected)
    for (name, genome), (exp_name, exp_genome) in zip(
        cn.DEFAULT_ARCHETYPE_POOL, expected
    ):
        assert name == exp_name
        assert (
            np.asarray(genome, dtype=np.float64).tobytes()
            == np.asarray(exp_genome, dtype=np.float64).tobytes()
        )
