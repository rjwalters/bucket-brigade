"""Tests for the k = 1 improvability oracle (issue #428).

Modeled on ``tests/test_per_cell_baselines.py`` (episode conventions) and
``tests/test_entropy_vs_trainability.py`` (importlib loading of an
``experiments/`` script). All tests run tiny episode counts sequentially —
fast, no training, no network.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from bucket_brigade.baselines.per_cell import (
    _run_random_episode,
    _seeds_for,
    make_phase_diagram_scenario,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT / "experiments" / "nash" / "phase_diagram" / "improvability_oracle.py"
)

spec = importlib.util.spec_from_file_location("improvability_oracle", SCRIPT_PATH)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)

BATTERY_NAMES = (
    "uniform",
    "always_rest",
    "specialist",
    "firefighter[owned,work=1.00]",
    "firefighter[any,work=1.00]",
)


def _run_main(tmp_path: Path, name: str, extra: list[str] | None = None) -> dict:
    out_json = tmp_path / f"{name}.json"
    out_md = tmp_path / f"{name}.md"
    argv = [
        "--cells",
        "b0.10_k0.10_c0.50",
        "--n-episodes",
        "4",
        "--n-boot",
        "20",
        "--num-workers",
        "1",
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
    ] + (extra or [])
    rc = oracle.main(argv)
    assert rc == 0
    assert out_md.exists()
    return json.loads(out_json.read_text())


# ---------------------------------------------------------------------------
# cell-tag parsing
# ---------------------------------------------------------------------------


class TestParseCellTag:
    def test_parses_default_cells(self):
        assert oracle.parse_cell_tag("b0.10_k0.10_c0.50") == (0.10, 0.10, 0.50)
        assert oracle.parse_cell_tag("b0.90_k0.10_c2.00") == (0.90, 0.10, 2.00)

    def test_rejects_malformed_tag(self):
        with pytest.raises(ValueError, match="cell tag"):
            oracle.parse_cell_tag("beta0.1_k0.1_c0.5")

    def test_c2_cell_builds_scenario(self):
        """Edge case from the issue test plan: a c2.00 tag must build without
        error (the other three no_convergence cells for a follow-up)."""
        beta, kappa, cost = oracle.parse_cell_tag("b0.50_k0.10_c2.00")
        s = make_phase_diagram_scenario(beta, kappa, cost)
        assert s.cost_to_work_one_night == 2.0
        assert s.prob_solo_agent_extinguishes_fire == 0.1


# ---------------------------------------------------------------------------
# (a) output JSON schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_output_schema(self, tmp_path):
        result = _run_main(tmp_path, "schema", extra=["--random-search", "2"])
        assert result["issue"] == 428
        assert set(result["cells"].keys()) == {"b0.10_k0.10_c0.50"}
        cell = result["cells"]["b0.10_k0.10_c0.50"]
        assert cell["cell_tag"] == "b0.10_k0.10_c0.50"
        assert cell["beta"] == 0.10
        assert cell["kappa"] == 0.10
        assert cell["c"] == 0.50

        # All battery policies present with team + BR-agent metrics and CIs.
        assert set(cell["policies"].keys()) == set(BATTERY_NAMES)
        for name, rec in cell["policies"].items():
            for metric in ("team", "br_agent"):
                m = rec[metric]
                assert np.isfinite(m["mean"])
                assert m["ci95_lo"] <= m["mean"] <= m["ci95_hi"]
            assert 0.0 <= rec["br_work_rate"] <= 1.0
            assert rec["n_episodes"] == 4
            if name != "uniform":
                d = rec["team_delta_vs_uniform_paired"]
                assert d["ci95_lo"] <= d["mean"] <= d["ci95_hi"]

        # Headroom summary present.
        head = cell["headroom"]
        assert head["best_team_policy"]
        assert np.isfinite(head["team_pct_over_uniform"])

        # Random-search section present with the requested sample count.
        assert cell["random_search"]["n_samples"] == 2
        assert len(cell["random_search"]["samples"]) == 2

    def test_random_search_zero_skips_cleanly(self, tmp_path):
        result = _run_main(tmp_path, "nosearch", extra=["--random-search", "0"])
        cell = result["cells"]["b0.10_k0.10_c0.50"]
        assert "random_search" not in cell
        assert set(cell["policies"].keys()) == set(BATTERY_NAMES)


# ---------------------------------------------------------------------------
# (b) determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_identical_json(self, tmp_path):
        a = _run_main(tmp_path, "det_a", extra=["--seed", "42"])
        b = _run_main(tmp_path, "det_b", extra=["--seed", "42"])
        assert a == b

    def test_different_seed_differs(self, tmp_path):
        a = _run_main(tmp_path, "seed_a", extra=["--seed", "1"])
        b = _run_main(tmp_path, "seed_b", extra=["--seed", "2"])
        a_mean = a["cells"]["b0.10_k0.10_c0.50"]["policies"]["uniform"]["team"]["mean"]
        b_mean = b["cells"]["b0.10_k0.10_c0.50"]["policies"]["uniform"]["team"]["mean"]
        assert a_mean != b_mean


# ---------------------------------------------------------------------------
# (c) structural sanity
# ---------------------------------------------------------------------------


class TestStructuralSanity:
    def test_always_rest_never_works(self, tmp_path):
        """always_rest must contribute exactly zero work cost: BR work rate
        is identically 0 (structural, not statistical)."""
        result = _run_main(tmp_path, "rest")
        cell = result["cells"]["b0.10_k0.10_c0.50"]
        assert cell["policies"]["always_rest"]["br_work_rate"] == 0.0

    def test_uniform_bit_identical_to_per_cell_random_episode(self):
        """The uniform battery member replays ``per_cell._run_random_episode``
        bit-identically for the same (scenario, seed) — the whole point of
        drawing the full joint uniform block before overwriting agent 0."""
        scenario = make_phase_diagram_scenario(0.1, 0.1, 0.5)
        spec_uniform = {"name": "uniform", "kind": "uniform"}
        for seed in _seeds_for(0, 5):
            expected = _run_random_episode((scenario, seed))
            team, _br, _wr = oracle._run_oracle_episode((scenario, seed, spec_uniform))
            assert team == expected

    def test_specialist_equals_owned_firefighter_at_work_1(self):
        """For agent 0 the specialist policy and firefighter[owned, work=1.0]
        are behaviorally identical (work lowest-index burning owned house,
        else rest) — trajectories must match exactly."""
        scenario = make_phase_diagram_scenario(0.1, 0.1, 0.5)
        spec_specialist = {"name": "specialist", "kind": "specialist"}
        spec_ff = {
            "name": "ff",
            "kind": "firefighter",
            "scope_owned_only": True,
            "work_prob": 1.0,
        }
        for seed in _seeds_for(3, 5):
            a = oracle._run_oracle_episode((scenario, seed, spec_specialist))
            b = oracle._run_oracle_episode((scenario, seed, spec_ff))
            assert a == b


# ---------------------------------------------------------------------------
# battery construction
# ---------------------------------------------------------------------------


class TestBattery:
    def test_base_battery_names(self):
        specs = oracle.battery_specs(random_search=0, seed=0)
        assert [s["name"] for s in specs] == list(BATTERY_NAMES)
        assert specs[0]["kind"] == "uniform"  # evaluate_cell relies on this

    def test_random_search_specs_deterministic(self):
        a = oracle.battery_specs(random_search=8, seed=0)
        b = oracle.battery_specs(random_search=8, seed=0)
        assert [s["name"] for s in a] == [s["name"] for s in b]
        assert all(s.get("from_random_search") for s in a[len(BATTERY_NAMES) :])
