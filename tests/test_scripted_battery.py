"""Tests for the all-scripted team battery (issue #436, Part A).

Modeled on ``tests/test_improvability_oracle.py`` (importlib loading of an
``experiments/`` script, tiny sequential episode counts — fast, no training,
no network).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from bucket_brigade.baselines.per_cell import _run_random_episode, _seeds_for
from bucket_brigade.envs.scenarios_generated import get_scenario_by_name

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "experiments" / "p3_specialization" / "scripted_battery.py"

spec = importlib.util.spec_from_file_location("scripted_battery", SCRIPT_PATH)
battery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(battery)

BATTERY_NAMES = (
    "uniform",
    "always_rest",
    "specialist",
    "firefighter[owned,work=1.00]x4",
    "firefighter[any,work=1.00]x4",
    "1xfirefighter[any]+3xrest",
    "2xfirefighter[any]+2xrest",
    "3xfirefighter[any]+1xrest",
)


def _run_main(tmp_path: Path, name: str, extra: list[str] | None = None) -> dict:
    out_dir = tmp_path / name
    argv = [
        "--scenario",
        "rest_trap",
        "--n-episodes",
        "4",
        "--n-episodes-final",
        "4",
        "--n-boot",
        "20",
        "--num-workers",
        "1",
        "--random-search",
        "0",
        "--out-dir",
        str(out_dir),
    ] + (extra or [])
    rc = battery.main(argv)
    assert rc == 0
    assert (out_dir / "rest_trap.md").exists()
    return json.loads((out_dir / "rest_trap.json").read_text())


# ---------------------------------------------------------------------------
# Battery construction
# ---------------------------------------------------------------------------


class TestBattery:
    def test_base_battery_names(self):
        specs = battery.battery_specs(random_search=0, seed=0)
        assert [s["name"] for s in specs] == list(BATTERY_NAMES)
        assert specs[0]["name"] == "uniform"  # evaluate_battery relies on this

    def test_every_spec_has_four_members(self):
        for s in battery.battery_specs(random_search=3, seed=0):
            assert len(s["members"]) == battery.NUM_AGENTS

    def test_random_search_specs_deterministic(self):
        a = battery.battery_specs(random_search=8, seed=0)
        b = battery.battery_specs(random_search=8, seed=0)
        assert [s["name"] for s in a] == [s["name"] for s in b]
        assert all(s.get("from_random_search") for s in a[len(BATTERY_NAMES) :])

    def test_ne_shaped_profiles_mix_firefighters_and_rest(self):
        specs = {s["name"]: s for s in battery.battery_specs(0, 0)}
        mix = specs["1xfirefighter[any]+3xrest"]["members"]
        assert mix[0]["kind"] == "firefighter" and not mix[0]["scope_owned_only"]
        assert all(m["kind"] == "always_rest" for m in mix[1:])


# ---------------------------------------------------------------------------
# Structural sanity
# ---------------------------------------------------------------------------


class TestStructuralSanity:
    def test_uniform_bit_identical_to_per_cell_random_episode(self):
        """The uniform team replays ``per_cell._run_random_episode``
        bit-identically for the same (scenario, seed) — the same convention
        behind ``SCENARIO_RANDOM_BASELINES``."""
        scenario = get_scenario_by_name("rest_trap", num_agents=4)
        uniform = {"name": "uniform", "members": [{"kind": "uniform"}] * 4}
        for seed in _seeds_for(0, 5):
            expected = _run_random_episode((scenario, seed))
            team, _wr = battery._run_team_episode((scenario, seed, uniform))
            assert team == expected

    def test_always_rest_never_works(self):
        scenario = get_scenario_by_name("rest_trap", num_agents=4)
        rest = {"name": "always_rest", "members": [{"kind": "always_rest"}] * 4}
        for seed in _seeds_for(1, 3):
            _team, work_rate = battery._run_team_episode((scenario, seed, rest))
            assert work_rate == 0.0

    def test_specialist_equals_owned_firefighter_team(self):
        """specialist x4 and firefighter[owned, work=1.0] x4 are behaviorally
        identical — trajectories must match exactly."""
        scenario = get_scenario_by_name("rest_trap", num_agents=4)
        spec_specialist = {
            "name": "specialist",
            "members": [{"kind": "specialist"}] * 4,
        }
        spec_ff = {
            "name": "ff_owned",
            "members": [
                {"kind": "firefighter", "scope_owned_only": True, "work_prob": 1.0}
            ]
            * 4,
        }
        for seed in _seeds_for(3, 5):
            a = battery._run_team_episode((scenario, seed, spec_specialist))
            b = battery._run_team_episode((scenario, seed, spec_ff))
            assert a == b


# ---------------------------------------------------------------------------
# Output schema + determinism (end-to-end at tiny n)
# ---------------------------------------------------------------------------


class TestSchema:
    def test_output_schema(self, tmp_path):
        result = _run_main(tmp_path, "schema")
        assert result["issue"] == 436
        assert result["scenario"] == "rest_trap"
        assert result["cited_random_baseline"] == pytest.approx(302.87)

        screen = result["screen"]
        assert set(screen["policies"].keys()) == set(BATTERY_NAMES)
        for name, rec in screen["policies"].items():
            t = rec["team"]
            assert np.isfinite(t["mean"])
            assert t["ci95_lo"] <= t["mean"] <= t["ci95_hi"]
            assert 0.0 <= rec["team_work_rate"] <= 1.0
            assert rec["n_episodes"] == 4
            if name != "uniform":
                d = rec["team_delta_vs_uniform_paired"]
                assert d["ci95_lo"] <= d["mean"] <= d["ci95_hi"]
        assert screen["screen_best"]["name"] in BATTERY_NAMES

        final = result["final"]
        assert final["n_episodes"] == 4
        assert final["winner"]["name"] == screen["screen_best"]["name"]
        # Final stage uses fresh seeds (offset from the screen stage).
        assert final["stage_seed"] != result["config"]["seed"]

        sb = result["scripted_best"]
        assert sb["name"] == final["winner"]["name"]
        assert sb["ci95_lo"] <= sb["value"] <= sb["ci95_hi"]
        assert isinstance(sb["beats_random"], bool)

        prov = result["provenance"]
        assert prov["host"]
        assert prov["git_sha"]

    def test_deterministic_given_seed(self, tmp_path):
        a = _run_main(tmp_path, "det_a", extra=["--seed", "7"])
        b = _run_main(tmp_path, "det_b", extra=["--seed", "7"])
        # Provenance (host/sha) is environment-dependent; everything
        # measured must be identical.
        for key in ("screen", "final", "scripted_best"):
            assert a[key] == b[key]

    def test_random_search_included(self, tmp_path):
        result = _run_main(tmp_path, "search", extra=["--random-search", "2"])
        rs = result["screen"]["random_search"]
        assert rs["n_samples"] == 2
        assert len(rs["samples"]) == 2


# ---------------------------------------------------------------------------
# Markdown rendering (both verdict branches, synthetic input — no episodes)
# ---------------------------------------------------------------------------


def _synthetic_result(*, beats_random: bool) -> dict:
    """Minimal result dict exercising ``render_markdown`` without running
    any episodes (PR #440 review: the ``beats_random=False`` branch was
    previously untested)."""
    team = {"mean": 300.0, "ci95_lo": 298.0, "ci95_hi": 302.0}
    delta = {"mean": -2.0, "ci95_lo": -4.0, "ci95_hi": 0.5}
    return {
        "scenario": "rest_trap",
        "config": {
            "n_episodes": 4,
            "n_episodes_final": 4,
            "seed": 0,
            "n_boot": 20,
            "random_search": 0,
        },
        "provenance": {"host": "testhost", "git_sha": "deadbeef"},
        "cited_random_baseline": 302.87,
        "screen": {
            "policies": {
                "uniform": {"team": team, "team_work_rate": 0.5, "n_episodes": 4},
                "always_rest": {
                    "team": team,
                    "team_delta_vs_uniform_paired": delta,
                    "team_work_rate": 0.0,
                    "n_episodes": 4,
                },
            },
            "screen_best": {"name": "always_rest"},
        },
        "final": {
            "n_episodes": 4,
            "uniform": {"team": team},
            "winner": {"name": "always_rest", "team": team},
            "winner_delta_vs_uniform_paired": delta,
        },
        "scripted_best": {
            "name": "always_rest",
            "value": 300.0,
            "ci95_lo": 298.0,
            "ci95_hi": 302.0,
            "n_episodes": 4,
            "beats_random": beats_random,
        },
    }


class TestRenderMarkdown:
    def test_beats_random_false_branch(self):
        md = battery.render_markdown(_synthetic_result(beats_random=False))
        assert "no battery member beats the uniform-random baseline" in md
        assert "`scripted_best` <= random" in md
        assert "recorded as absent" in md
        # The failure-mode verdict must NOT claim an upper anchor exists.
        assert "decisively above" not in md

    def test_beats_random_true_branch(self):
        md = battery.render_markdown(_synthetic_result(beats_random=True))
        assert "measured `scripted_best` upper anchor" in md
        assert "decisively above" in md
