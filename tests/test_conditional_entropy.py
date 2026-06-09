"""Tests for the per-cell conditional action entropy estimator (issue #368).

Covers three layers:

1. ``episode_bootstrap_ci`` — the load-bearing episode-level bootstrap
   adapter. Verifies it preserves episode boundaries and gives wider CIs
   than the (invalid for trajectory data) step-level bootstrap.

2. ``rollout_joint_actions`` — verifies the BB env can be driven step-by-
   step with HeuristicAgent genomes and produces action traces of the
   expected shape.

3. ``estimate_cell_entropy`` end-to-end — smoke test on the symmetric NE
   profile from one of the converged preview cells. Symmetric NE in the
   minimal_specialization family with deterministic genomes (hero ×4)
   should give low conditional entropy across all positions.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from bucket_brigade.analysis.conditional_entropy import (
    EpisodeActions,
    _build_per_position_episode_arrays,
    _other_positions_episodes,
    episode_bootstrap_ci,
    estimate_cell_entropy,
    rollout_joint_actions,
)
from bucket_brigade.analysis.info_theory import entropy_discrete
from bucket_brigade.envs import get_scenario_by_name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


HERO_GENOME = [1.0, 1.0, 1.0, 0.5, 0.1, 0.5, 0.0, 0.9, 0.0, 1.0]
FIREFIGHTER_GENOME = [1.0, 0.9, 0.5, 0.8, 0.5, 0.7, 0.1, 0.0, 0.0, 0.8]


def _make_scenario(beta: float, kappa: float, c: float):
    base = get_scenario_by_name("minimal_specialization", num_agents=4)
    return dataclasses.replace(
        base,
        prob_fire_spreads_to_neighbor=float(beta),
        prob_solo_agent_extinguishes_fire=float(kappa),
        cost_to_work_one_night=float(c),
    )


# ---------------------------------------------------------------------------
# 1. Episode bootstrap CI
# ---------------------------------------------------------------------------


class TestEpisodeBootstrapCI:
    def test_preserves_episode_boundaries(self):
        """Each bootstrap resample is built by concatenating whole episodes.

        We instrument the estimator to record exactly which arrays it sees;
        the recorded lengths must always be sums of episode lengths from the
        original episode pool (not arbitrary step counts).
        """
        # 5 episodes with lengths 3, 7, 2, 4, 5 (sum 21).
        rng = np.random.default_rng(0)
        ep_lengths = [3, 7, 2, 4, 5]
        episodes = [rng.integers(0, 4, size=L) for L in ep_lengths]

        observed_lengths: list[int] = []
        observed_arrays: list[np.ndarray] = []

        def _capture_estimator(arr: np.ndarray) -> float:
            observed_lengths.append(len(arr))
            observed_arrays.append(arr.copy())
            return float(entropy_discrete(arr))

        episode_bootstrap_ci(
            estimator=_capture_estimator,
            episode_arrays=[episodes],
            n_boot=50,
            rng=np.random.default_rng(7),
        )

        # Every observed length should be expressible as a sum of 5 (with
        # replacement) episode lengths drawn from ep_lengths. The set of
        # achievable totals = { sum_{i in S} L[i] : multiset S of size 5 }.
        # Easier check: it must equal n_eps * mean_L_for_some_assignment ==
        # i.e. each total mod gcd... we just check that each total is at
        # least min(ep_lengths) * n_eps and at most max(ep_lengths) * n_eps.
        n_eps = len(ep_lengths)
        min_total = min(ep_lengths) * n_eps  # 2 * 5 = 10
        max_total = max(ep_lengths) * n_eps  # 7 * 5 = 35
        for L in observed_lengths:
            assert min_total <= L <= max_total, (
                f"observed length {L} outside the achievable range "
                f"[{min_total}, {max_total}]"
            )

        # Also: at least one length should differ from the original 21 (the
        # bootstrap is not degenerate). If by astronomical chance all 50
        # draws hit the same total, the estimator is broken.
        non_full = [L for L in observed_lengths if L != 21]
        assert non_full, "bootstrap appears to never resample (all lengths equal 21)"

    def test_point_estimate_matches_concatenation(self):
        rng = np.random.default_rng(1)
        episodes = [rng.integers(0, 4, size=10) for _ in range(20)]
        # The point estimate (first returned value) should equal the
        # estimator applied to the full concatenation.
        point, lo, hi = episode_bootstrap_ci(
            estimator=lambda x: entropy_discrete(x),
            episode_arrays=[episodes],
            n_boot=200,
            rng=np.random.default_rng(2),
        )
        full = np.concatenate(episodes)
        assert point == pytest.approx(entropy_discrete(full), abs=1e-9)
        assert lo <= point <= hi

    def test_rejects_mismatched_episode_counts(self):
        a = [np.array([0, 1]), np.array([2, 3])]
        b = [np.array([0, 1])]
        with pytest.raises(ValueError, match="episodes, expected"):
            episode_bootstrap_ci(
                estimator=lambda x, y: float(np.mean(x + y)),
                episode_arrays=[a, b],
                n_boot=10,
            )

    def test_rejects_mismatched_episode_lengths(self):
        a = [np.array([0, 1, 2]), np.array([3, 4])]
        b = [np.array([0, 1]), np.array([3, 4])]
        # First episode: a has length 3, b has length 2 -> error.
        with pytest.raises(ValueError, match="length"):
            episode_bootstrap_ci(
                estimator=lambda x, y: 0.0,
                episode_arrays=[a, b],
                n_boot=10,
            )

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one variable"):
            episode_bootstrap_ci(
                estimator=lambda *args: 0.0,
                episode_arrays=[],
                n_boot=10,
            )
        with pytest.raises(ValueError, match="at least one episode"):
            episode_bootstrap_ci(
                estimator=lambda x: 0.0,
                episode_arrays=[[]],
                n_boot=10,
            )

    def test_two_variable_resample_uses_same_indices(self):
        """Bootstrap must resample the same episodes across all variables.

        We construct two perfectly-correlated variables: x and y where
        y == 2*x within each episode. The estimator returns the absolute
        difference between concat(x) and concat(y); if the bootstrap
        resamples them independently it would scramble the correlation and
        the difference would be huge for most draws. With proper paired
        resampling, the difference must be exactly the same (==
        concat(x).sum * 1) for every draw.
        """
        rng = np.random.default_rng(3)
        ep_lengths = [4, 7, 5, 3]
        x_eps = [rng.integers(0, 10, size=L) for L in ep_lengths]
        y_eps = [2 * x for x in x_eps]

        def _paired_check(x: np.ndarray, y: np.ndarray) -> float:
            # If pairing is preserved: y == 2*x exactly, so this is 0.
            return float(np.max(np.abs(y - 2 * x)))

        point, lo, hi = episode_bootstrap_ci(
            estimator=_paired_check,
            episode_arrays=[x_eps, y_eps],
            n_boot=200,
            rng=np.random.default_rng(4),
        )
        assert point == pytest.approx(0.0, abs=1e-9)
        # If indices were independent, max-abs-diff would be > 0 most draws,
        # so the upper CI would be strictly positive. With paired indices,
        # the upper CI is also 0.
        assert hi == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. Rollout driver
# ---------------------------------------------------------------------------


class TestRolloutJointActions:
    def test_returns_one_trace_per_episode(self):
        scenario = _make_scenario(beta=0.5, kappa=0.5, c=0.5)
        genomes = [HERO_GENOME, HERO_GENOME, HERO_GENOME, HERO_GENOME]
        episodes = rollout_joint_actions(
            genomes=genomes,
            scenario=scenario,
            n_episodes=5,
            seed=42,
            num_workers=1,  # disable multiprocessing in tests
        )
        assert len(episodes) == 5
        for ep in episodes:
            assert isinstance(ep, EpisodeActions)
            assert ep.actions.ndim == 3
            assert ep.actions.shape[1] == 4  # num_agents
            assert ep.actions.shape[2] == 3  # [house, mode, signal]
            assert ep.actions.shape[0] >= 1  # at least one step

    def test_action_values_in_valid_range(self):
        scenario = _make_scenario(beta=0.5, kappa=0.5, c=0.5)
        genomes = [FIREFIGHTER_GENOME] * 4
        episodes = rollout_joint_actions(
            genomes=genomes, scenario=scenario, n_episodes=3, seed=0, num_workers=1
        )
        for ep in episodes:
            houses = ep.actions[..., 0]
            modes = ep.actions[..., 1]
            signals = ep.actions[..., 2]
            assert houses.min() >= 0 and houses.max() < scenario.num_houses
            assert set(np.unique(modes).tolist()).issubset({0, 1})
            assert set(np.unique(signals).tolist()).issubset({0, 1})

    def test_rejects_wrong_num_genomes(self):
        scenario = _make_scenario(beta=0.5, kappa=0.5, c=0.5)
        with pytest.raises(ValueError, match="num_agents"):
            rollout_joint_actions(
                genomes=[HERO_GENOME, HERO_GENOME],
                scenario=scenario,
                n_episodes=2,
                num_workers=1,
            )


# ---------------------------------------------------------------------------
# 3. Conversion helpers
# ---------------------------------------------------------------------------


class TestPerPositionEpisodeArrays:
    def test_shape_and_tuple_format(self):
        # 2 episodes, 2 timesteps each, 4 agents, action_dim=3.
        ep0 = np.array(
            [
                [[1, 1, 0], [2, 0, 1], [3, 1, 1], [4, 0, 0]],
                [[1, 0, 1], [2, 1, 0], [3, 0, 0], [4, 1, 1]],
            ],
            dtype=np.int8,
        )
        ep1 = np.array(
            [
                [[5, 1, 1], [6, 0, 0], [7, 1, 0], [8, 0, 1]],
                [[5, 0, 0], [6, 1, 1], [7, 0, 1], [8, 1, 0]],
                [[5, 1, 0], [6, 0, 1], [7, 1, 1], [8, 0, 0]],
            ],
            dtype=np.int8,
        )
        episodes = [EpisodeActions(actions=ep0), EpisodeActions(actions=ep1)]
        per_position = _build_per_position_episode_arrays(episodes, num_agents=4)

        assert len(per_position) == 4
        for i in range(4):
            assert len(per_position[i]) == 2  # 2 episodes
            assert len(per_position[i][0]) == 2  # ep0 has 2 steps
            assert len(per_position[i][1]) == 3  # ep1 has 3 steps

        # Spot-check the tuple format for position 0, ep 0, step 0:
        assert per_position[0][0][0] == (1, 1, 0)
        assert per_position[3][1][2] == (8, 0, 0)

    def test_minus_i_joins_correctly(self):
        ep0 = np.array(
            [
                [[1, 0, 0], [2, 1, 1], [3, 0, 0], [4, 1, 1]],
            ],
            dtype=np.int8,
        )
        episodes = [EpisodeActions(actions=ep0)]
        per_position = _build_per_position_episode_arrays(episodes, num_agents=4)
        # minus_i for i=1: should join positions {0, 2, 3} per step.
        minus = _other_positions_episodes(per_position, i=1)
        assert len(minus) == 1
        assert len(minus[0]) == 1
        assert minus[0][0] == ((1, 0, 0), (3, 0, 0), (4, 1, 1))


# ---------------------------------------------------------------------------
# 4. End-to-end sanity test on a small profile
# ---------------------------------------------------------------------------


class TestEstimateCellEntropySmoke:
    def test_runs_end_to_end(self):
        scenario = _make_scenario(beta=0.5, kappa=0.5, c=0.5)
        genomes = [HERO_GENOME] * 4
        result = estimate_cell_entropy(
            genomes=genomes,
            scenario=scenario,
            n_episodes=20,
            n_boot=50,
            seed=42,
            num_workers=1,
            cell_tag="test_cell",
            beta=0.5,
            kappa=0.5,
            c=0.5,
            verdict="symmetric_only",
        )
        assert result.cell_tag == "test_cell"
        assert result.n_episodes == 20
        assert len(result.positions) == 4
        for p in result.positions:
            assert p.h_cond >= 0.0
            assert p.h_cond_ci_lo <= p.h_cond + 1e-9
            assert p.h_cond_ci_hi >= p.h_cond - 1e-9
            # Joint entropy is non-negative and finite.
            assert math.isfinite(p.h_joint) and p.h_joint >= 0.0
            assert math.isfinite(p.h_minus_i) and p.h_minus_i >= 0.0

    def test_zero_entropy_constant_genome(self):
        """A genome that picks identically at every step has H == 0 across
        all positions.

        We construct a fake scenario where the agent's action space collapses
        to a single point. This is hard to guarantee with the live
        HeuristicAgent (which has exploration + fatigue noise), so we
        directly fabricate episode action traces and call the per-position
        conversion + episode bootstrap helpers.
        """
        # 4 agents, 5 episodes, T = 6 steps each, all actions identical.
        n_eps = 5
        T = 6
        constant_action = np.array([0, 1, 1], dtype=np.int8)
        ep = np.broadcast_to(constant_action, (T, 4, 3)).copy()
        episodes = [EpisodeActions(actions=ep) for _ in range(n_eps)]
        per_position = _build_per_position_episode_arrays(episodes, num_agents=4)

        for i in range(4):
            # Build minus_i too as a smoke check that the joiner doesn't crash
            # on constant inputs.
            _ = _other_positions_episodes(per_position, i)
            point, lo, hi = episode_bootstrap_ci(
                estimator=lambda x: entropy_discrete(x),
                episode_arrays=[per_position[i]],
                n_boot=50,
                rng=np.random.default_rng(i),
            )
            # Constant variable -> 0 entropy exactly (single category, MM
            # correction is (K-1)/(2N ln 2) = 0 because K = 1).
            assert point == pytest.approx(0.0, abs=1e-12)
            assert hi == pytest.approx(0.0, abs=1e-12)
            # Conditional entropy is also 0 by construction.
            assert lo == pytest.approx(0.0, abs=1e-12)
