"""Regression tests pinning beta-inertness in bernoulli extinguish mode (issue #458).

Mechanism (see ``bucket-brigade-core/src/engine/core.rs`` step order and
``engine/phases.rs::spread_fires``): in the default ``"bernoulli"``
extinguish mode the burn-out phase runs before the spread phase and ruins
every still-BURNING house, so ``spread_fires`` never sees a BURNING source.
``prob_fire_spreads_to_neighbor`` (beta) therefore never gates a spread and
draws zero RNG — two rollouts differing only in beta are bit-identical
under a shared seed.

These tests pin three facts:

1. **Bernoulli inertness (Rust core)**: same seed, different beta =>
   bit-identical house trajectories and rewards.
2. **Bernoulli inertness (Python env)**: the pure-Python mirror engine has
   the same phase order and the same inertness.
3. **Continuous-mode liveness (Rust core)**: in ``extinguish_mode=
   "continuous"`` (#253) fires persist into the spread phase, so beta DOES
   change the dynamics — guarding against a future refactor that makes
   beta inert everywhere.

Plus the reason beta must not be deleted as "dead code": it is a live
observation feature (``scenario_info[0]``), so different beta values
perturb trained-policy network inputs even in bernoulli mode.

Artifact-level pins of the same fact (bit-identical cross-beta phase-diagram
columns) live in ``tests/test_beta_residuals.py`` / PR #450; this file is
the engine-level pin.
"""

from __future__ import annotations

import numpy as np
import pytest

from bucket_brigade.envs import BucketBrigadeEnv
from bucket_brigade.envs.scenarios_generated import Scenario

pytest.importorskip("bucket_brigade_core")
import bucket_brigade_core  # noqa: E402

SEED = 20260701
NUM_AGENTS = 4
NUM_STEPS = 20


def _rust_scenario(beta: float, **overrides) -> "bucket_brigade_core.Scenario":
    """Rust core scenario with lively fire dynamics and configurable beta."""
    kwargs = dict(
        prob_fire_spreads_to_neighbor=beta,
        prob_solo_agent_extinguishes_fire=0.5,
        prob_house_catches_fire=0.3,
        team_reward_house_survives=10.0,
        team_penalty_house_burns=10.0,
        cost_to_work_one_night=0.5,
        min_nights=12,
        reward_own_house_survives=1.0,
        reward_other_house_survives=0.0,
        penalty_own_house_burns=2.0,
        penalty_other_house_burns=0.0,
    )
    kwargs.update(overrides)
    return bucket_brigade_core.Scenario(**kwargs)


def _python_scenario(beta: float) -> Scenario:
    return Scenario(
        prob_fire_spreads_to_neighbor=beta,
        prob_solo_agent_extinguishes_fire=0.5,
        prob_house_catches_fire=0.3,
        team_reward_house_survives=10.0,
        team_penalty_house_burns=10.0,
        reward_own_house_survives=1.0,
        reward_other_house_survives=0.0,
        penalty_own_house_burns=2.0,
        penalty_other_house_burns=0.0,
        cost_to_work_one_night=0.5,
        min_nights=12,
        num_agents=NUM_AGENTS,
    )


def _fixed_actions(step: int) -> list[list[int]]:
    """Deterministic mixed work/rest action schedule (same for both envs)."""
    return [[(step + i) % 10, (step + i) % 2, i % 2] for i in range(NUM_AGENTS)]


def _rollout_rust(scenario) -> tuple[list[list[int]], list[list[float]]]:
    env = bucket_brigade_core.BucketBrigade(scenario, NUM_AGENTS, SEED)
    houses_trace: list[list[int]] = []
    rewards_trace: list[list[float]] = []
    for step in range(NUM_STEPS):
        rewards, done, _info = env.step(_fixed_actions(step))
        houses_trace.append(list(env.get_current_state().houses))
        rewards_trace.append(list(rewards))
        if done:
            break
    return houses_trace, rewards_trace


def _rollout_python(scenario: Scenario) -> tuple[list[list[int]], list[list[float]]]:
    env = BucketBrigadeEnv(scenario=scenario)
    env.reset(seed=SEED)
    houses_trace: list[list[int]] = []
    rewards_trace: list[list[float]] = []
    for step in range(NUM_STEPS):
        if env.done:
            break
        actions = np.array(_fixed_actions(step), dtype=np.int8)
        _obs, rewards, dones, _info = env.step(actions)
        houses_trace.append(env.houses.tolist())
        rewards_trace.append(rewards.tolist())
        if bool(np.all(dones)):
            break
    return houses_trace, rewards_trace


class TestBernoulliBetaInertness:
    """Same seed + different beta => bit-identical rollouts (bernoulli mode)."""

    def test_rust_core_trajectories_bit_identical_across_beta(self):
        houses_lo, rewards_lo = _rollout_rust(_rust_scenario(beta=0.05))
        houses_hi, rewards_hi = _rollout_rust(_rust_scenario(beta=0.95))

        # The rollout must exercise fire dynamics for the pin to mean
        # anything: at least one house must burn (become RUINED) somewhere.
        assert any(2 in houses for houses in houses_lo), (
            "Rollout never produced a ruined house; the inertness pin is vacuous"
        )
        assert houses_lo == houses_hi, (
            "beta changed bernoulli-mode house dynamics — the #458 phase-order "
            "inertness no longer holds (burn-out no longer shields the spread "
            "phase). This is a registry-version-level behavior change: every "
            "committed baseline/NE artifact would be invalidated. See issue #458."
        )
        assert rewards_lo == rewards_hi, (
            "beta changed bernoulli-mode rewards despite identical house "
            "trajectories — see issue #458"
        )

    def test_python_env_trajectories_bit_identical_across_beta(self):
        houses_lo, rewards_lo = _rollout_python(_python_scenario(beta=0.05))
        houses_hi, rewards_hi = _rollout_python(_python_scenario(beta=0.95))

        assert any(2 in houses for houses in houses_lo), (
            "Rollout never produced a ruined house; the inertness pin is vacuous"
        )
        assert houses_lo == houses_hi, (
            "beta changed the Python env's bernoulli-mode house dynamics — "
            "phase-order inertness (issue #458) no longer holds"
        )
        assert rewards_lo == rewards_hi


class TestContinuousModeBetaLiveness:
    """Counterexample: beta IS live in continuous extinguish mode (#253).

    With ``extinguish_mode="continuous"`` burn-out returns early, fires
    persist into the spread phase, and beta gates real spread events. All
    agents rest so no fire is ever extinguished; with beta=1.0 every fire
    ignites its safe neighbors while with beta=0.0 none do, so house
    trajectories must diverge once any fire ignites.
    """

    def test_rust_core_beta_changes_continuous_mode_dynamics(self):
        def rollout(beta: float) -> list[list[int]]:
            scenario = _rust_scenario(
                beta=beta,
                extinguish_mode="continuous",
                suppression_per_worker=0.5,
            )
            env = bucket_brigade_core.BucketBrigade(scenario, NUM_AGENTS, SEED)
            all_rest = [[0, 0, 0] for _ in range(NUM_AGENTS)]
            trace = []
            for _ in range(NUM_STEPS):
                _rewards, done, _info = env.step(all_rest)
                trace.append(list(env.get_current_state().houses))
                if done:
                    break
            return trace

        houses_none = rollout(beta=0.0)
        houses_full = rollout(beta=1.0)
        assert houses_none != houses_full, (
            "beta had no effect even in continuous extinguish mode — fire "
            "spread appears to be dead everywhere, not just in bernoulli "
            "mode (issue #458 documents bernoulli-only inertness)"
        )


class TestBetaIsALiveObservationFeature:
    """Beta must not be removed as dead code: agents observe it directly."""

    def test_scenario_info_slot0_carries_beta(self):
        for beta in (0.05, 0.95):
            env = bucket_brigade_core.BucketBrigade(
                _rust_scenario(beta=beta), NUM_AGENTS, SEED
            )
            obs = env.get_observation(0)
            assert obs.scenario_info[0] == pytest.approx(beta), (
                "scenario_info[0] no longer carries "
                "prob_fire_spreads_to_neighbor — trained-policy observation "
                "layouts depend on this slot (issue #458)"
            )

    def test_python_env_observation_carries_beta(self):
        env = BucketBrigadeEnv(scenario=_python_scenario(beta=0.7))
        obs = env.reset(seed=SEED)
        assert obs["scenario_info"][0] == pytest.approx(0.7)
