"""Unit tests for the ``bucket_brigade.baselines`` package (issue #199).

The specialist policy must:

* Return ``MultiDiscrete([10, 2])``-compatible actions (shape (2,), int64,
  values in range).
* Choose WORK on a burning owned house, REST otherwise.
* Be reusable across scenarios (no scenario-specific assumption).

These tests are pure-Python and very fast (<1s) so they live in the normal
``tests/`` directory.
"""

from __future__ import annotations

import numpy as np
import pytest

from bucket_brigade.baselines import (
    SpecialistPolicy,
    specialist_action,
    specialist_action_joint,
)
from bucket_brigade.baselines.specialist import _owned_houses
from bucket_brigade.envs.bucket_brigade_env import BucketBrigadeEnv
from bucket_brigade.envs.scenarios_generated import get_scenario_by_name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_obs(houses: np.ndarray) -> dict:
    """Build the minimal observation dict the specialist consumes."""
    return {"houses": houses}


# ---------------------------------------------------------------------------
# Ownership round-robin
# ---------------------------------------------------------------------------


def test_owned_houses_round_robin_4_agents() -> None:
    """Ownership matches the canonical round-robin np.arange(10) % num_agents.

    For num_agents=4: agent 0 -> {0,4,8}, agent 1 -> {1,5,9},
                       agent 2 -> {2,6},   agent 3 -> {3,7}.
    """
    assert _owned_houses(0, 4, 10).tolist() == [0, 4, 8]
    assert _owned_houses(1, 4, 10).tolist() == [1, 5, 9]
    assert _owned_houses(2, 4, 10).tolist() == [2, 6]
    assert _owned_houses(3, 4, 10).tolist() == [3, 7]


# ---------------------------------------------------------------------------
# Specialist policy: action shape and validity
# ---------------------------------------------------------------------------


def test_specialist_action_returns_valid_multidiscrete() -> None:
    """Action is shape (2,), int64, with values in ``MultiDiscrete([10, 2])``."""
    obs = _make_obs(np.zeros(10, dtype=np.int8))  # all SAFE
    act = specialist_action(obs, agent_id=0, num_agents=4)

    assert isinstance(act, np.ndarray)
    assert act.shape == (2,)
    assert act.dtype == np.int64
    assert 0 <= int(act[0]) < 10
    assert int(act[1]) in (0, 1)


def test_specialist_action_joint_shape_for_4_agents() -> None:
    """Joint specialist returns a (num_agents, 2) int64 array."""
    obs = _make_obs(np.zeros(10, dtype=np.int8))
    joint = specialist_action_joint(obs, num_agents=4)
    assert joint.shape == (4, 2)
    assert joint.dtype == np.int64
    assert np.all((joint[:, 0] >= 0) & (joint[:, 0] < 10))
    assert np.all((joint[:, 1] >= 0) & (joint[:, 1] < 2))


# ---------------------------------------------------------------------------
# Specialist policy: behavior
# ---------------------------------------------------------------------------


def test_specialist_works_lowest_index_burning_owned_house() -> None:
    """When agent 0 has burning owned houses {4, 8}, it works house 4."""
    houses = np.zeros(10, dtype=np.int8)
    houses[4] = BucketBrigadeEnv.BURNING
    houses[8] = BucketBrigadeEnv.BURNING
    obs = _make_obs(houses)

    act = specialist_action(obs, agent_id=0, num_agents=4)
    assert int(act[0]) == 4
    assert int(act[1]) == BucketBrigadeEnv.WORK


def test_specialist_rests_when_no_owned_house_burning() -> None:
    """When no owned house is BURNING, agent rests (mode=REST)."""
    # House 1 (owned by agent 1) is burning; agent 0 owns {0,4,8}, none burning.
    houses = np.zeros(10, dtype=np.int8)
    houses[1] = BucketBrigadeEnv.BURNING
    obs = _make_obs(houses)

    act = specialist_action(obs, agent_id=0, num_agents=4)
    assert int(act[1]) == BucketBrigadeEnv.REST


def test_specialist_ignores_others_burning_houses() -> None:
    """A burning OTHER house should not pull agent off its own (resting) state."""
    # Agent 0 owns {0,4,8}; only house 5 (owned by agent 1) is burning.
    houses = np.zeros(10, dtype=np.int8)
    houses[5] = BucketBrigadeEnv.BURNING
    obs = _make_obs(houses)

    act_agent_0 = specialist_action(obs, agent_id=0, num_agents=4)
    act_agent_1 = specialist_action(obs, agent_id=1, num_agents=4)

    # Agent 0 has nothing to do.
    assert int(act_agent_0[1]) == BucketBrigadeEnv.REST
    # Agent 1 fights its own burning house.
    assert int(act_agent_1[0]) == 5
    assert int(act_agent_1[1]) == BucketBrigadeEnv.WORK


def test_specialist_ignores_ruined_owned_houses() -> None:
    """A RUINED owned house is not BURNING; the specialist should not work it."""
    houses = np.zeros(10, dtype=np.int8)
    houses[0] = BucketBrigadeEnv.RUINED  # owned by agent 0, but already ruined
    obs = _make_obs(houses)

    act = specialist_action(obs, agent_id=0, num_agents=4)
    assert int(act[1]) == BucketBrigadeEnv.REST


def test_specialist_policy_wrapper_matches_joint_function() -> None:
    """``SpecialistPolicy(...)`` is equivalent to ``specialist_action_joint``."""
    houses = np.zeros(10, dtype=np.int8)
    houses[4] = BucketBrigadeEnv.BURNING
    obs = _make_obs(houses)

    policy = SpecialistPolicy(num_agents=4)
    assert np.array_equal(
        policy(obs),
        specialist_action_joint(obs, num_agents=4),
    )


def test_specialist_rejects_invalid_agent_id() -> None:
    obs = _make_obs(np.zeros(10, dtype=np.int8))
    with pytest.raises(ValueError, match="agent_id"):
        specialist_action(obs, agent_id=4, num_agents=4)
    with pytest.raises(ValueError, match="agent_id"):
        specialist_action(obs, agent_id=-1, num_agents=4)


# ---------------------------------------------------------------------------
# minimal_specialization scenario integration
# ---------------------------------------------------------------------------


def test_minimal_specialization_scenario_loads_with_expected_vectors() -> None:
    """The new #199 scenario constructs and has the locked-in reward vectors."""
    s = get_scenario_by_name("minimal_specialization", num_agents=4)
    assert s.team_reward_house_survives == 10.0
    assert s.team_penalty_house_burns == 10.0
    assert s.reward_own_house_survives == [50.0, 50.0, 50.0, 50.0]
    assert s.reward_other_house_survives == [0.0, 0.0, 0.0, 0.0]
    assert s.penalty_own_house_burns == [100.0, 100.0, 100.0, 100.0]
    assert s.penalty_other_house_burns == [0.0, 0.0, 0.0, 0.0]
    assert s.cost_to_work_one_night == 0.5
    assert s.min_nights == 12


def test_minimal_specialization_env_plays_to_completion_with_specialist() -> None:
    """End-to-end smoke: env constructs, specialist drives, episode terminates."""
    s = get_scenario_by_name("minimal_specialization", num_agents=4)
    env = BucketBrigadeEnv(scenario=s)
    obs = env.reset(seed=0)
    steps = 0
    while not env.done and steps < 500:  # 500 step safety cap
        actions = specialist_action_joint(obs, num_agents=4)
        assert actions.shape == (4, 2)
        obs, _, _, _ = env.step(actions)
        steps += 1
    assert env.done, "Episode did not terminate within 500 steps"
    assert steps >= s.min_nights
