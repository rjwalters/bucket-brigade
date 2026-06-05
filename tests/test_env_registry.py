"""Tests for the versioned scenario registry + Gymnasium adapter (issue #369)."""

from __future__ import annotations

import numpy as np
import pytest

import bucket_brigade
from bucket_brigade.envs import gym_adapter
from bucket_brigade.envs.registry import (
    DEFAULT_NUM_AGENTS,
    SCENARIO_VERSIONS,
    get_scenario_by_id,
    list_versioned_scenarios,
    parse_scenario_id,
)


# Sample ID we use for compliance tests — minimal_specialization is the
# canonical P3 diagnostic scenario and has unambiguous semantics.
_PRIMARY_ID = "minimal_specialization-v1"


class TestRegistryShape:
    """Static shape checks on the frozen registry itself."""

    def test_registry_nonempty(self):
        assert len(SCENARIO_VERSIONS) >= 2, (
            "Issue #369 acceptance criterion: at least 2 versioned scenario "
            "IDs must be registered."
        )

    def test_every_id_parses(self):
        """Every registered key must conform to the '<name>-v<int>' shape."""
        for scenario_id in SCENARIO_VERSIONS:
            base, version = parse_scenario_id(scenario_id)
            assert base, f"Empty base name in {scenario_id!r}"
            assert version >= 1, f"Version must be >= 1 in {scenario_id!r}"

    def test_required_ids_present(self):
        """The two acceptance-criterion IDs from the issue body MUST be registered."""
        ids = list_versioned_scenarios()
        assert "minimal_specialization-v1" in ids
        assert "rest_trap-v1" in ids

    def test_list_envs_matches_registry_keys(self):
        """``bucket_brigade.list_envs()`` must agree with the registry keys."""
        assert bucket_brigade.list_envs() == sorted(SCENARIO_VERSIONS.keys())


class TestParseScenarioId:
    def test_round_trip(self):
        assert parse_scenario_id("minimal_specialization-v1") == (
            "minimal_specialization",
            1,
        )

    def test_multi_dash_base(self):
        # rpartition ensures only the last '-v<int>' is the version split.
        # We don't currently use multi-dash names but the parser must be safe.
        base, v = parse_scenario_id("my-fancy-name-v3")
        assert base == "my-fancy-name"
        assert v == 3

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            parse_scenario_id("no_version_here")
        with pytest.raises(ValueError):
            parse_scenario_id("name-vNaN")
        with pytest.raises(ValueError):
            # Missing base.
            parse_scenario_id("-v1")


class TestScenarioLookup:
    def test_unknown_id_raises(self):
        with pytest.raises(KeyError) as exc:
            get_scenario_by_id("nope-v999")
        # Error message must include the available ID list so users can
        # self-serve recovery.
        assert "Available IDs" in str(exc.value)

    def test_default_num_agents(self):
        sc = get_scenario_by_id(_PRIMARY_ID)
        assert sc.num_agents == DEFAULT_NUM_AGENTS

    def test_num_agents_override(self):
        # Use ``default-v1`` because it has scalar ownership rewards
        # (auto-promoted to length-num_agents at construction). Some
        # scenarios — notably ``minimal_specialization-v1`` — hard-code
        # 4-element ownership vectors so they are 4-agent-only by design,
        # and overriding num_agents would raise a length-validation
        # error from ``Scenario.__post_init__``.
        sc = get_scenario_by_id("default-v1", num_agents=3)
        assert sc.num_agents == 3


class TestMakeRoundTrip:
    """Round-trip ``make(id)`` for every registered ID."""

    @pytest.mark.parametrize("scenario_id", sorted(SCENARIO_VERSIONS.keys()))
    def test_make_returns_env(self, scenario_id):
        """Acceptance criterion: ``make(id)`` works for every registered ID."""
        env = bucket_brigade.make(scenario_id)
        assert isinstance(env, gym_adapter.BucketBrigadeGymEnv)
        assert env.metadata.get("scenario_id") == scenario_id

    @pytest.mark.parametrize("scenario_id", sorted(SCENARIO_VERSIONS.keys()))
    def test_reset_returns_compliant_obs(self, scenario_id):
        """Gymnasium reset() returns (obs, info) and obs is in observation_space."""
        env = bucket_brigade.make(scenario_id)
        obs, info = env.reset(seed=0)
        assert isinstance(info, dict)
        # Issue acceptance: scenario_id is surfaced in info for traceability.
        assert info.get("scenario_id") == scenario_id
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == env.observation_space.dtype
        assert env.observation_space.contains(obs), (
            f"reset() obs not in declared observation_space for {scenario_id!r}"
        )

    @pytest.mark.parametrize("scenario_id", sorted(SCENARIO_VERSIONS.keys()))
    def test_step_returns_5_tuple(self, scenario_id):
        """Gymnasium step() returns (obs, reward, terminated, truncated, info)."""
        env = bucket_brigade.make(scenario_id)
        env.reset(seed=0)
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5, (
            f"step() must return 5-tuple for Gymnasium compliance; got {len(result)}"
        )
        obs, reward, terminated, truncated, info = result
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == env.observation_space.dtype
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        # Truncated is always False — episode length is dynamics-driven.
        assert truncated is False
        assert isinstance(info, dict)
        assert "per_agent_rewards" in info


class TestSpaceStability:
    """Spaces must be stable across ``make()`` calls for the same ID."""

    def test_action_space_stable(self):
        e1 = bucket_brigade.make(_PRIMARY_ID)
        e2 = bucket_brigade.make(_PRIMARY_ID)
        assert e1.action_space.shape == e2.action_space.shape
        assert np.array_equal(e1.action_space.nvec, e2.action_space.nvec)

    def test_observation_space_stable(self):
        e1 = bucket_brigade.make(_PRIMARY_ID)
        e2 = bucket_brigade.make(_PRIMARY_ID)
        assert e1.observation_space.shape == e2.observation_space.shape
        assert e1.observation_space.dtype == e2.observation_space.dtype

    def test_action_space_layout(self):
        """Per-agent layout must be [num_houses, 2, 2] repeated num_agents times."""
        env = bucket_brigade.make(_PRIMARY_ID)
        expected = np.array([env.num_houses, 2, 2] * env.num_agents, dtype=np.int64)
        assert np.array_equal(env.action_space.nvec, expected)


class TestSeedDeterminism:
    """Reproducibility: identical seeds must yield identical initial obs."""

    def test_same_seed_same_initial_obs(self):
        e1 = bucket_brigade.make(_PRIMARY_ID)
        e2 = bucket_brigade.make(_PRIMARY_ID)
        obs1, _ = e1.reset(seed=42)
        obs2, _ = e2.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

    def test_different_seeds_diverge_after_steps(self):
        """Sanity check: distinct seeds eventually produce distinct obs.

        Initial reset obs may legitimately be deterministic (e.g.
        all houses safe, all agents at location 0) — the seed only
        affects the stochastic dynamics that fire from ``step()``.
        Roll a short trajectory with a fixed action sequence and assert
        the two seeds diverge somewhere in the rollout. If they don't,
        the seed is being ignored.
        """
        e1 = bucket_brigade.make(_PRIMARY_ID)
        e2 = bucket_brigade.make(_PRIMARY_ID)
        e1.reset(seed=0)
        e2.reset(seed=12345)
        # Fixed action: all-rest at house 0.
        action = np.zeros(e1.action_space.shape, dtype=np.int64)
        diverged = False
        for _ in range(20):
            o1, *_ = e1.step(action)
            o2, *_ = e2.step(action)
            if not np.array_equal(o1, o2):
                diverged = True
                break
        assert diverged, (
            "Two distinct seeds produced identical obs across 20 steps "
            "with the same action sequence — seed plumbing is likely broken."
        )


class TestGymnasiumCompliance:
    """Smoke test that the env passes the basic Gymnasium contract."""

    def test_step_before_reset_raises(self):
        env = bucket_brigade.make(_PRIMARY_ID)
        with pytest.raises(RuntimeError):
            env.step(env.action_space.sample())

    def test_full_episode_roll(self):
        """Roll an episode to termination with random actions.

        This is the smoke test the issue body calls out:
        ``python -c "import bucket_brigade; e = bucket_brigade.make('minimal_specialization-v1'); e.reset()"``
        plus a few steps to ensure step() also works.
        """
        env = bucket_brigade.make(_PRIMARY_ID)
        env.reset(seed=7)
        max_steps = 200  # Bounded so a runaway loop fails CI loudly.
        for _ in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == env.observation_space.shape
            if terminated or truncated:
                break
        else:
            pytest.fail(
                f"Episode did not terminate within {max_steps} steps; "
                f"possible env loop bug."
            )

    def test_close_is_safe(self):
        env = bucket_brigade.make(_PRIMARY_ID)
        env.reset(seed=0)
        env.close()  # Must not raise.


class TestDocstringPolicy:
    """The version-bump policy MUST be documented (issue #369 acceptance)."""

    def test_registry_module_documents_policy(self):
        from bucket_brigade.envs import registry as reg_mod

        assert reg_mod.__doc__ is not None
        doc = reg_mod.__doc__
        # The policy keywords must appear so reviewers can grep for them.
        assert "Version-bump policy" in doc
        assert "observation space" in doc
        assert "action space" in doc
        assert "reward function" in doc
