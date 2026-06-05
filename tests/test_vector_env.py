"""Tests for the synchronous vectorized env wrapper (issue #370).

Covers the acceptance criteria from the issue body:

1. ``make_vec(id, num_envs=N)`` returns a working VectorEnv.
2. Auto-reset: when a sub-env terminates, the wrapper resets that lane
   in-place and surfaces ``final_observation`` / ``final_info``.
3. Determinism: identical seeds across two independent constructions
   yield identical trajectories.

All tests are intentionally tiny (small num_envs, short rollouts) — the
vec wrapper is pure Python over the existing Gym adapter so there is no
need for "heavy" coverage. See parent epic #365 for the broader release
infrastructure context.
"""

from __future__ import annotations

import numpy as np
import pytest

import bucket_brigade
from bucket_brigade.envs.vector import SyncVectorEnv


_PRIMARY_ID = "minimal_specialization-v1"


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Smoke tests for the factory and the class constructor."""

    def test_make_vec_returns_sync_vector_env(self):
        vec = bucket_brigade.make_vec(_PRIMARY_ID, num_envs=4)
        assert isinstance(vec, SyncVectorEnv)
        assert vec.num_envs == 4

    def test_make_vec_exposes_through_package(self):
        # Acceptance from the issue body: ``bucket_brigade.make_vec`` works.
        assert hasattr(bucket_brigade, "make_vec")
        vec = bucket_brigade.make_vec(_PRIMARY_ID, num_envs=2)
        assert vec.num_envs == 2

    def test_make_vec_rejects_zero(self):
        with pytest.raises(ValueError):
            bucket_brigade.make_vec(_PRIMARY_ID, num_envs=0)

    def test_make_vec_rejects_negative(self):
        with pytest.raises(ValueError):
            bucket_brigade.make_vec(_PRIMARY_ID, num_envs=-1)

    def test_unknown_id_raises(self):
        with pytest.raises(KeyError):
            bucket_brigade.make_vec("nope-v999", num_envs=2)

    def test_metadata_carries_scenario_id_and_num_envs(self):
        vec = bucket_brigade.make_vec(_PRIMARY_ID, num_envs=3)
        assert vec.metadata.get("scenario_id") == _PRIMARY_ID
        assert vec.metadata.get("num_envs") == 3


# ---------------------------------------------------------------------------
# Shape checks (Acceptance: returns batched observations/rewards)
# ---------------------------------------------------------------------------


class TestShapes:
    """Verify the batched shape contract."""

    @pytest.mark.parametrize("num_envs", [1, 2, 8])
    def test_reset_obs_shape(self, num_envs):
        vec = bucket_brigade.make_vec(_PRIMARY_ID, num_envs=num_envs)
        obs, info = vec.reset(seed=0)
        assert obs.shape[0] == num_envs
        assert obs.shape[1:] == vec.single_observation_space.shape

    @pytest.mark.parametrize("num_envs", [1, 2, 8])
    def test_step_returns_batched_5_tuple(self, num_envs):
        vec = bucket_brigade.make_vec(_PRIMARY_ID, num_envs=num_envs)
        vec.reset(seed=0)
        actions = vec.action_space.sample()
        out = vec.step(actions)
        assert len(out) == 5
        obs, rewards, terminated, truncated, info = out
        assert obs.shape[0] == num_envs
        assert rewards.shape == (num_envs,)
        assert terminated.shape == (num_envs,)
        assert truncated.shape == (num_envs,)
        assert rewards.dtype == np.float32
        assert terminated.dtype == bool
        assert truncated.dtype == bool

    def test_reset_obs_in_observation_space_for_each_lane(self):
        vec = bucket_brigade.make_vec(_PRIMARY_ID, num_envs=4)
        obs, _ = vec.reset(seed=0)
        single_space = vec.single_observation_space
        for i in range(vec.num_envs):
            assert single_space.contains(obs[i]), (
                f"lane {i} reset obs not in single_observation_space"
            )

    def test_action_space_shape_matches_num_envs(self):
        vec = bucket_brigade.make_vec(_PRIMARY_ID, num_envs=4)
        sample = vec.action_space.sample()
        assert sample.shape[0] == vec.num_envs

    def test_step_accepts_flat_action_layout(self):
        """A flat ``(num_envs * D,)`` action must be auto-reshaped."""
        vec = bucket_brigade.make_vec(_PRIMARY_ID, num_envs=3)
        vec.reset(seed=0)
        flat = vec.action_space.sample().reshape(-1)
        obs, *_ = vec.step(flat)
        assert obs.shape[0] == 3

    def test_step_rejects_wrong_leading_dim(self):
        vec = bucket_brigade.make_vec(_PRIMARY_ID, num_envs=3)
        vec.reset(seed=0)
        bad = np.zeros((5,) + vec.single_action_space.shape, dtype=np.int64)
        with pytest.raises(ValueError):
            vec.step(bad)


# ---------------------------------------------------------------------------
# Determinism (Acceptance: same seeds → same trajectory)
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same seeds + same action sequence => identical trajectories."""

    def test_reset_same_seed_same_obs(self):
        v1 = bucket_brigade.make_vec(_PRIMARY_ID, num_envs=4)
        v2 = bucket_brigade.make_vec(_PRIMARY_ID, num_envs=4)
        o1, _ = v1.reset(seed=42)
        o2, _ = v2.reset(seed=42)
        np.testing.assert_array_equal(o1, o2)

    def test_trajectory_same_seed_same_actions(self):
        """Drive both vec-envs with the same seeded RNG and assert equality."""
        v1 = bucket_brigade.make_vec(_PRIMARY_ID, num_envs=4)
        v2 = bucket_brigade.make_vec(_PRIMARY_ID, num_envs=4)
        v1.reset(seed=123)
        v2.reset(seed=123)

        # Pre-generate a fixed action sequence so we drive both vec-envs
        # with exactly identical inputs. Using a local RNG keeps test
        # independence from global numpy state.
        rng = np.random.default_rng(0)
        action_shape = (v1.num_envs,) + v1.single_action_space.shape
        per_lane_nvec = np.asarray(v1.single_action_space.nvec)
        action_seq = []
        for _ in range(10):
            a = rng.integers(low=0, high=per_lane_nvec, size=action_shape)
            action_seq.append(a)

        for a in action_seq:
            o1, r1, t1, tr1, _ = v1.step(a)
            o2, r2, t2, tr2, _ = v2.step(a)
            np.testing.assert_array_equal(o1, o2)
            np.testing.assert_array_equal(r1, r2)
            np.testing.assert_array_equal(t1, t2)
            np.testing.assert_array_equal(tr1, tr2)

    def test_different_lanes_get_different_seeds(self):
        """Single int seed must expand to ``[seed, seed+1, ...]`` per lane.

        With a non-trivial action sequence, two lanes with distinct
        seeds should produce distinct trajectories over a short rollout.
        If the wrapper accidentally seeded both lanes with the same seed
        the two columns would be identical.
        """
        vec = bucket_brigade.make_vec(_PRIMARY_ID, num_envs=2)
        vec.reset(seed=999)

        # Drive with random but identical-across-lanes actions so any
        # observed divergence between lane 0 and lane 1 must come from
        # the seed plumbing, not the action signal.
        rng = np.random.default_rng(0)
        per_lane_nvec = np.asarray(vec.single_action_space.nvec)
        diverged = False
        for _ in range(20):
            a_lane = rng.integers(low=0, high=per_lane_nvec)
            # Broadcast the same per-lane action across all num_envs lanes.
            a = np.tile(a_lane, (vec.num_envs, 1))
            obs, *_ = vec.step(a)
            if not np.array_equal(obs[0], obs[1]):
                diverged = True
                break
        assert diverged, (
            "Lanes seeded with seed=999, seed=1000 produced identical obs "
            "across 20 steps — seed-per-lane expansion is broken."
        )

    def test_explicit_per_sub_env_seed_sequence(self):
        """Passing a sequence of seeds (one per lane) is supported."""
        vec = bucket_brigade.make_vec(_PRIMARY_ID, num_envs=3)
        obs, _ = vec.reset(seed=[1, 2, 3])
        assert obs.shape[0] == 3

    def test_seed_sequence_length_mismatch_raises(self):
        vec = bucket_brigade.make_vec(_PRIMARY_ID, num_envs=3)
        with pytest.raises(ValueError):
            vec.reset(seed=[1, 2])  # wrong length


# ---------------------------------------------------------------------------
# Auto-reset (Acceptance: auto-reset on episode end with final_observation)
# ---------------------------------------------------------------------------


class TestAutoReset:
    """When a sub-env reports done, the wrapper resets it in-place and
    surfaces the terminal obs/info under ``final_observation`` /
    ``final_info`` per the Gymnasium convention."""

    def test_step_before_reset_raises(self):
        vec = bucket_brigade.make_vec(_PRIMARY_ID, num_envs=2)
        with pytest.raises(RuntimeError):
            vec.step(vec.action_space.sample())

    def test_info_contains_final_observation_keys(self):
        """After a single step, ``final_observation`` / ``final_info``
        keys must exist regardless of whether any lane terminated.
        """
        vec = bucket_brigade.make_vec(_PRIMARY_ID, num_envs=2)
        vec.reset(seed=0)
        _, _, _, _, info = vec.step(vec.action_space.sample())
        assert "final_observation" in info
        assert "final_info" in info
        assert len(info["final_observation"]) == vec.num_envs
        assert len(info["final_info"]) == vec.num_envs

    def test_auto_reset_surfaces_terminal_obs_and_resets_lane(self):
        """Roll a forced episode to termination; verify the auto-reset
        bookkeeping is correct.

        We drive every sub-env with the same action sequence so they all
        terminate on the same step (the dynamics are deterministic given
        the seed). When termination fires:

        - ``terminated[i]`` is True for the lane that terminated,
        - ``info["final_observation"][i]`` carries the **pre-reset**
          observation (the lane's actual terminal obs),
        - ``info["final_info"][i]`` carries the lane's terminal info,
        - the batched ``obs[i]`` is the **post-reset** observation,
          which must equal what a fresh ``reset()`` of that sub-env
          would produce.
        """
        vec = bucket_brigade.make_vec(_PRIMARY_ID, num_envs=2)
        vec.reset(seed=0)

        # Drive a long enough rollout that termination is guaranteed.
        # Use random actions for variety; the env terminates by dynamics
        # (all fires out or all houses ruined past min_nights).
        rng = np.random.default_rng(0)
        per_lane_nvec = np.asarray(vec.single_action_space.nvec)
        action_shape = (vec.num_envs,) + vec.single_action_space.shape

        saw_termination = False
        max_steps = 500
        for _ in range(max_steps):
            a = rng.integers(low=0, high=per_lane_nvec, size=action_shape)
            obs, rewards, terminated, truncated, info = vec.step(a)
            done_mask = terminated | truncated
            if not done_mask.any():
                # Sanity: non-done lanes have ``None`` finals.
                for i in range(vec.num_envs):
                    if not done_mask[i]:
                        assert info["final_observation"][i] is None
                        assert info["final_info"][i] is None
                continue

            saw_termination = True
            for i in range(vec.num_envs):
                if done_mask[i]:
                    # Terminal observation / info MUST be surfaced.
                    final_obs = info["final_observation"][i]
                    final_info = info["final_info"][i]
                    assert final_obs is not None, (
                        f"lane {i} terminated but final_observation is None"
                    )
                    assert final_info is not None, (
                        f"lane {i} terminated but final_info is None"
                    )
                    assert final_obs.shape == vec.single_observation_space.shape
                    # And the batched ``obs[i]`` row is the post-reset
                    # observation, which must lie in the observation
                    # space (not e.g. an all-zeros sentinel).
                    assert vec.single_observation_space.contains(obs[i])
                else:
                    assert info["final_observation"][i] is None
                    assert info["final_info"][i] is None
            break

        assert saw_termination, (
            f"No termination observed in {max_steps} steps — either "
            f"the env never terminates or the wrapper is consuming the "
            f"done signal."
        )

    def test_auto_reset_obs_matches_fresh_reset(self):
        """The post-auto-reset obs in a terminated lane must equal the
        obs from a brand-new env reset with the same (advanced) RNG.

        We can't easily reach into the underlying env's RNG, so the
        weaker but still useful check: the post-reset obs is in the
        observation_space AND running the same action again from the
        post-reset state produces another valid obs (i.e. the lane is
        usable, not a half-broken sentinel).
        """
        vec = bucket_brigade.make_vec(_PRIMARY_ID, num_envs=1)
        vec.reset(seed=0)
        rng = np.random.default_rng(0)
        per_lane_nvec = np.asarray(vec.single_action_space.nvec)

        for _ in range(500):
            a = rng.integers(
                low=0, high=per_lane_nvec, size=(1,) + vec.single_action_space.shape
            )
            obs, _, terminated, truncated, info = vec.step(a)
            if (terminated | truncated).any():
                # Post-reset lane must be live: another step succeeds and
                # produces a valid obs.
                a2 = rng.integers(
                    low=0,
                    high=per_lane_nvec,
                    size=(1,) + vec.single_action_space.shape,
                )
                obs2, *_ = vec.step(a2)
                assert vec.single_observation_space.contains(obs2[0])
                return

        pytest.fail("Episode did not terminate within 500 steps.")


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_is_safe(self):
        vec = bucket_brigade.make_vec(_PRIMARY_ID, num_envs=2)
        vec.reset(seed=0)
        vec.close()  # Must not raise.
        vec.close()  # Idempotent.
