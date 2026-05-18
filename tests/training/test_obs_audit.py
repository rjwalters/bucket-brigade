"""Tests for the live observation auditor (issue #274).

Coverage:

- Round-trip: ``ObsAuditSample`` -> JSONL -> ``ObsAuditSample``.
- Bit-identity: training-with-auditor produces identical rollout metrics
  to training-without-auditor when the same seed is fixed.
- Identity-tail extraction: the auditor's ``identity_tail`` slice is the
  trailing ``num_agents`` floats of ``flat_obs`` and equals the one-hot
  for ``agent_id``.
- Action sanitization parity: when ``action_validity_mode == "adjacent_only"``
  the auditor captures both ``raw_action`` and ``action_taken``; an
  out-of-reach raw target is rewritten by the env's ``_sanitize_actions``,
  and the auditor's two action slots reflect the divergence.
- Edge cases: ``N=0`` is a no-op (no file, no overhead); ``N >= total_steps``
  records every step.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from bucket_brigade.envs.bucket_brigade_env import BucketBrigadeEnv
from bucket_brigade.envs.scenarios_generated import get_scenario_by_name
from bucket_brigade.training.joint_trainer import (
    JointPPOTrainer,
    flatten_dict_obs,
)
from bucket_brigade.training.obs_audit import ObsAuditor, ObsAuditSample


NUM_AGENTS = 4
ROLLOUT_STEPS = 32
ACTION_DIMS = [10, 2, 2]


def _env_fn():
    return BucketBrigadeEnv(num_agents=NUM_AGENTS)


def _adjacent_env_fn():
    """Env-fn variant with the post-#316 adjacent-only action validity mode."""
    scenario = get_scenario_by_name("default", num_agents=NUM_AGENTS)
    scenario.action_validity_mode = "adjacent_only"
    return BucketBrigadeEnv(scenario=scenario)


def _make_trainer(obs_auditor=None, seed=0, env_fn_to_use=_env_fn) -> JointPPOTrainer:
    env = env_fn_to_use()
    obs = env.reset(seed=seed)
    obs_dim = flatten_dict_obs(obs, agent_id=0, num_agents=NUM_AGENTS).shape[0]
    return JointPPOTrainer(
        env_fn=env_fn_to_use,
        num_agents=NUM_AGENTS,
        obs_dim=obs_dim,
        action_dims=ACTION_DIMS,
        hidden_size=16,
        minibatch_size=32,
        ppo_epochs=2,
        seed=seed,
        obs_auditor=obs_auditor,
    )


# ----------------------------------------------------------------------
# 1. ObsAuditSample serialization round-trip
# ----------------------------------------------------------------------


class TestSerializationRoundTrip:
    def test_to_from_json_dict_preserves_arrays(self):
        flat_obs = np.array([0.5, -1.0, 2.0, 0.0], dtype=np.float32)
        identity_tail = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        sample = ObsAuditSample(
            t=7,
            agent_id=1,
            flat_obs=flat_obs,
            env_state={
                "houses": np.array([0, 1, 2, 0]),
                "signals": np.array([1, 0]),
            },
            raw_action=np.array([3, 1, 1], dtype=np.int64),
            action_taken=np.array([3, 1], dtype=np.int64),
            reward=0.25,
            identity_tail=identity_tail,
            scenario_info={"action_validity_mode": "always_valid"},
        )
        d = sample.to_json_dict()
        # Must be JSON-serializable (no numpy scalars).
        s = json.dumps(d)
        d2 = json.loads(s)
        restored = ObsAuditSample.from_json_dict(d2)
        np.testing.assert_array_equal(restored.flat_obs, flat_obs)
        np.testing.assert_array_equal(
            restored.raw_action, np.array([3, 1, 1], dtype=np.int64)
        )
        np.testing.assert_array_equal(
            restored.action_taken, np.array([3, 1], dtype=np.int64)
        )
        np.testing.assert_array_equal(restored.identity_tail, identity_tail)
        assert restored.t == 7
        assert restored.agent_id == 1
        assert restored.reward == pytest.approx(0.25)
        assert restored.scenario_info["action_validity_mode"] == "always_valid"

    def test_dump_and_load_jsonl(self, tmp_path: Path):
        auditor = ObsAuditor(num_samples=5, total_steps=20, seed=0)
        # Manually inject samples (simulate what the wire-in would do).
        for t in auditor.picked_steps():
            for i in range(NUM_AGENTS):
                auditor._global_step = t  # noqa: SLF001 — direct manipulation OK in test
                auditor.maybe_record(
                    agent_id=i,
                    flat_obs=np.full(8, float(t), dtype=np.float32),
                    env_state={"houses": np.array([0, 1, 2])},
                    raw_action=np.array([i, 0, 1], dtype=np.int64),
                    action_taken=np.array([i, 0], dtype=np.int64),
                    reward=float(t * 0.1),
                    num_agents=NUM_AGENTS,
                )
        out = tmp_path / "audit.jsonl"
        auditor.dump(out)
        assert out.exists()
        loaded = ObsAuditor.load(out)
        assert len(loaded) == len(auditor.samples)
        for src, dst in zip(auditor.samples, loaded):
            assert src.t == dst.t
            assert src.agent_id == dst.agent_id
            np.testing.assert_array_equal(src.flat_obs, dst.flat_obs)


# ----------------------------------------------------------------------
# 2. Bit-identity: audit is observe-only
# ----------------------------------------------------------------------


class TestBitIdentity:
    """The audit-disabled path must be bit-for-bit equivalent to the
    audit-enabled path on metrics that PPO actually consumes."""

    def test_rollout_metrics_unchanged_by_auditor(self):
        # Two identically-seeded trainers, one with an auditor wired in.
        torch.manual_seed(0)
        np.random.seed(0)
        t1 = _make_trainer(obs_auditor=None, seed=42)
        roll_a = t1.collect_rollout(ROLLOUT_STEPS)

        torch.manual_seed(0)
        np.random.seed(0)
        auditor = ObsAuditor(num_samples=10, total_steps=ROLLOUT_STEPS, seed=0)
        t2 = _make_trainer(obs_auditor=auditor, seed=42)
        roll_b = t2.collect_rollout(ROLLOUT_STEPS)

        # All quantities that the trainer consumes must match bit-for-bit.
        np.testing.assert_array_equal(
            roll_a.observations.cpu().numpy(),
            roll_b.observations.cpu().numpy(),
        )
        for i in range(NUM_AGENTS):
            np.testing.assert_array_equal(
                roll_a.actions[i].cpu().numpy(),
                roll_b.actions[i].cpu().numpy(),
            )
            np.testing.assert_array_equal(
                roll_a.rewards[i].cpu().numpy(),
                roll_b.rewards[i].cpu().numpy(),
            )
            np.testing.assert_allclose(
                roll_a.log_probs[i].cpu().numpy(),
                roll_b.log_probs[i].cpu().numpy(),
                rtol=0.0,
                atol=0.0,
            )

        # But the auditor should have recorded samples.
        assert len(auditor.samples) > 0
        assert len(auditor.samples) == len(auditor.picked_steps()) * NUM_AGENTS

    def test_disabled_auditor_is_no_op(self, tmp_path: Path):
        auditor = ObsAuditor(num_samples=0, total_steps=100, seed=0)
        assert auditor.enabled is False
        assert auditor.picked_steps() == ()
        # maybe_record must not append even when called.
        auditor.maybe_record(
            agent_id=0,
            flat_obs=np.zeros(8, dtype=np.float32),
            env_state={"houses": np.array([0])},
            raw_action=np.array([0, 0, 0], dtype=np.int64),
            action_taken=np.array([0, 0], dtype=np.int64),
            reward=0.0,
            num_agents=4,
        )
        assert auditor.samples == []
        # dump() must NOT create a file when audit is disabled.
        out = tmp_path / "should_not_exist.jsonl"
        auditor.dump(out)
        assert not out.exists()


# ----------------------------------------------------------------------
# 3. Identity-tail extraction
# ----------------------------------------------------------------------


class TestIdentityTailExtraction:
    def test_handcrafted_flat_obs(self):
        """Build a flat_obs by hand and verify the trailing N floats are
        the agent's one-hot."""
        obs_dim = 12
        num_agents = 4
        for agent_id in range(num_agents):
            flat = np.zeros(obs_dim, dtype=np.float32)
            # Populate the non-identity prefix with arbitrary values.
            flat[: obs_dim - num_agents] = np.arange(
                obs_dim - num_agents, dtype=np.float32
            )
            # Set the one-hot identity tail.
            flat[obs_dim - num_agents + agent_id] = 1.0

            auditor = ObsAuditor(num_samples=1, total_steps=1, seed=0)
            auditor.maybe_record(
                agent_id=agent_id,
                flat_obs=flat,
                env_state={"houses": np.array([0])},
                raw_action=np.array([0, 0, 0], dtype=np.int64),
                action_taken=np.array([0, 0], dtype=np.int64),
                reward=0.0,
                num_agents=num_agents,
            )
            assert len(auditor.samples) == 1
            sample = auditor.samples[0]
            # The identity_tail must be the last N elements of flat_obs.
            np.testing.assert_array_equal(sample.identity_tail, flat[-num_agents:])
            # And the argmax must equal agent_id.
            assert int(np.argmax(sample.identity_tail)) == agent_id
            # And the slice must be a valid one-hot.
            assert sample.identity_tail.sum() == pytest.approx(1.0)
            assert (sample.identity_tail >= 0).all()

    def test_extracted_from_real_rollout(self):
        """End-to-end: every audit sample's identity_tail matches its agent_id."""
        auditor = ObsAuditor(
            num_samples=ROLLOUT_STEPS, total_steps=ROLLOUT_STEPS, seed=0
        )
        trainer = _make_trainer(obs_auditor=auditor, seed=42)
        trainer.collect_rollout(ROLLOUT_STEPS)
        assert len(auditor.samples) == ROLLOUT_STEPS * NUM_AGENTS
        for sample in auditor.samples:
            assert sample.identity_tail.shape == (NUM_AGENTS,)
            # The agent's one-hot must be correct.
            assert int(np.argmax(sample.identity_tail)) == sample.agent_id
            assert sample.identity_tail[sample.agent_id] == pytest.approx(1.0)
            # The rest must be zero.
            other = np.delete(sample.identity_tail, sample.agent_id)
            assert (other == 0).all()


# ----------------------------------------------------------------------
# 4. Action sanitization parity (PR #316)
# ----------------------------------------------------------------------


class TestActionSanitizationParity:
    def test_always_valid_mode_raw_equals_taken_house(self):
        """With ``action_validity_mode == "always_valid"`` (the default), the
        sanitized house index equals the raw house index."""
        auditor = ObsAuditor(
            num_samples=ROLLOUT_STEPS, total_steps=ROLLOUT_STEPS, seed=0
        )
        trainer = _make_trainer(obs_auditor=auditor, seed=42)
        trainer.collect_rollout(ROLLOUT_STEPS)
        assert len(auditor.samples) > 0
        for sample in auditor.samples:
            # raw_action is [house, mode, signal]; action_taken is [house, mode].
            assert sample.raw_action[0] == sample.action_taken[0]
            assert sample.raw_action[1] == sample.action_taken[1]
            # Scenario info must mark the mode as always_valid (or None for
            # scenarios that pre-date #316).
            avm = sample.scenario_info.get("action_validity_mode")
            assert avm in (None, "always_valid")

    def test_adjacent_only_mode_records_raw_and_sanitized(self):
        """With ``action_validity_mode == "adjacent_only"``, an out-of-reach
        raw target must be rewritten by the env's ``_sanitize_actions``.

        We can't easily force a specific raw action through the trainer's
        policy network in a unit test, but we can run a rollout in
        ``adjacent_only`` mode and verify that:

        - The auditor records the mode in scenario_info.
        - At least *some* sample shows ``raw_action[0] != action_taken[0]``
          (or, if the policy happens to always pick in-range, that the
          sanitization-divergence count matches the env's own behavior).
        """
        auditor = ObsAuditor(
            num_samples=ROLLOUT_STEPS, total_steps=ROLLOUT_STEPS, seed=0
        )
        trainer = _make_trainer(
            obs_auditor=auditor, seed=42, env_fn_to_use=_adjacent_env_fn
        )
        trainer.collect_rollout(ROLLOUT_STEPS)
        assert len(auditor.samples) > 0
        # The scenario info must mark the mode.
        for sample in auditor.samples:
            assert sample.scenario_info.get("action_validity_mode") == "adjacent_only"
        # The audit captures both raw and sanitized fields; verify they have
        # the right shapes regardless of whether divergence occurred.
        for sample in auditor.samples:
            assert sample.raw_action.shape == (3,)
            assert sample.action_taken.shape == (2,)
        # We expect *some* divergence over a 32-step rollout with random init.
        # The env's home positions are spread around the ring, so the random
        # policy will almost always emit an out-of-reach target sometimes.
        divergences = sum(
            1
            for sample in auditor.samples
            if sample.raw_action[0] != sample.action_taken[0]
        )
        # This is statistical, so allow a wide margin. With 4 agents on 10
        # houses where each agent can only reach 3 (home + 2 adjacent), a
        # uniform policy diverges ~70% of the time.
        assert divergences > 0, (
            "Expected at least one raw != sanitized action over the "
            "rollout, got 0. Either the policy happened to always pick "
            "in-range or the wire-in failed to capture raw_action."
        )

    def test_manually_compute_sanitization(self):
        """Direct env-level sanity check: feed an out-of-reach action to the
        env in ``adjacent_only`` mode and verify ``last_actions`` reflects
        the rewrite, while a hand-built audit sample captures both."""
        env = _adjacent_env_fn()
        env.reset(seed=0)
        # Pick agent 0's home and a guaranteed-out-of-reach target.
        home = int(env.agent_home_positions[0])
        far = (home + 5) % env.num_houses  # ring distance ~5, > 1

        # Build a raw action that targets ``far`` for agent 0, home for others.
        raw_actions = np.zeros((NUM_AGENTS, 3), dtype=np.int64)
        for i in range(NUM_AGENTS):
            raw_actions[i, 0] = int(env.agent_home_positions[i])
        raw_actions[0, 0] = far

        env.step(raw_actions)
        # After step, env.last_actions[0, 0] must be home (sanitized).
        assert int(env.last_actions[0, 0]) == home
        # Build a sample matching what the wire-in would record.
        auditor = ObsAuditor(num_samples=1, total_steps=1, seed=0)
        auditor.maybe_record(
            agent_id=0,
            flat_obs=np.zeros(20, dtype=np.float32),
            env_state={"houses": env.houses.copy()},
            raw_action=raw_actions[0],
            action_taken=np.asarray(env.last_actions[0], dtype=np.int64),
            reward=0.0,
            scenario_info={"action_validity_mode": "adjacent_only"},
            num_agents=NUM_AGENTS,
        )
        s = auditor.samples[0]
        assert int(s.raw_action[0]) == far
        assert int(s.action_taken[0]) == home
        assert s.raw_action[0] != s.action_taken[0]


# ----------------------------------------------------------------------
# 5. Edge cases
# ----------------------------------------------------------------------


class TestEdgeCases:
    def test_num_samples_zero_is_no_op(self):
        auditor = ObsAuditor(num_samples=0, total_steps=100, seed=0)
        trainer = _make_trainer(obs_auditor=auditor, seed=42)
        trainer.collect_rollout(ROLLOUT_STEPS)
        assert auditor.samples == []
        assert auditor.enabled is False

    def test_num_samples_exceeds_total_steps_records_all(self):
        auditor = ObsAuditor(num_samples=1000, total_steps=10, seed=0)
        # When N >= total_steps, every step is picked.
        assert len(auditor.picked_steps()) == 10
        assert set(auditor.picked_steps()) == set(range(10))

    def test_picked_steps_are_deterministic_for_seed(self):
        a = ObsAuditor(num_samples=20, total_steps=1000, seed=42).picked_steps()
        b = ObsAuditor(num_samples=20, total_steps=1000, seed=42).picked_steps()
        c = ObsAuditor(num_samples=20, total_steps=1000, seed=43).picked_steps()
        assert a == b
        assert a != c
        assert len(a) == 20

    def test_negative_args_rejected(self):
        with pytest.raises(ValueError):
            ObsAuditor(num_samples=-1, total_steps=10, seed=0)
        with pytest.raises(ValueError):
            ObsAuditor(num_samples=1, total_steps=-1, seed=0)
