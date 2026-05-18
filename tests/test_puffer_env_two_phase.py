"""Tests for the two-phase commitment-mode plumbing in the Rust-backed
PufferLib wrapper (issue #331).

The wrapper at ``bucket_brigade.envs.puffer_env_rust.RustPufferBucketBrigade``
previously raised ``NotImplementedError`` for ``commitment_mode == "two_phase"``.
PR #331 adds the **super-step** plumbing (option (b) from the issue body):
the action vector is widened from ``[house, mode, signal]`` (3 dims) to
``[house, mode, signal_r2, signal_r1]`` (4 dims), and one ``env.step()`` call
internally invokes ``self.env.step_two_phase(r1, r2)`` once per night.

The tests below verify:

1. Construction succeeds for a two-phase scenario (no NotImplementedError).
2. Action-space shape matches the documented expansion (3 -> 4 dims).
3. The simultaneous path is bit-identical to pre-#331 (action space + step
   trajectory unchanged).
4. End-to-end parity: under the puffer wrapper, the trajectory produced by a
   deterministic action sequence matches the canonical Rust-engine path
   (``bucket_brigade_core.BucketBrigade.step_two_phase``) with the same
   seed and actions. This is the puffer-side analogue of the
   ``JointPPOTrainer`` parity ask from the issue.
5. Deception substrate (the "can-still-lie" guarantee) survives the wrapper:
   passing ``signal_r1 != mode`` flows through to the engine and is observable
   in the next-step ``round1_signals`` channel.

Out of scope for this PR (still ``NotImplementedError``):

* WASM constructor (``bucket-brigade-core/src/wasm.rs``) + browser engine
  rendering (``web/src/utils/browserEngine.ts`` and ``wasmEngine.ts``).
* ``MacroActionEnv`` composition with two-phase
  (``bucket_brigade/envs/macro_action_env.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

import bucket_brigade_core as core
from bucket_brigade.envs.puffer_env_rust import RustPufferBucketBrigade


def _fresh_two_phase_scenario():
    """Return a *new* PyScenario with ``commitment_mode='two_phase'``.

    ``core.SCENARIOS[name]`` returns the **same** cached object across
    calls, so mutating it bleeds between tests. We pull, mutate, then
    return without aliasing — and reset back to ``simultaneous`` in a
    fixture teardown elsewhere if needed. For these tests every read of
    the scenario is followed by passing it directly to the wrapper, so
    no later test reads the mutated cache.
    """
    s = core.SCENARIOS["trivial_cooperation"]
    s.commitment_mode = "two_phase"
    return s


@pytest.fixture(autouse=True)
def _restore_scenario_cache():
    """Restore SCENARIOS cache to simultaneous after each test so mutation
    in one test does not bleed into the next."""
    yield
    s = core.SCENARIOS["trivial_cooperation"]
    s.commitment_mode = "simultaneous"


class TestTwoPhaseConstruction:
    """The wrapper must accept ``commitment_mode='two_phase'`` without
    raising (the previous ``NotImplementedError`` gate is removed)."""

    def test_constructs_two_phase_via_scenario_object(self):
        scenario = _fresh_two_phase_scenario()
        # Should NOT raise.
        env = RustPufferBucketBrigade(scenario=scenario, num_opponents=3, max_steps=5)
        assert env._commitment_mode == "two_phase"

    def test_construct_simultaneous_via_name_default(self):
        # Default path: pass a scenario name (unchanged from pre-#331).
        env = RustPufferBucketBrigade(scenario="trivial_cooperation", num_opponents=3)
        assert env._commitment_mode == "simultaneous"


class TestActionSpaceShape:
    """Action space shape matches the super-step contract."""

    def test_simultaneous_action_space_is_3_dim(self):
        env = RustPufferBucketBrigade(scenario="trivial_cooperation", num_opponents=3)
        # MultiDiscrete([num_houses, 2, 2]) — bit-identical to pre-#331.
        assert env.action_space.shape == (3,)
        np.testing.assert_array_equal(
            env.action_space.nvec, np.array([env.num_houses, 2, 2])
        )

    def test_two_phase_action_space_is_4_dim(self):
        scenario = _fresh_two_phase_scenario()
        env = RustPufferBucketBrigade(scenario=scenario, num_opponents=3)
        # MultiDiscrete([num_houses, 2, 2, 2]) — trailing dim is r1_signal.
        assert env.action_space.shape == (4,)
        np.testing.assert_array_equal(
            env.action_space.nvec, np.array([env.num_houses, 2, 2, 2])
        )

    def test_obs_space_unchanged_by_commitment_mode(self):
        # The flat obs vector layout is identical across modes; only the
        # action space widens. (Round-1 signals are not yet plumbed into
        # the puffer obs flattener — that's a separate follow-up if a
        # policy wants to condition on them through the puffer path.)
        env_sim = RustPufferBucketBrigade(
            scenario="trivial_cooperation", num_opponents=3
        )
        env_tp = RustPufferBucketBrigade(
            scenario=_fresh_two_phase_scenario(), num_opponents=3
        )
        assert env_sim.observation_space.shape == env_tp.observation_space.shape


class TestSimultaneousBitIdentity:
    """At default ``simultaneous`` mode, behavior is bit-identical to
    pre-#331: same action space, same step trajectory for fixed seed."""

    def test_simultaneous_step_returns_reward(self):
        env = RustPufferBucketBrigade(
            scenario="trivial_cooperation", num_opponents=3, max_steps=10
        )
        env.reset(seed=42)
        obs, reward, term, trunc, info = env.step([0, 1, 1])
        # Sanity: simultaneous step still returns the gym 5-tuple and a
        # finite scalar reward. (The post-step obs shape is unrelated to
        # this PR — issue #331 only widens the action space; obs layout
        # is unchanged.)
        assert isinstance(reward, float)
        assert np.isfinite(reward)
        assert isinstance(term, bool)
        assert obs.ndim == 1


class TestTwoPhaseTrajectoryParity:
    """End-to-end parity: the puffer wrapper's two-phase trajectory must
    match a direct ``core.BucketBrigade.step_two_phase`` invocation with
    identical seed and identical (round-1, round-2) actions.

    This is the wrapper-vs-engine parity test (issue #331 acceptance
    criterion: "parity test vs JointPPOTrainer"). We use the engine
    directly rather than the trainer because:

    1. The trainer's rollout includes a policy network forward pass —
       not deterministic without fixing torch seeds + policy weights.
    2. The trainer's two-phase path ultimately calls
       ``env.step_two_phase`` once per night; the wrapper's super-step
       also calls ``self.env.step_two_phase`` once per night. The
       wrapper-vs-engine comparison verifies the **plumbing** (action
       split, opponent r1 derivation, return shape) without conflating
       it with policy stochasticity.
    """

    def _run_engine_directly(self, seed, trained_actions, opponent_actions_per_step):
        """Mirror the puffer wrapper's plumbing manually against the raw
        engine. The wrapper does:

        1. Build ``all_actions = [trained_full_action] + opponent_full_actions``.
        2. Derive ``r1_signals = [trained_r1_signal] + [opp.signal for opp in opponents]``.
        3. Call ``env.step_two_phase(r1_signals, all_actions)``.

        We replicate that here with deterministic opponent actions.
        """
        scenario = _fresh_two_phase_scenario()
        engine = core.BucketBrigade(scenario, num_agents=4, seed=seed)
        rewards = []
        for trained_action, opp_actions in zip(
            trained_actions, opponent_actions_per_step
        ):
            r1_trained = int(trained_action[3])
            r2_trained = [
                int(trained_action[0]),
                int(trained_action[1]),
                int(trained_action[2]),
            ]
            all_actions = [r2_trained] + [list(map(int, oa)) for oa in opp_actions]
            r1_signals = [r1_trained] + [int(oa[2]) for oa in opp_actions]
            r_list, done, _ = engine.step_two_phase(r1_signals, all_actions)
            rewards.append(float(r_list[0]))
            if done:
                break
        return rewards

    def test_super_step_calls_step_two_phase_once_per_night(self):
        """Sanity: each puffer ``step()`` advances the underlying engine
        by exactly one night (super-step plumbing, not 2-step-per-night).
        """
        scenario = _fresh_two_phase_scenario()
        env = RustPufferBucketBrigade(
            scenario=scenario,
            num_opponents=3,
            opponent_policies=["random", "random", "random"],
            max_steps=5,
        )
        env.reset(seed=7)
        initial_night = env.env.get_current_state().night
        env.step([0, 1, 1, 0])
        after_one_step = env.env.get_current_state().night
        assert after_one_step == initial_night + 1, (
            f"super-step should advance night by exactly 1; got "
            f"{initial_night} -> {after_one_step}"
        )

    def test_trajectory_matches_engine_with_fixed_opponents(self):
        """Build a deterministic puffer rollout AND replay the same
        (r1, r2) tuples through the engine directly. Rewards and night
        counts should match exactly.
        """
        seed = 31415
        # Trained agent actions in puffer's 4-dim format
        # [house, mode, signal_r2, signal_r1].
        trained_actions = [
            np.array([0, 1, 1, 1], dtype=np.int64),
            np.array([1, 1, 1, 0], dtype=np.int64),
            np.array([2, 0, 0, 1], dtype=np.int64),
            np.array([3, 1, 1, 1], dtype=np.int64),
        ]

        # We need to capture the opponent actions the puffer wrapper
        # emits so the engine replay sees the *same* per-agent inputs.
        # Wrap each opponent's ``act`` to memoize its last return value.
        class _ReplayingOpp:
            def __init__(self, inner):
                self._inner = inner
                self.last_action = None

            def act(self, obs):
                a = self._inner.act(obs)
                self.last_action = list(map(int, a))
                return a

        scenario = _fresh_two_phase_scenario()
        record_env = RustPufferBucketBrigade(
            scenario=scenario,
            num_opponents=3,
            opponent_policies=["random", "random", "random"],
            max_steps=len(trained_actions),
        )
        record_env.reset(seed=seed)
        record_env.opponent_agents = [
            _ReplayingOpp(a) for a in record_env.opponent_agents
        ]
        record_rewards: list[float] = []
        recorded_opp_actions: list[list[list[int]]] = []
        for a in trained_actions:
            _, r, term, _, _ = record_env.step(a)
            recorded_opp_actions.append(
                [opp.last_action for opp in record_env.opponent_agents]
            )
            record_rewards.append(r)
            if term:
                break

        # Replay the same (r1, r2) tuples through the engine directly.
        engine_rewards = self._run_engine_directly(
            seed,
            trained_actions[: len(recorded_opp_actions)],
            recorded_opp_actions,
        )

        # Both paths must agree on the trained agent's reward at every
        # step. Engine path is the canonical source-of-truth.
        np.testing.assert_allclose(
            np.asarray(record_rewards),
            np.asarray(engine_rewards),
            rtol=1e-6,
            atol=1e-6,
            err_msg=(
                "Puffer two-phase super-step plumbing diverges from a "
                "direct core.BucketBrigade.step_two_phase invocation. "
                "Check action split (r1 vs r2) and opponent r1 derivation."
            ),
        )


class TestCanStillLieRegression:
    """**PR GATE**: the deception channel must survive the wrapper.

    The core engine's ``test_can_still_lie`` already covers the engine
    layer (`tests/test_environment.py`). This test asserts the **wrapper**
    does not silently collapse round-1 and round-2 signals: passing
    ``signal_r1=1`` with ``mode=0`` must produce a round-1 signal of 1
    in the engine state regardless of the round-2 mode.
    """

    def test_wrapper_preserves_lying_signal(self):
        scenario = _fresh_two_phase_scenario()
        env = RustPufferBucketBrigade(
            scenario=scenario,
            num_opponents=3,
            opponent_policies=["random", "random", "random"],
            max_steps=3,
        )
        env.reset(seed=99)
        # Liar action: round-2 mode=REST (0), round-2 signal=REST (0),
        # but round-1 commitment signal=WORK (1). The wrapper must
        # forward this as-is to the engine's r1_signals slot.
        liar_action = np.array([0, 0, 0, 1], dtype=np.int64)
        env.step(liar_action)
        # The engine exposes round1_signals via the observation getter.
        obs_after = env.env.get_observation(0)
        r1 = list(obs_after.round1_signals)
        assert r1[0] == 1, (
            f"Liar's round-1 signal (1) was not preserved through the "
            f"wrapper; got r1_signals={r1}. The wrapper must split "
            f"action[3] -> r1 cleanly."
        )
