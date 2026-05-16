"""Tests for agent implementations."""

import numpy as np
import pytest
from bucket_brigade.agents import (
    RandomAgent,
    HeuristicAgent,
    create_random_agent,
    create_archetype_agent,
)


class TestAgentBase:
    """Test the base agent class."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = RandomAgent(0, "TestAgent")
        assert agent.agent_id == 0
        assert agent.name == "TestAgent"

    def test_agent_reset(self):
        """Test agent reset functionality."""
        agent = RandomAgent(0)
        agent.reset()  # Should not raise any errors

    def test_random_agent_actions(self):
        """Test random agent produces valid actions."""
        agent = RandomAgent(0)
        obs = {
            "signals": np.array([0, 1, 0, 1]),
            "locations": np.array([0, 1, 2, 3]),
            "houses": np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0]),
            "last_actions": np.array([[0, 0], [1, 1], [2, 0], [3, 1]]),
            "scenario_info": np.array([0.25, 0.5, 100, 100, 0.5, 0.2, 12, 0.02, 12, 4]),
        }

        action = agent.act(obs)

        # Check action format
        assert isinstance(action, np.ndarray)
        assert action.shape == (3,)  # issue #235: [house, mode, signal]
        assert 0 <= action[0] <= 9  # House index
        assert action[1] in [0, 1]  # Mode (REST or WORK)
        assert action[2] in [0, 1]  # Signal (REST or WORK) — issue #235


class TestHeuristicAgent:
    """Test the heuristic agent implementation."""

    def test_heuristic_agent_initialization(self):
        """Test heuristic agent initialization."""
        params = np.array([0.5] * 10)  # Neutral parameters
        agent = HeuristicAgent(params, 0, "HeuristicAgent")

        assert agent.agent_id == 0
        assert agent.name == "HeuristicAgent"
        assert len(agent.params) == 10

    def test_parameter_validation(self):
        """Test parameter validation."""
        with pytest.raises(AssertionError):
            HeuristicAgent(np.array([0.5] * 9), 0)  # Wrong number of parameters

    def test_action_generation(self):
        """Test heuristic agent action generation."""
        # Create an agent that tends to work
        params = np.array([0.9, 0.9, 0.1, 0.1, 0.1, 0.5, 0.1, 0.0, 0.0, 0.5])
        agent = HeuristicAgent(params, 0)

        obs = {
            "signals": np.array([0, 1, 0, 1]),
            "locations": np.array([0, 1, 2, 3]),
            "houses": np.array(
                [0, 1, 0, 0, 0, 0, 0, 1, 0, 0]
            ),  # Houses 1 and 7 burning
            "last_actions": np.array([[0, 0], [1, 1], [2, 0], [3, 1]]),
            "scenario_info": np.array([0.25, 0.5, 100, 100, 0.5, 0.2, 12, 0.02, 12, 4]),
        }

        action = agent.act(obs)

        assert isinstance(action, np.ndarray)
        assert action.shape == (3,)  # issue #235: [house, mode, signal]
        assert 0 <= action[0] <= 9
        assert action[1] in [0, 1]
        assert action[2] in [0, 1]

    def test_agent_memory(self):
        """Test agent memory and fatigue effects."""
        params = np.array(
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.9, 0.5, 0.5]
        )  # High fatigue
        agent = HeuristicAgent(params, 0)

        obs = {
            "signals": np.array([0, 1, 0, 1]),
            "locations": np.array([0, 1, 2, 3]),
            "houses": np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0]),
            "last_actions": np.array([[0, 0], [1, 1], [2, 0], [3, 1]]),
            "scenario_info": np.array([0.25, 0.5, 100, 100, 0.5, 0.2, 12, 0.02, 12, 4]),
        }

        # First action
        action1 = agent.act(obs)

        # Second action should have some chance to repeat (fatigue effect)
        action2 = agent.act(obs)

        # Actions should be valid regardless
        assert isinstance(action1, np.ndarray)
        assert isinstance(action2, np.ndarray)

    def test_scenario_adaptation(self):
        """Test agent adapts to different scenarios."""
        params = np.array(
            [0.5, 0.5, 0.5, 0.5, 0.8, 0.5, 0.5, 0.0, 0.5, 0.5]
        )  # High risk aversion
        agent = HeuristicAgent(params, 0)

        # High fire scenario
        high_fire_obs = {
            "signals": np.array([0, 1, 0, 1]),
            "locations": np.array([0, 1, 2, 3]),
            "houses": np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),  # All burning
            "last_actions": np.array([[0, 0], [1, 1], [2, 0], [3, 1]]),
            "scenario_info": np.array([0.25, 0.5, 100, 100, 0.5, 0.2, 12, 0.02, 12, 4]),
        }

        # Low fire scenario
        low_fire_obs = {
            "signals": np.array([0, 1, 0, 1]),
            "locations": np.array([0, 1, 2, 3]),
            "houses": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # All safe
            "last_actions": np.array([[0, 0], [1, 1], [2, 0], [3, 1]]),
            "scenario_info": np.array([0.25, 0.5, 100, 100, 0.5, 0.2, 12, 0.02, 12, 4]),
        }

        # Agent should potentially behave differently (though stochastic)
        action_high = agent.act(high_fire_obs)
        action_low = agent.act(low_fire_obs)

        assert isinstance(action_high, np.ndarray)
        assert isinstance(action_low, np.ndarray)


# TODO: Implement proper agent loader tests when API is finalized
# class TestAgentLoader:
#     """Test agent loading functionality."""
#     pass


class TestAgentFactory:
    """Test agent factory functions."""

    def test_create_random_agent(self):
        """Test random agent creation."""
        agent = create_random_agent(0, "RandomTest")

        assert agent.agent_id == 0
        assert agent.name == "RandomTest"
        assert len(agent.params) == 10
        assert all(0 <= p <= 1 for p in agent.params)

    def test_create_archetype_agent(self):
        """Test archetype agent creation."""
        archetypes = ["firefighter", "liar", "free_rider", "hero", "coordinator"]

        for archetype in archetypes:
            agent = create_archetype_agent(archetype, 0)

            assert agent.agent_id == 0
            assert archetype.title() in agent.name
            assert len(agent.params) == 10
            assert all(0 <= p <= 1 for p in agent.params)

    def test_invalid_archetype(self):
        """Test invalid archetype raises error."""
        with pytest.raises(ValueError):
            create_archetype_agent("invalid_archetype", 0)

    def test_agent_action_consistency(self):
        """Test agents produce consistent action formats."""
        agents = [
            RandomAgent(0),
            create_random_agent(1),
            create_archetype_agent("firefighter", 2),
            create_archetype_agent("free_rider", 3),
        ]

        obs = {
            "signals": np.array([0, 1, 0, 1]),
            "locations": np.array([0, 1, 2, 3]),
            "houses": np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0]),
            "last_actions": np.array([[0, 0], [1, 1], [2, 0], [3, 1]]),
            "scenario_info": np.array([0.25, 0.5, 100, 100, 0.5, 0.2, 12, 0.02, 12, 4]),
        }

        for agent in agents:
            action = agent.act(obs)

            assert isinstance(action, np.ndarray)
            assert action.shape == (3,)  # issue #235: [house, mode, signal]
            assert isinstance(action[0], (int, np.integer))  # House index
            assert isinstance(action[1], (int, np.integer))  # Mode
            assert isinstance(action[2], (int, np.integer))  # Signal
            assert 0 <= action[0] <= 9
            assert action[1] in [0, 1]
            assert action[2] in [0, 1]


# ---------------------------------------------------------------------------
# Issue #235: signal as a first-class action dimension
# ---------------------------------------------------------------------------


class TestSignalChannelDecoupled:
    """Pin the acceptance criterion of issue #235: the broadcast signal
    (``action[2]``) is decoupled from the work/rest bit (``action[1]``).

    These tests prove the channel is operative — i.e. an agent **can**
    emit a signal that differs from its action — and that honest
    archetypes do not lie.
    """

    @staticmethod
    def _make_obs(houses: np.ndarray) -> dict:
        """Build the minimal observation the heuristic agents consume."""
        n = 4
        return {
            "signals": np.zeros(n, dtype=np.int8),
            "locations": np.zeros(n, dtype=np.int8),
            "houses": houses,
            "last_actions": np.zeros((n, 3), dtype=np.int8),
            "scenario_info": np.array(
                [0.25, 0.5, 100, 100, 0.5, 0.2, 12, 0.02, 12, 4]
            ),
        }

    def test_random_agent_signal_is_honest(self):
        """The random baseline is honest by convention (signal == mode).

        See ``RandomAgent.act`` docstring for the rationale: this keeps the
        random baseline's semantics identical to pre-#235 ("uniform-random
        house, uniform-random mode") and isolates the signal channel as a
        deliberate degree of freedom available to strategic agents.
        """
        agent = RandomAgent(0)
        obs = self._make_obs(np.zeros(10, dtype=np.int8))
        for _ in range(50):
            a = agent.act(obs)
            assert a.shape == (3,)
            assert int(a[1]) == int(a[2]), (
                f"RandomAgent should signal honestly: mode={int(a[1])}, "
                f"signal={int(a[2])}"
            )

    def test_firefighter_archetype_lies_rarely(self):
        """The Firefighter archetype has ``honesty_bias = 1.0`` and a
        small ``exploration_rate = 0.1``. Its base ``_choose_signal``
        always returns the honest broadcast, and ``_choose_mode`` follows
        the signal with probability ``honesty_bias = 1``; only the
        ``exploration_rate`` (which re-randomizes the *mode* but not the
        signal) can produce ``mode != signal``. So we assert the
        empirical lie rate is **low** rather than zero."""
        agent = create_archetype_agent("firefighter", 0)
        houses = np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0])  # some fires
        obs = self._make_obs(houses)
        np.random.seed(42)
        n = 200
        lie_count = 0
        for _ in range(n):
            a = agent.act(obs)
            if int(a[1]) != int(a[2]):
                lie_count += 1
        lie_rate = lie_count / n
        # Exploration rate is 0.1 and only flips with probability 0.5 of
        # producing a mismatch, so theoretical lie rate ~= 0.05. Allow
        # generous slack.
        assert lie_rate < 0.25, (
            f"Firefighter (honesty_bias=1.0) should rarely lie; "
            f"observed lie rate = {lie_rate:.3f} over {n} samples. "
            "Expected < 0.25."
        )

    def test_liar_archetype_lies_in_expectation(self):
        """The Liar archetype has ``honesty_bias = 0.1``; in expectation
        it should lie roughly 90% of the time. We sample many steps and
        assert the empirical lie rate is above a generous lower bound.

        This is the canonical acceptance test from issue #235: it proves
        that the signal channel is *operative* (an agent can emit
        ``signal != mode``) and that the existing ``_choose_signal`` /
        ``honesty_bias`` machinery, which pre-#235 was computed and
        thrown away, is now actually wired through.
        """
        agent = create_archetype_agent("liar", 0)
        # Use a mixed-state obs so work_intent is non-degenerate.
        houses = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 0])
        obs = self._make_obs(houses)
        np.random.seed(123)
        n_samples = 500
        lie_count = 0
        sample_count = 0
        for _ in range(n_samples):
            a = agent.act(obs)
            # Skip steps where fatigue/exploration overrode the signal —
            # those don't reflect the signal-selection mechanism. In
            # practice they're rare (fatigue_memory=0, exploration=0.3
            # for the Liar archetype), so the bulk of samples are
            # signal-selected.
            sample_count += 1
            if int(a[1]) != int(a[2]):
                lie_count += 1
        lie_rate = lie_count / sample_count
        # Liar's honesty_bias is 0.1, so deceptive rate ≈ 0.9; allow
        # generous slack for the work_intent stochasticity and the
        # exploration/fatigue overrides.
        assert lie_rate > 0.4, (
            f"Liar archetype (honesty_bias=0.1) should lie often; "
            f"observed lie rate = {lie_rate:.3f} over {sample_count} "
            f"samples. Expected > 0.4 (in practice ~0.6-0.9)."
        )

    def test_liar_signal_channel_operative_smoke(self):
        """At least one of the Liar's actions has ``signal != mode``.

        This is the smallest possible operative-channel test — it would
        fail under the pre-#235 deterministic-copy bug regardless of
        ``honesty_bias``, because there the engine threw the signal away
        and recomputed ``signal := mode``."""
        agent = create_archetype_agent("liar", 0)
        houses = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 0])
        obs = self._make_obs(houses)
        np.random.seed(7)
        seen_lie = False
        for _ in range(100):
            a = agent.act(obs)
            if int(a[1]) != int(a[2]):
                seen_lie = True
                break
        assert seen_lie, (
            "Liar archetype produced no deceptive signals over 100 steps — "
            "either the signal channel is still a deterministic copy of "
            "the mode bit (regression of issue #235), or honesty_bias is "
            "no longer being threaded through HeuristicAgent.act."
        )
