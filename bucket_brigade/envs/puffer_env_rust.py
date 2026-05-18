"""
Rust-backed PufferLib-compatible environment wrapper for Bucket Brigade.

This wraps the Rust bucket_brigade_core to work with PufferLib's multi-agent
training framework with 100x performance improvement.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Optional, Tuple, List, Any
import logging
import bucket_brigade_core as core

from ..agents import create_random_agent, create_archetype_agent

logger = logging.getLogger(__name__)


def _heuristic_action(
    theta: np.ndarray,
    obs_dict: dict[str, np.ndarray],
    agent_id: int,
    rng: np.random.Generator,
) -> list[int]:
    """Simplified heuristic action selection for opponent agents.

    Issue #235: returns a 3-element ``[house, mode, signal]``. This helper
    is the opponent-agent fast-path and defaults to honest signaling
    (``signal == mode``); the full Python ``HeuristicAgent.act`` is what
    drives the Liar archetype's deceptive behavior.

    Issue #254: ring size is inferred from ``len(obs_dict['houses'])``
    rather than hardcoding 10. With v2_minimal (2 houses) the owned-house
    derivation wraps modulo the ring size, matching the engine's
    round-robin ``house_owners`` assignment.
    """
    work_tendency = theta[1]
    own_house_priority = theta[3]
    rest_reward_bias = theta[8]

    num_houses = len(obs_dict["houses"])

    if rng.random() < work_tendency * (1 - rest_reward_bias):
        owned_house = agent_id % num_houses
        if obs_dict["houses"][owned_house] == 1 and rng.random() < own_house_priority:
            house = owned_house
        else:
            burning = [i for i, h in enumerate(obs_dict["houses"]) if h == 1]
            if burning:
                house = rng.choice(burning)
            else:
                house = owned_house
        mode = 1  # WORK
    else:
        house = agent_id % num_houses
        mode = 0  # REST

    # Honest signal: broadcast matches actual mode.
    return [house, mode, mode]


class RustPufferBucketBrigade(gym.Env):
    """
    Rust-backed PufferLib-compatible wrapper for Bucket Brigade.

    100x faster than Python implementation for RL training.
    """

    def __init__(
        self,
        scenario: Optional[Any] = None,
        num_opponents: int = 3,
        opponent_policies: Optional[List[str]] = None,
        max_steps: int = 50,
    ):
        """
        Initialize the Rust-backed PufferLib environment.

        Args:
            scenario: Either a scenario name (``str``) from ``core.SCENARIOS``
                (uses ``"trivial_cooperation"`` if ``None``) **or** a
                pre-built ``PyScenario`` object. Passing the object lets
                callers customize ``commitment_mode`` (issue #331) and other
                mutable fields without depending on global SCENARIOS state.
            num_opponents: Number of opponent agents
            opponent_policies: List of opponent policy types
            max_steps: Maximum steps per episode
        """
        super().__init__()

        self.num_opponents = num_opponents
        self.num_agents = num_opponents + 1  # +1 for trained agent
        self.max_steps = max_steps
        self.steps_taken = 0

        # Default opponent policies
        if opponent_policies is None:
            opponent_policies = ["random", "random", "firefighter", "coordinator"]
        self.opponent_policies = opponent_policies[:num_opponents]

        # Get scenario from Rust core (single source of truth). Issue #331:
        # also accept a pre-built ``PyScenario`` object so callers can
        # opt into ``commitment_mode="two_phase"`` (or other field
        # mutations) without polluting the global SCENARIOS dict.
        if scenario is None or isinstance(scenario, str):
            scenario_name = scenario or "trivial_cooperation"
            if scenario_name not in core.SCENARIOS:
                raise ValueError(
                    f"Unknown scenario '{scenario_name}'. "
                    f"Available: {list(core.SCENARIOS.keys())}"
                )
            self.rust_scenario = core.SCENARIOS[scenario_name]
            self.scenario_name = scenario_name
        else:
            # Treat as a pre-built scenario object (PyScenario).
            self.rust_scenario = scenario
            self.scenario_name = getattr(scenario, "name", "<custom>")

        # Issue #252/#331: within-night commitment mode. When the scenario
        # opts into ``commitment_mode == "two_phase"`` the wrapper takes
        # the **super-step** plumbing path (option (b) from issue #331):
        # the two micro-rounds are flattened into a single
        # ``env.step(action)`` from Puffer's perspective. The action
        # vector is widened from ``[house, mode, signal_r2]`` (3 dims) to
        # ``[house, mode, signal_r2, signal_r1]`` (4 dims). Internally we
        # call ``self.env.step_two_phase(r1_signals, r2_actions)`` once
        # per super-step, where ``r2_actions = [house, mode, signal_r2]``
        # and ``r1_signals = [signal_r1]`` per agent.
        #
        # Why super-step (option b) and not 2-step-per-night (option a):
        # Puffer's vectorized rollout assumes one action tensor per
        # ``env.step()``; doubling the step count to expose round-1 as
        # its own step would confuse the rollout buffer and require
        # paired-transition handling upstream. The super-step path keeps
        # the per-env step count identical to simultaneous mode at the
        # cost of folding the round-1 signal head into the same forward
        # pass as the round-2 action head. Trade-off: the round-1
        # log-prob is not split out into a separate PPO update term, but
        # the env-side mechanic (round-1 obs channel, deception
        # opportunity) is preserved end-to-end.
        #
        # See ``JointPPOTrainer.collect_rollout`` for the alternative
        # rollout-side path that splits round-1 and round-2 into two
        # separate forward passes; the puffer path here is the
        # vectorized-friendly variant.
        self._commitment_mode: str = str(
            getattr(self.rust_scenario, "commitment_mode", "simultaneous")
        )

        # Issue #254: derive ring size from the scenario rather than
        # hardcoding 10. Pre-#254 scenarios keep num_houses=10 (Rust
        # default), so action/observation shapes are bit-exact for them.
        # New scenarios like v2_minimal use a smaller ring (2 houses).
        self.num_houses: int = int(getattr(self.rust_scenario, "num_houses", 10))

        # Initialize Rust environment
        self.env: Optional[core.BucketBrigade] = None  # Will be created in reset()

        # RNG for heuristic decisions
        self.rng = np.random.RandomState()

        # Create opponent agents (will be refreshed each episode)
        self.opponent_agents: List[Any] = []

        # Track last observation
        self._last_obs: Optional[Dict[str, np.ndarray]] = None

        # PufferLib observation/action spaces.
        # Issue #204 — the flattened observation now ends with a per-agent
        # identity one-hot of length ``num_agents`` so per-agent obs vectors
        # are distinct. This wrapper only exposes the trained agent's view
        # (always ``agent_id=0``), but the dim must include the identity
        # tail so downstream policy networks size their input layer correctly.
        # Issue #254: `houses` and scenario_info dimensions now scale with
        # num_houses for the houses channel (scenario_info stays at 10
        # elements — it's a fixed feature vector, not a per-house tensor).
        obs_size = (
            self.num_houses
            + self.num_agents
            + self.num_agents
            + (self.num_agents * 2)
            + 10
            + self.num_agents  # identity one-hot (issue #204)
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Action layout:
        # - ``simultaneous`` (default): ``[house_index, mode, signal]``
        #   (issue #235). 3-element MultiDiscrete. ``house_index`` in
        #   ``[0, num_houses-1]``, ``mode`` in ``{0=REST, 1=WORK}``,
        #   ``signal`` in ``{0=REST, 1=WORK}``. The signal is broadcast
        #   independently of the mode; honest agents emit
        #   ``signal == mode``, liars don't.
        # - ``two_phase`` (issue #252/#331): ``[house_index, mode,
        #   signal_r2, signal_r1]``. 4-element MultiDiscrete. The extra
        #   trailing dim is the round-1 non-binding commitment signal
        #   (binary). ``signal_r2`` (column 2) is the round-2 broadcast
        #   that overwrites ``self.signals``; ``signal_r1`` (column 3) is
        #   written into ``round1_signals`` (visible to the round-2 obs).
        #   Deception is preserved: ``signal_r1 != mode`` is allowed.
        # Issue #254: ``house_index`` range scales with scenario num_houses.
        if self._commitment_mode == "two_phase":
            self.action_space = spaces.MultiDiscrete([self.num_houses, 2, 2, 2])
        else:
            self.action_space = spaces.MultiDiscrete([self.num_houses, 2, 2])

        # Track episode statistics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []

        # Cache for observations
        self._last_obs = None

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        # Reset step counter
        self.steps_taken = 0

        # Create new Rust environment for this episode
        episode_seed = self.rng.randint(0, 2**31 - 1) if seed is not None else None
        self.env = core.BucketBrigade(
            self.rust_scenario, self.num_agents, seed=episode_seed
        )

        # Create new opponent agents for this episode
        self._create_opponent_agents()

        # Get initial observation from Rust env
        rust_obs = self.env.get_observation(0)
        obs_dict = {
            "houses": np.array(rust_obs.houses),
            "signals": np.array(rust_obs.signals),
            "locations": np.array(rust_obs.locations),
            "last_actions": np.zeros(
                (self.num_agents, 2)
            ),  # No actions yet (issue #235: width stays 2 per obs_to_vector compat)
            "scenario_info": np.array(
                [
                    self.rust_scenario.prob_fire_spreads_to_neighbor,
                    self.rust_scenario.prob_solo_agent_extinguishes_fire,
                    self.rust_scenario.prob_house_catches_fire,
                    self.rust_scenario.team_reward_house_survives,
                    self.rust_scenario.team_penalty_house_burns,
                    self.rust_scenario.cost_to_work_one_night,
                    float(self.rust_scenario.min_nights),
                    float(self.num_agents),
                    0.0,  # Padding
                    0.0,  # Padding
                ]
            ),
        }
        self._last_obs = obs_dict

        # Convert observation to flat array for PufferLib
        flat_obs = self._flatten_observation(obs_dict, agent_id=0)

        return flat_obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment.

        Issue #331: when the scenario has ``commitment_mode == "two_phase"``
        the action layout is ``[house, mode, signal_r2, signal_r1]`` (4
        dims). Internally this method calls
        ``self.env.step_two_phase(r1_signals, r2_actions)`` once per
        ``step()`` (the super-step plumbing). For simultaneous mode the
        action layout and behavior are bit-identical to pre-#331.
        """
        self.steps_taken += 1

        # Convert action to [house, mode, signal] format (issue #235),
        # plus extract the round-1 signal in two-phase mode (issue #331).
        # Legacy 2-element actions default ``signal := mode`` (honest).
        trained_r1_signal: int
        if self._commitment_mode == "two_phase":
            # Expected 4-element action; tolerate length-3 by defaulting
            # the round-1 signal to the round-2 signal (honest two-phase).
            if len(action) >= 4:
                agent_action = [int(action[0]), int(action[1]), int(action[2])]
                trained_r1_signal = int(action[3])
            elif len(action) == 3:
                agent_action = [int(action[0]), int(action[1]), int(action[2])]
                trained_r1_signal = int(action[2])
            else:
                mode = int(action[1])
                agent_action = [int(action[0]), mode, mode]
                trained_r1_signal = mode
        else:
            if len(action) >= 3:
                agent_action = [int(action[0]), int(action[1]), int(action[2])]
            else:
                mode = int(action[1])
                agent_action = [int(action[0]), mode, mode]
            trained_r1_signal = 0  # unused

        # Get actions from all agents
        all_actions = [agent_action]  # Trained agent first

        # Get opponent actions using heuristic or agent policy
        for i, opponent in enumerate(self.opponent_agents):
            opponent_action = opponent.act(self._last_obs)
            all_actions.append(opponent_action)

        # Step the Rust environment
        assert self.env is not None, "Environment not initialized"
        if self._commitment_mode == "two_phase":
            # Round-1 commitment signals: trained agent's r1_signal from
            # the expanded action; opponents commit honestly (r1 ==
            # round-2 signal). This matches the JointPPOTrainer
            # convention for heuristic opponents: opponents are not
            # asked to "lie" at the commitment phase by default. A
            # follow-up could expose round-1 signaling to opponent
            # policies if needed.
            r1_signals: List[int] = [trained_r1_signal]
            for opp_a in all_actions[1:]:
                # opp_a is [house, mode, signal] (length 3) from the
                # opponent policy. Use the broadcast signal as the
                # round-1 commitment (honest two-phase signal).
                r1_signals.append(int(opp_a[2]) if len(opp_a) >= 3 else int(opp_a[1]))
            rewards_list, done, info = self.env.step_two_phase(r1_signals, all_actions)
        else:
            rewards_list, done, info = self.env.step(all_actions)

        # Get observations for all agents
        rust_obs = self.env.get_observation(0)
        obs_dict = {
            "houses": np.array(rust_obs.houses),
            "signals": np.array(rust_obs.signals),
            "locations": np.array(rust_obs.locations),
            "last_actions": np.array(all_actions),
            "scenario_info": self._last_obs["scenario_info"]
            if self._last_obs is not None
            else np.zeros(10),  # Reuse
        }
        self._last_obs = obs_dict

        # Extract reward for trained agent (first agent)
        reward = float(rewards_list[0])

        # Check termination conditions
        terminated = done
        truncated = self.steps_taken >= self.max_steps

        # Convert observation for trained agent
        flat_obs = self._flatten_observation(obs_dict, agent_id=0)

        return flat_obs, reward, terminated, truncated, info

    def _create_opponent_agents(self) -> None:
        """Create opponent agents for this episode."""
        self.opponent_agents = []

        for i, policy_type in enumerate(self.opponent_policies):
            agent_id = i + 1  # 0 is trained agent

            if policy_type == "random":
                agent = create_random_agent(agent_id)
            elif policy_type in [
                "firefighter",
                "coordinator",
                "free_rider",
                "liar",
                "hero",
            ]:
                agent = create_archetype_agent(policy_type, agent_id)
            else:
                # Default to random
                agent = create_random_agent(agent_id)

            self.opponent_agents.append(agent)

    def _flatten_observation(
        self, obs: Dict[str, np.ndarray], agent_id: int
    ) -> np.ndarray:
        """Convert observation dict to flat array for PufferLib.

        Issue #204: the flat layout now ends with a length-``num_agents``
        identity one-hot for ``agent_id`` so per-agent observations are
        provably distinct. The previous layout dropped ``agent_id``
        entirely, which was the latent half of the "identical input to all
        agents" PPO bug.
        """
        houses = obs["houses"].astype(np.float32)
        signals = obs["signals"].astype(np.float32)
        locations = obs["locations"].astype(np.float32)
        last_actions = obs["last_actions"].flatten().astype(np.float32)
        scenario_info = obs["scenario_info"].astype(np.float32)

        if not 0 <= agent_id < self.num_agents:
            raise ValueError(
                f"_flatten_observation: agent_id {agent_id} out of range "
                f"[0, {self.num_agents})"
            )
        identity = np.zeros(self.num_agents, dtype=np.float32)
        identity[agent_id] = 1.0

        # Concatenate all observation components
        flat_obs = np.concatenate(
            [
                houses,  # 10 values
                signals,  # N values
                locations,  # N values
                last_actions,  # N*2 values
                scenario_info,  # 10 values
                identity,  # N values (issue #204)
            ]
        )

        return flat_obs

    def render(self) -> None:
        """Render the environment."""
        # Could implement basic console rendering if needed
        pass

    def close(self) -> None:
        """Close the environment."""
        pass


class RustPufferBucketBrigadeVectorized(RustPufferBucketBrigade):
    """
    Vectorized Rust-backed version for multiple parallel environments.
    """

    def __init__(self, num_envs: int = 1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.num_envs = num_envs

        # Vectorized observation space (issue #204 — include identity one-hot tail).
        # Issue #254: houses channel scales with num_houses; rest unchanged.
        obs_size = (
            self.num_houses
            + self.num_agents
            + self.num_agents
            + (self.num_agents * 2)
            + 10
            + self.num_agents  # identity one-hot (issue #204)
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_envs, obs_size), dtype=np.float32
        )

        # Issue #235: 3-element action [house, mode, signal] per env.
        # Issue #254: house dim scales with num_houses.
        # Issue #331: in two-phase mode, action gains a trailing
        # ``signal_r1`` dim so the per-env shape is 4 not 3.
        if self._commitment_mode == "two_phase":
            self.action_space = spaces.MultiDiscrete(
                [num_envs, self.num_houses, 2, 2, 2]
            )
        else:
            self.action_space = spaces.MultiDiscrete([num_envs, self.num_houses, 2, 2])

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset all environments."""
        if seed is not None:
            np.random.seed(seed)

        # For simplicity, just reset the single environment
        # In a full implementation, this would handle multiple envs
        obs, info = super().reset(seed, options)
        return np.array([obs]), info

    def step(  # type: ignore[override]
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Step all environments."""
        # For simplicity, just step the single environment
        # In a full implementation, this would handle multiple envs
        obs, reward, terminated, truncated, info = super().step(actions[0])
        return (
            np.array([obs]),
            np.array([reward]),
            np.array([terminated]),
            np.array([truncated]),
            [info],
        )


# Factory functions for easy environment creation
def make_rust_env(
    scenario_name: str = "trivial_cooperation", num_opponents: int = 3
) -> RustPufferBucketBrigade:
    """Create a Rust-backed PufferLib environment with specified scenario.

    Scenarios are loaded directly from bucket_brigade_core.SCENARIOS (Rust source of truth).
    """
    return RustPufferBucketBrigade(scenario_name, num_opponents)


def make_rust_vectorized_env(
    num_envs: int = 8,
    scenario_name: str = "trivial_cooperation",
    num_opponents: int = 3,
) -> RustPufferBucketBrigadeVectorized:
    """Create a vectorized Rust-backed environment for parallel training.

    Scenarios are loaded directly from bucket_brigade_core.SCENARIOS (Rust source of truth).
    """
    return RustPufferBucketBrigadeVectorized(
        num_envs=num_envs, scenario=scenario_name, num_opponents=num_opponents
    )
