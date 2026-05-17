"""
Bucket Brigade multi-agent environment implementation.
"""

import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from .scenarios_generated import Scenario


class BucketBrigadeEnv:
    """
    Multi-agent environment implementing the Bucket Brigade firefighting game.

    Agents cooperate on a ring of houses to prevent fires from spreading.
    Ring length is ``scenario.num_houses`` (defaults to 10 for every
    pre-#254 scenario; ``v2_minimal`` uses 2). The game ends when all fires
    are extinguished or all houses are ruined.
    """

    # House states
    SAFE = 0
    BURNING = 1
    RUINED = 2

    # Agent modes
    REST = 0
    WORK = 1

    def __init__(self, scenario: Optional[Scenario] = None, num_agents: int = 4):
        """
        Initialize the Bucket Brigade environment.

        Args:
            scenario: Game scenario parameters. If None, uses default scenario.
            num_agents: Number of agents (4-10). Ignored if scenario is provided.
        """
        if scenario is None:
            from .scenarios_generated import default_scenario

            scenario = default_scenario(num_agents)

        self.scenario = scenario
        self.num_agents = scenario.num_agents
        # Issue #254: read ring size from the scenario rather than hardcoding
        # 10. Every pre-#254 scenario keeps num_houses=10 (the dataclass
        # default), so this is bit-exact backward compatible.
        self.num_houses = int(getattr(scenario, "num_houses", 10))

        # Environment state
        self.houses = np.zeros(
            self.num_houses, dtype=np.int8
        )  # House states: SAFE, BURNING, RUINED
        self.locations = np.zeros(self.num_agents, dtype=np.int8)  # Agent positions
        self.signals = np.zeros(
            self.num_agents, dtype=np.int8
        )  # Agent signals: REST, WORK
        # Issue #235 note: action is now [house, mode, signal] (length 3).
        # But the obs vector ``last_actions`` width stays 2 (house, mode)
        # per the curator's recommendation — the broadcast signal is
        # already exposed through obs.signals, so widening last_actions
        # would just add a redundant feature column. We slice the signal
        # off before storing into ``last_actions``.
        self.last_actions = np.zeros(
            (self.num_agents, 2), dtype=np.int8
        )  # [house, mode] — signal lives in self.signals

        # Game state
        self.night = 0
        self.done = False
        self.rewards = np.zeros(self.num_agents, dtype=np.float32)

        # Trajectory recording for replays
        self.trajectory: List[Dict] = []

        # Previous house state for reward computation
        self._prev_houses_state: np.ndarray = np.zeros(self.num_houses, dtype=np.int8)

        # House ownership: assign agents to houses in round-robin fashion.
        # Issue #254: vector length scales with num_houses.
        self.house_owners = np.arange(self.num_houses) % self.num_agents

        # Per-agent "home position" on the ring (issue #203). Used by
        # ``_compute_rewards`` to scale the per-step work cost by the ring-arc
        # distance from the agent's home to the house it works at. Sourced from
        # ``scenario.agent_home_positions`` when non-empty; otherwise the
        # round-robin ``house_owners`` anchor (agent i -> house i) is reused.
        # When ``scenario.distance_cost_alpha == 0.0`` (the implicit default
        # for every pre-#203 scenario) the spatial term collapses to zero, so
        # behavior is bit-exactly identical to the pre-#203 env.
        #
        # Issue #254: the round-robin fallback now wraps modulo
        # ``num_houses`` so it stays in-range when ``num_agents > num_houses``
        # (e.g. v2_minimal: 4 agents on 2 houses -> [0, 1, 0, 1]).
        if getattr(self.scenario, "agent_home_positions", None):
            self.agent_home_positions = np.array(
                self.scenario.agent_home_positions, dtype=np.int8
            )
        else:
            self.agent_home_positions = np.array(
                [i % self.num_houses for i in range(self.num_agents)], dtype=np.int8
            )

        # Initialize random state for reproducibility
        self.rng = np.random.RandomState()

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Reset the environment to start a new game.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Initial observation dictionary
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        # Reset game state
        self.night = 0
        self.done = False
        self.rewards = np.zeros(self.num_agents, dtype=np.float32)
        self.trajectory = []
        # Houses and the prev-houses cache must be cleared before
        # _initialize_houses() rolls fresh fires --- otherwise RUINED houses
        # from a previous episode leak into the new one.
        # Issue #254: vector length scales with num_houses.
        self.houses = np.zeros(self.num_houses, dtype=np.int8)
        self._prev_houses_state = np.zeros(self.num_houses, dtype=np.int8)

        # Initialize house states
        self._initialize_houses()

        # Initialize agent positions and signals
        self.locations = np.zeros(self.num_agents, dtype=np.int8)
        self.signals = np.zeros(self.num_agents, dtype=np.int8)
        # Issue #235: keep last_actions width 2 in obs (see __init__ comment).
        self.last_actions = np.zeros((self.num_agents, 2), dtype=np.int8)

        # Record initial state
        self._record_night()

        return self._get_observation()

    def step(
        self, actions: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict]:
        """
        Execute one night of the game.

        Args:
            actions: Per-agent actions, shape (N, 3) with
                ``[house_index, mode_flag, signal]`` (issue #235). Length-2
                inputs are accepted for backward compatibility and treated
                as honest (``signal := mode_flag``).

        Returns:
            observation, rewards, dones, info
        """
        if self.done:
            raise RuntimeError("Game is already finished")

        # Accept legacy length-2 actions by promoting them to honest
        # length-3 actions (signal == mode_flag). This keeps any external
        # caller that still emits the pre-#235 shape working.
        if actions.shape[1] == 2:
            signal_col = actions[:, 1:2]
            actions = np.concatenate([actions, signal_col], axis=1)

        # 1. Signal phase (issue #235): signals are now their own action
        # dimension. Pre-#235 ``self.signals = actions[:, 1]`` made the
        # signal a deterministic copy of the work bit; now agents can
        # broadcast independently of their mode (e.g. lie).
        self.signals = actions[:, 2].copy()

        # 2. Action phase: update agent locations.
        # Issue #235: store only ``[house, mode]`` in last_actions to keep
        # the obs vector width unchanged (the broadcast signal is
        # already exposed via ``self.signals``).
        self.last_actions = actions[:, :2].copy()
        self.locations = actions[:, 0].copy()

        # 3. Extinguish phase
        # Agents respond to fires visible at start of turn
        self._extinguish_fires(actions)

        # 4. Burn-out phase
        # Unextinguished fires become ruined houses
        self._burn_out_houses()

        # 5. Spread phase
        # Fires spread to neighbors (visible next turn)
        self._spread_fires()

        # 6. Spontaneous ignition phase
        # New fires can ignite on any night (visible next turn)
        # This matches the Rust implementation and game design specification
        self._spark_fires()

        # 7. Compute rewards
        self.rewards = self._compute_rewards(actions)

        # 8. Check termination
        self.done = self._check_termination()

        # 9. Record this night
        self._record_night()

        # 10. Advance to next night
        self.night += 1

        return (
            self._get_observation(),
            self.rewards.copy(),
            np.full(self.num_agents, self.done),
            {},
        )

    def _initialize_houses(self) -> None:
        """Initialize houses with probabilistic fires based on prob_house_catches_fire."""
        # Each house has independent probability of starting on fire.
        # Issue #254: iterate over scenario.num_houses, not literal 10.
        for house_idx in range(self.num_houses):
            if self.rng.random() < self.scenario.prob_house_catches_fire:
                self.houses[house_idx] = self.BURNING

    def _extinguish_fires(self, actions: np.ndarray) -> None:
        """Extinguish fires based on worker presence using independent probabilities.

        Each worker has probability `kappa` of extinguishing the fire independently.
        Formula: P(extinguish with k workers) = 1 - (1 - kappa)^k
        This matches the Rust implementation and game design specification.
        """
        for house_idx in range(self.num_houses):
            if self.houses[house_idx] != self.BURNING:
                continue

            # Count workers at this house
            workers_here = np.sum(
                (actions[:, 0] == house_idx) & (actions[:, 1] == self.WORK)
            )

            # Probability of extinguishing: independent probabilities model
            # P(at least one success) = 1 - P(all fail) = 1 - (1-p)^k
            p_extinguish = (
                1.0
                - (1.0 - self.scenario.prob_solo_agent_extinguishes_fire)
                ** workers_here
            )

            if self.rng.random() < p_extinguish:
                self.houses[house_idx] = self.SAFE

    def _spread_fires(self) -> None:
        """Spread fires to neighboring safe houses.

        Issue #254: ring length is now `scenario.num_houses` rather than a
        hardcoded 10, and the neighbor wraparound modulo tracks the
        scenario value so 2-house and other small-ring topologies wrap
        correctly. Mirrors the Rust ``engine/phases.rs::spread_fires``
        behavior.
        """
        # Burning houses try to ignite their safe neighbors
        for house_idx in range(self.num_houses):
            if self.houses[house_idx] != self.BURNING:
                continue

            # Check both neighbors
            for neighbor_offset in [-1, 1]:
                neighbor_idx = (house_idx + neighbor_offset) % self.num_houses
                if self.houses[neighbor_idx] == self.SAFE:
                    if self.rng.random() < self.scenario.prob_fire_spreads_to_neighbor:
                        self.houses[neighbor_idx] = self.BURNING

    def _burn_out_houses(self) -> None:
        """Burning houses that neither extinguished nor spread become ruined."""
        # Any remaining burning houses become ruined
        burning_mask = self.houses == self.BURNING
        self.houses[burning_mask] = self.RUINED

    def _spark_fires(self) -> None:
        """Add spontaneous fires."""
        # Issue #254: iterate over scenario.num_houses, not literal 10.
        for house_idx in range(self.num_houses):
            if (
                self.houses[house_idx] == self.SAFE
                and self.rng.random() < self.scenario.prob_house_catches_fire
            ):
                self.houses[house_idx] = self.BURNING

    def _compute_rewards(self, actions: np.ndarray) -> np.ndarray:
        """Compute rewards for all agents."""
        # Store previous house states to compute changes
        prev_houses = (
            np.array(self._prev_houses_state)
            if hasattr(self, "_prev_houses_state")
            else self.houses.copy()
        )
        self._prev_houses_state = self.houses.copy()

        # Count final outcomes
        saved_houses = np.sum(self.houses == self.SAFE)
        ruined_houses = np.sum(self.houses == self.RUINED)
        # Issue #254: divide by scenario.num_houses (defaults to 10 for
        # every pre-#254 scenario, so the math is unchanged).
        num_houses_f = float(self.num_houses)
        total_saved_fraction = saved_houses / num_houses_f
        total_burned_fraction = ruined_houses / num_houses_f

        # Team reward component (shared by all)
        team_reward = (
            self.scenario.team_reward_house_survives * total_saved_fraction
            - self.scenario.team_penalty_house_burns * total_burned_fraction
        )

        # Individual rewards
        individual_rewards = np.zeros(self.num_agents, dtype=np.float32)

        # Issue #203: spatial cost coefficient. When zero (the implicit
        # default for every pre-#203 scenario) the work cost reduces to the
        # unscaled ``cost_to_work_one_night`` so behavior is bit-exactly
        # unchanged.
        alpha = float(getattr(self.scenario, "distance_cost_alpha", 0.0))

        for agent_idx in range(self.num_agents):
            # Work/rest component.
            #
            # Issue #203 (option A): when ``alpha != 0`` the work cost is
            # additively scaled by the ring-arc distance between the agent's
            # home position and the house it works at:
            #     cost = base_cost + alpha * ring_dist(home, target).
            if actions[agent_idx, 1] == self.WORK:
                base_cost = self.scenario.cost_to_work_one_night
                if alpha == 0.0:
                    work_cost = base_cost
                else:
                    home = int(self.agent_home_positions[agent_idx])
                    target = int(actions[agent_idx, 0])
                    raw = abs(home - target)
                    # Issue #254: ring length now reads from the scenario.
                    dist = min(raw, self.num_houses - raw)
                    work_cost = base_cost + alpha * dist
                individual_rewards[agent_idx] -= work_cost
            else:
                individual_rewards[agent_idx] += 0.5  # Rest reward

            # Per-house ownership rewards.
            # For each of the 10 houses, decide whether the agent owns it and
            # apply the appropriate per-house reward field. This wires up the
            # four previously-unused Scenario ownership reward fields
            # (`reward_own_house_survives`, `reward_other_house_survives`,
            # `penalty_own_house_burns`, `penalty_other_house_burns`).
            #
            # As of issue #198 these four fields are per-agent vectors of
            # length ``num_agents``; we index by ``agent_idx`` here. The
            # ``Scenario.__post_init__`` auto-promotes scalar JSON inputs
            # to ``[scalar] * num_agents`` so existing scenarios behave
            # identically.
            for house_idx in range(self.num_houses):
                is_own = self.house_owners[house_idx] == agent_idx

                # Save event: any non-SAFE state -> SAFE this step.
                if (
                    prev_houses[house_idx] != self.SAFE
                    and self.houses[house_idx] == self.SAFE
                ):
                    individual_rewards[agent_idx] += (
                        self.scenario.reward_own_house_survives[agent_idx]
                        if is_own
                        else self.scenario.reward_other_house_survives[agent_idx]
                    )

                # Currently-ruined penalty (applied every step the house is
                # RUINED). The penalty field stores the magnitude as a positive
                # number; subtract it.
                if self.houses[house_idx] == self.RUINED:
                    individual_rewards[agent_idx] -= (
                        self.scenario.penalty_own_house_burns[agent_idx]
                        if is_own
                        else self.scenario.penalty_other_house_burns[agent_idx]
                    )

            # Team reward component (full public goods incentive)
            individual_rewards[agent_idx] += team_reward

        return individual_rewards

    def _check_termination(self) -> bool:
        """Check if the game should end."""
        # Must play at least min_nights
        if self.night < self.scenario.min_nights:
            return False

        # End if all houses are safe or all are ruined
        all_safe = np.all(self.houses == self.SAFE)
        all_ruined = np.all(self.houses == self.RUINED)

        # Also end if no fires are burning (game is stable)
        no_fires = np.sum(self.houses == self.BURNING) == 0

        return bool(all_safe or all_ruined or no_fires)

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation for all agents."""
        return {
            "signals": self.signals.copy(),
            "locations": self.locations.copy(),
            "houses": self.houses.copy(),
            "last_actions": self.last_actions.copy(),
            "scenario_info": self.scenario.to_feature_vector(),
        }

    def _record_night(self) -> None:
        """Record current night state for replay."""
        night_data = {
            "night": self.night,
            "houses": self.houses.tolist(),
            "signals": self.signals.tolist(),
            "locations": self.locations.tolist(),
            "actions": self.last_actions.tolist(),
            "rewards": self.rewards.tolist(),
        }
        self.trajectory.append(night_data)

    def save_replay(self, path: str) -> None:
        """
        Save complete game trajectory as JSON.

        Args:
            path: File path to save replay
        """
        replay_data = {
            "scenario": {
                "prob_fire_spreads_to_neighbor": self.scenario.prob_fire_spreads_to_neighbor,
                "prob_solo_agent_extinguishes_fire": self.scenario.prob_solo_agent_extinguishes_fire,
                "prob_house_catches_fire": self.scenario.prob_house_catches_fire,
                "team_reward_house_survives": self.scenario.team_reward_house_survives,
                "team_penalty_house_burns": self.scenario.team_penalty_house_burns,
                "reward_own_house_survives": self.scenario.reward_own_house_survives,
                "reward_other_house_survives": self.scenario.reward_other_house_survives,
                "penalty_own_house_burns": self.scenario.penalty_own_house_burns,
                "penalty_other_house_burns": self.scenario.penalty_other_house_burns,
                "cost_to_work_one_night": self.scenario.cost_to_work_one_night,
                "min_nights": self.scenario.min_nights,
                "num_agents": self.scenario.num_agents,
                # Issue #254: include num_houses so replays of non-10-house
                # scenarios (e.g. v2_minimal) round-trip cleanly.
                "num_houses": int(getattr(self.scenario, "num_houses", 10)),
            },
            "nights": self.trajectory,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(replay_data, f, indent=2)

    def render(self) -> None:
        """Simple text-based rendering for debugging."""
        house_symbols = ["□", "🔥", "💀"]  # SAFE, BURNING, RUINED

        print(f"Night {self.night}:")
        print("Houses:", "".join(house_symbols[state] for state in self.houses))
        print("Signals:", "".join("R" if s == self.REST else "W" for s in self.signals))
        print("Locations:", self.locations)
        print("Rewards:", self.rewards)
        print()

    @property
    def observation_space(self) -> Any:
        """Gym/PufferLib compatible observation space."""
        # This would need pufferlib import, simplified for now
        return None

    @property
    def action_space(self) -> Any:
        """Gym/PufferLib compatible action space."""
        # This would need pufferlib import, simplified for now
        return None
