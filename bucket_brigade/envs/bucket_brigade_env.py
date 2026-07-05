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

        # Issue #253: per-house accumulated suppression progress for the
        # `"continuous"` extinguish mode. Each work step at a burning house
        # adds ``suppression_per_worker * workers_here`` to the per-house
        # accumulator; the fire transitions BURNING -> SAFE deterministically
        # when the accumulator reaches 1.0. Zeroed on ignition and burn-out
        # so per-fire progress does not leak across the same-house fire
        # cycle. In Bernoulli mode (the pre-#253 default) the vector is
        # allocated but never written, so memory cost is negligible and
        # behavior is bit-exactly identical to pre-#253.
        self.fire_progress: np.ndarray = np.zeros(self.num_houses, dtype=np.float32)

        # Issue #252: round-1 non-binding commitment signals from the
        # `"two_phase"` commitment mode. Length matches num_agents. Only
        # written by `step_two_phase`; the single-phase `step()` path
        # leaves this vector at its default (all-zeros) so the obs is
        # byte-identical to pre-#252 once the channel is excluded from
        # the flat obs vector. Exposed via `_get_observation` so the
        # round-2 policy forward can condition on round-1 signals.
        self.round1_signals: np.ndarray = np.zeros(self.num_agents, dtype=np.int8)

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
        # Issue #253: zero the suppression accumulator. Mirrors the Rust
        # `BucketBrigade::reset` behavior so cross-language parity holds.
        self.fire_progress = np.zeros(self.num_houses, dtype=np.float32)

        # Issue #252: zero the round-1 commitment-signal buffer at reset
        # so cross-episode signal state does not leak. In simultaneous
        # mode the buffer is never written; this is a no-op for those
        # scenarios.
        self.round1_signals = np.zeros(self.num_agents, dtype=np.int8)

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

        # Issue #252: refuse single-phase `step()` calls on a two-phase
        # scenario. The two-phase mechanic requires running the signal
        # round before the action round; calling `step()` directly would
        # silently skip the signal phase and leak round-1 signals across
        # nights. Callers should use `step_two_phase(round1_signals,
        # round2_actions)` instead. Mirrors the Rust defensive panic in
        # `BucketBrigade::step`.
        if getattr(self.scenario, "commitment_mode", "simultaneous") == "two_phase":
            raise RuntimeError(
                f"BucketBrigadeEnv.step() called on a two-phase scenario; "
                f"use step_two_phase(round1_signals, round2_actions) instead. "
                f"commitment_mode={self.scenario.commitment_mode!r}"
            )

        # Accept legacy length-2 actions by promoting them to honest
        # length-3 actions (signal == mode_flag). This keeps any external
        # caller that still emits the pre-#235 shape working.
        if actions.shape[1] == 2:
            signal_col = actions[:, 1:2]
            actions = np.concatenate([actions, signal_col], axis=1)

        # 0. Action-validity sanitization (issue #251). When
        # ``scenario.action_validity_mode == "adjacent_only"``, rewrite any
        # agent action whose target house is more than ring-distance 1 from
        # the agent's home position to that home position (a no-op move
        # into home). Mode bit and signal bit are preserved. When the mode
        # is the default ``"always_valid"`` this is a true bit-exact no-op:
        # the input array passes through untouched. Every downstream phase
        # (extinguish, rewards, observation) consumes the sanitized actions,
        # not the raw input, so policies can't cheat the constraint by
        # separately attacking different phases. Mirrors the canonical Rust
        # implementation in ``bucket-brigade-core/src/engine/core.rs``
        # (``sanitize_actions``).
        actions = self._sanitize_actions(actions)

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
        # NOTE (issue #458): in the default "bernoulli" extinguish mode,
        # phase 4 has already ruined every still-burning house, so this
        # phase is a structural no-op and prob_fire_spreads_to_neighbor
        # (beta) is dynamics-inert (never gates a spread, draws zero RNG).
        # Only "continuous" mode (#253) makes fire spread live. Mirrors
        # the Rust step order in engine/core.rs.
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

    def step_two_phase(
        self,
        round1_signals: np.ndarray,
        round2_actions: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict]:
        """Issue #252: two-phase non-binding signaling step (option C / C1
        from architect proposal #234).

        One call advances the night by one step, internally fusing the
        signal-phase write and the action-phase step. Required for
        scenarios with ``commitment_mode == "two_phase"``; the
        single-phase ``step()`` raises a clear error on those scenarios.

        Mechanic:

        1. **Round 1 (signal phase)**: round-1 signals are written into
           ``self.round1_signals``, which is exposed via the observation
           dict. The night does NOT advance, houses do NOT update,
           ``self.signals`` does NOT update (round 2 overwrites it
           below), no work cost is incurred, no reward is produced.
        2. **Round 2 (action phase)**: the engine runs the regular
           ``step()`` body on ``round2_actions``. The round-2 signal
           value (column 2 of the action) overwrites
           ``self.signals`` matching today's semantics so v1-trained
           policies still parse obs correctly. The round-1 signals stay
           in ``self.round1_signals`` until the next ``step_two_phase``
           overwrites them.

        **Deception channel survives**: round-2 mode (column 1 of the
        action) is not constrained by the round-1 signal at all.
        Policies can emit ``round1_signal=1 (Work)`` then ``round2_mode=0
        (Rest)``. The PR-gate test
        ``tests/test_environment.py::TestCommitmentMode::test_can_still_lie``
        exercises this directly with a hardcoded Liar policy.

        Mirrors the canonical Rust implementation in
        ``bucket-brigade-core/src/engine/core.rs::step_two_phase``.

        Args:
            round1_signals: Per-agent round-1 commitment signals, shape
                ``(num_agents,)`` or ``(num_agents, 1)``, dtype int.
                Values in ``{0, 1}``.
            round2_actions: Per-agent round-2 actions, shape
                ``(num_agents, 3)`` with ``[house, mode, signal]``.
                Length-2 actions are accepted for legacy callers and
                promoted with ``signal := mode``.

        Returns:
            ``(observation, rewards, dones, info)`` matching ``step()``.
        """
        if self.done:
            raise RuntimeError("Game is already finished")
        mode = getattr(self.scenario, "commitment_mode", "simultaneous")
        if mode != "two_phase":
            raise RuntimeError(
                f"BucketBrigadeEnv.step_two_phase requires "
                f"commitment_mode='two_phase'; got {mode!r}. "
                f"Use step() for simultaneous mode."
            )

        # Validate shapes.
        r1 = np.asarray(round1_signals).reshape(-1)
        if r1.shape[0] != self.num_agents:
            raise ValueError(
                f"round1_signals length {r1.shape[0]} != num_agents {self.num_agents}"
            )
        r2 = np.asarray(round2_actions)
        if r2.shape[0] != self.num_agents:
            raise ValueError(
                f"round2_actions length {r2.shape[0]} != num_agents {self.num_agents}"
            )

        # Round 1 (signal phase): write the round-1 signals into the obs
        # channel. No other state mutates.
        self.round1_signals = r1.astype(np.int8)

        # Round 2 (action phase): inline the body of `step()` rather than
        # delegating, because `step()` panics on two-phase scenarios as a
        # guardrail. The body below is byte-identical to `step()` apart
        # from that guardrail check.
        if r2.shape[1] == 2:
            signal_col = r2[:, 1:2]
            r2 = np.concatenate([r2, signal_col], axis=1)
        r2 = self._sanitize_actions(r2)
        self.signals = r2[:, 2].copy()
        self.last_actions = r2[:, :2].copy()
        self.locations = r2[:, 0].copy()
        self._extinguish_fires(r2)
        self._burn_out_houses()
        self._spread_fires()
        self._spark_fires()
        self.rewards = self._compute_rewards(r2)
        self.done = self._check_termination()
        self._record_night()
        self.night += 1

        return (
            self._get_observation(),
            self.rewards.copy(),
            np.full(self.num_agents, self.done),
            {},
        )

    def _sanitize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Apply the per-agent position-constrained action mask (issue #251).

        When ``scenario.action_validity_mode == "always_valid"`` (the default)
        this is a true bit-exact pass-through: the input array is returned
        unchanged and the engine sees pre-#251 behavior. When the mode is
        ``"adjacent_only"``, agent ``i`` may target only houses whose ring
        distance from ``agent_home_positions[i]`` is at most 1; out-of-reach
        targets are rewritten to the agent's home position. The mode bit
        (column 1) and signal bit (column 2) are preserved — only the house
        index in column 0 is potentially rewritten.

        Mirrors the canonical Rust implementation in
        ``bucket-brigade-core/src/engine/core.rs::sanitize_actions``.
        """
        mode = getattr(self.scenario, "action_validity_mode", "always_valid")
        if mode == "always_valid":
            return actions
        # ``adjacent_only``: rewrite out-of-reach house indices to home.
        # Ring distance on a length-N ring: min(|a-b|, N - |a-b|).
        n = self.num_houses
        homes = self.agent_home_positions  # length-num_agents, dtype int8
        sanitized = actions.copy()
        targets = sanitized[:, 0].astype(np.int32)
        homes_i = homes.astype(np.int32)
        raw = np.abs(targets - homes_i)
        dist = np.minimum(raw, n - raw)
        out_of_reach = dist > 1
        # Cast back to the action array dtype to avoid widening.
        sanitized[out_of_reach, 0] = homes[out_of_reach].astype(sanitized.dtype)
        return sanitized

    def _initialize_houses(self) -> None:
        """Initialize houses with probabilistic fires based on prob_house_catches_fire."""
        # Each house has independent probability of starting on fire.
        # Issue #254: iterate over scenario.num_houses, not literal 10.
        for house_idx in range(self.num_houses):
            if self.rng.random() < self.scenario.prob_house_catches_fire:
                self.houses[house_idx] = self.BURNING

    def _extinguish_fires(self, actions: np.ndarray) -> None:
        """Extinguish fires based on worker presence.

        Dispatches on ``scenario.extinguish_mode`` (issue #253):

        - ``"bernoulli"`` (default): pre-#253 model. Each worker has
          probability ``kappa`` of extinguishing the fire independently;
          ``P(extinguish with k workers) = 1 - (1 - kappa)^k``. Single
          coin flip per step per burning house. Bit-exactly identical to
          the pre-#253 behavior on every existing scenario.

        - ``"continuous"``: damage-accumulation model. Each work step at
          a burning house adds ``workers_here * suppression_per_worker``
          to the per-house ``fire_progress`` accumulator; the fire
          transitions BURNING -> SAFE deterministically when the
          accumulator reaches 1.0 (and the accumulator is zeroed for
          the next ignition cycle). No RNG draws — the continuous
          dispatch is deterministic conditional on actions.

        Mirrors the Rust ``engine/phases.rs::extinguish_fires`` dispatch.
        """
        mode = getattr(self.scenario, "extinguish_mode", "bernoulli")
        if mode == "bernoulli":
            self._extinguish_fires_bernoulli(actions)
        elif mode == "continuous":
            self._extinguish_fires_continuous(actions)
        else:
            # Mirrors the Rust defensive panic — ``Scenario.__post_init__``
            # already rejects unknown modes, so this branch is dead in
            # normal use but surfaces a clear error if a future mode is
            # added without env wiring.
            raise ValueError(
                f"Unknown extinguish_mode={mode!r}; supported: "
                f"('bernoulli', 'continuous')."
            )

    def _extinguish_fires_bernoulli(self, actions: np.ndarray) -> None:
        """Pre-#253 Bernoulli extinguish model. Kept verbatim from the
        original ``_extinguish_fires`` so default-mode scenarios produce
        bit-exact identical behavior to the pre-#253 env.
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

    def _extinguish_fires_continuous(self, actions: np.ndarray) -> None:
        """Issue #253 continuous extinguish model. See class-level docstring
        on ``_extinguish_fires`` for the calibration narrative."""
        suppression_per_worker = float(
            getattr(self.scenario, "suppression_per_worker", 0.0)
        )
        for house_idx in range(self.num_houses):
            if self.houses[house_idx] != self.BURNING:
                continue
            workers_here = int(
                np.sum((actions[:, 0] == house_idx) & (actions[:, 1] == self.WORK))
            )
            if workers_here == 0:
                continue
            self.fire_progress[house_idx] += workers_here * suppression_per_worker
            if self.fire_progress[house_idx] >= 1.0:
                self.houses[house_idx] = self.SAFE
                self.fire_progress[house_idx] = 0.0

    def _spread_fires(self) -> None:
        """Spread fires to neighboring safe houses.

        Issue #254: ring length is now `scenario.num_houses` rather than a
        hardcoded 10, and the neighbor wraparound modulo tracks the
        scenario value so 2-house and other small-ring topologies wrap
        correctly. Mirrors the Rust ``engine/phases.rs::spread_fires``
        behavior.

        Beta-inertness (issue #458): in the default ``"bernoulli"``
        extinguish mode this phase is a structural no-op —
        ``_burn_out_houses`` runs first and ruins every still-BURNING
        house, so the BURNING guard below skips every house and
        ``prob_fire_spreads_to_neighbor`` never gates a spread (zero RNG
        draws; cross-beta trajectories are bit-identical under a shared
        seed, pinned by ``tests/test_beta_inertness.py``). Fire spread is
        only live in ``"continuous"`` extinguish mode (#253). Do NOT
        remove beta as dead code: it reaches agents as
        ``scenario_info[0]`` in observations.
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
        """Burning houses that neither extinguished nor spread become ruined.

        Issue #253: in continuous mode fires do NOT auto-burn-out — they
        persist across steps so the suppression accumulator can integrate
        credit over multiple work steps. Mirrors the Rust
        ``engine/phases.rs::burn_out_houses`` dispatch.
        """
        mode = getattr(self.scenario, "extinguish_mode", "bernoulli")
        if mode == "continuous":
            return
        # Bernoulli mode: pre-#253 behavior unchanged.
        burning_mask = self.houses == self.BURNING
        self.houses[burning_mask] = self.RUINED
        # Defensive: zero the accumulator for the just-ruined houses.
        # In Bernoulli mode the accumulator was never written so this is
        # a no-op, but keeps the invariant
        # "fire_progress[i] != 0 implies houses[i] == BURNING" exact.
        self.fire_progress[burning_mask] = 0.0

    def _spark_fires(self) -> None:
        """Add spontaneous fires."""
        # Issue #254: iterate over scenario.num_houses, not literal 10.
        for house_idx in range(self.num_houses):
            if (
                self.houses[house_idx] == self.SAFE
                and self.rng.random() < self.scenario.prob_house_catches_fire
            ):
                self.houses[house_idx] = self.BURNING
                # Issue #253: a fresh ignition gets a fresh accumulator.
                # Mirrors the Rust ``engine/phases.rs::spontaneous_ignition``
                # path so cross-language parity holds.
                self.fire_progress[house_idx] = 0.0

    def _compute_team_welfare_phi(self, houses: np.ndarray) -> float:
        """Issue #283: closed-form team-welfare potential Phi(s).

        Option B from the issue: a cheap, debuggable proxy for team value
        that's tied to the scenario's team-reward weights so the shaping
        signal stays on the same scale as the base team reward.

            Phi(s) = team_reward * (safe/N)
                     - team_penalty * (ruined/N)
                     - 0.5 * team_penalty * (burning/N)

        The 0.5x weight on burning houses captures "anticipated burnout
        cost" — a burning house is partway between SAFE and RUINED in
        expectation but is still recoverable, so we charge half the
        ruined penalty.

        IMPORTANT: This is a pure function of ``houses`` (the state-only
        signature is what makes the NHR telescoping identity hold). Do
        not let any policy-conditioned features leak in. Mirrors
        ``bucket-brigade-core/src/engine/rewards.rs::team_welfare_phi``.
        """
        kind = getattr(self.scenario, "team_welfare_kind", "none")
        if kind == "none":
            return 0.0
        if kind == "team_welfare_closed_form":
            num_houses_f = float(self.num_houses)
            num_safe = float(np.sum(houses == self.SAFE))
            num_ruined = float(np.sum(houses == self.RUINED))
            num_burning = float(np.sum(houses == self.BURNING))
            w_safe = float(self.scenario.team_reward_house_survives)
            w_penalty = float(self.scenario.team_penalty_house_burns)
            return (
                w_safe * (num_safe / num_houses_f)
                - w_penalty * (num_ruined / num_houses_f)
                - 0.5 * w_penalty * (num_burning / num_houses_f)
            )
        # Defensive — ``__post_init__`` already rejects unknown kinds, but
        # keep this branch so future kinds added without env support
        # surface a clear error rather than silently returning 0.
        raise ValueError(
            f"Unknown team_welfare_kind={kind!r}; supported: "
            f"('none', 'team_welfare_closed_form')."
        )

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

        # Issue #265: dense progress shaping. Per-step team-shared bonus
        # proportional to the change in the number of SAFE houses since
        # the previous step. ``progress_shaping_coef`` defaults to 0.0 on
        # every pre-#265 scenario, so this block is a no-op (preserving
        # bit-exact behavior). When non-zero, the team component picks up
        # ``coef * (cur_safe - prev_safe)`` and is broadcast to all agents
        # via the per-agent ``individual_rewards[i] += team_reward`` line
        # below. Mirrors ``bucket-brigade-core/src/engine/rewards.rs``.
        progress_coef = float(getattr(self.scenario, "progress_shaping_coef", 0.0))
        if progress_coef != 0.0:
            prev_safe = int(np.sum(prev_houses == self.SAFE))
            cur_safe = int(saved_houses)
            team_reward += progress_coef * float(cur_safe - prev_safe)

        # Individual rewards
        individual_rewards = np.zeros(self.num_agents, dtype=np.float32)

        # Issue #203: spatial cost coefficient. When zero (the implicit
        # default for every pre-#203 scenario) the work cost reduces to the
        # unscaled ``cost_to_work_one_night`` so behavior is bit-exactly
        # unchanged.
        alpha = float(getattr(self.scenario, "distance_cost_alpha", 0.0))

        # Issue #447: the flat per-step rest reward is a scenario weight
        # (``Scenario.reward_rest``), no longer a hardcoded ``+0.5``. The
        # default of 0.5 matches the historical constant, so every pre-#447
        # scenario is bit-exactly unchanged. Mirrors
        # ``bucket-brigade-core/src/engine/rewards.rs``.
        rest_reward = float(getattr(self.scenario, "reward_rest", 0.5))

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
                individual_rewards[agent_idx] += rest_reward  # Rest reward (#447)

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

        # Action-conditioned reward shaping (issue #259).
        #
        # When both alpha and beta are zero (the default for every pre-#259
        # scenario) this loop is skipped entirely so per-step rewards are
        # byte-identical to pre-#259 behavior. Mirrors the Rust
        # implementation in ``bucket-brigade-core/src/engine/rewards.rs``.
        #
        # alpha: credit-shared bonus for each worker that participated in
        # extinguishing a fire (BURNING -> SAFE transition this step). Each
        # worker at house ``h`` receives ``alpha / workers_at_h``. Note
        # this uses the *strict* BURNING(1) -> SAFE(0) check, which is
        # stricter than the save-event detection in the per-house ownership
        # loop above (that loop fires on any non-SAFE -> SAFE).
        #
        # beta: flat bonus for each agent working at a house that was SAFE
        # at start-of-step AND still SAFE at end-of-step (preventive
        # presence). Not credit-shared.
        #
        # REST actions (action[1] == 0) never receive either bonus.
        action_alpha = float(getattr(self.scenario, "action_shaping_alpha", 0.0))
        action_beta = float(getattr(self.scenario, "action_shaping_beta", 0.0))
        if action_alpha != 0.0 or action_beta != 0.0:
            # Pre-count workers per house (mirrors the vectorized count in
            # `_extinguish_fires`).
            workers_per_house = np.zeros(self.num_houses, dtype=np.int32)
            work_mask = actions[:, 1] == self.WORK
            for agent_idx in np.nonzero(work_mask)[0]:
                h = int(actions[agent_idx, 0])
                if 0 <= h < self.num_houses:
                    workers_per_house[h] += 1

            for agent_idx in range(self.num_agents):
                if actions[agent_idx, 1] != self.WORK:
                    continue  # REST receives no shaping bonus.
                h = int(actions[agent_idx, 0])
                if not (0 <= h < self.num_houses):
                    continue
                workers_h = int(workers_per_house[h])
                if workers_h == 0:
                    continue  # Defensive; should be >= 1 (this agent).

                # alpha: extinguish credit share. Strict BURNING -> SAFE.
                if (
                    action_alpha != 0.0
                    and prev_houses[h] == self.BURNING
                    and self.houses[h] == self.SAFE
                ):
                    individual_rewards[agent_idx] += action_alpha / float(workers_h)

                # beta: preventive presence. SAFE start AND SAFE end.
                if (
                    action_beta != 0.0
                    and prev_houses[h] == self.SAFE
                    and self.houses[h] == self.SAFE
                ):
                    individual_rewards[agent_idx] += action_beta

        # Potential-based team-welfare shaping (issue #283, NHR 1999).
        #
        # F(s, a, s') = lambda * (gamma * Phi(s') - Phi(s))
        #
        # Added to every agent's reward (team-shared bonus). At a terminal
        # transition Phi(s') is forced to 0 so the telescoping sum collapses
        # to a policy-independent constant `gamma^T * Phi(s_T) - Phi(s_0) =
        # -Phi(s_0)`, exactly preserving the NHR invariance theorem.
        #
        # When ``team_welfare_lambda == 0.0`` (the default for every pre-#283
        # scenario) this block is skipped entirely so per-step rewards are
        # byte-identical to pre-#283 behavior. Mirrors the Rust
        # implementation in ``bucket-brigade-core/src/engine/rewards.rs``.
        team_welfare_lambda = float(getattr(self.scenario, "team_welfare_lambda", 0.0))
        team_welfare_kind = getattr(self.scenario, "team_welfare_kind", "none")
        if team_welfare_lambda != 0.0 and team_welfare_kind != "none":
            gamma = float(getattr(self.scenario, "team_welfare_gamma", 1.0))
            phi_prev = self._compute_team_welfare_phi(prev_houses)
            # Determine whether s' is a terminal state. ``_check_termination``
            # is a pure function of ``self.houses`` and ``self.night`` (which
            # has not yet been advanced — see ``step()`` step order), so
            # calling it here is safe and idempotent.
            #
            # Terminal-state convention (NHR): Phi(terminal) := 0 so the
            # telescoping sum identity holds exactly. Without this, the
            # final-step bonus would leak a Phi(s_T) term that depends on
            # the policy's terminal state — breaking invariance.
            is_terminal_next = self._check_termination()
            phi_next = (
                0.0 if is_terminal_next else self._compute_team_welfare_phi(self.houses)
            )
            shaping_term = team_welfare_lambda * (gamma * phi_next - phi_prev)
            # Apply to every agent (team-shared bonus). Float32 to match
            # the rest of the reward vector dtype.
            individual_rewards += np.float32(shaping_term)

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
        """Get current observation for all agents.

        Issue #252: the ``round1_signals`` channel is always present
        but in simultaneous mode it is all-zeros (the signal phase is
        never run). Downstream consumers should ignore the field on
        simultaneous scenarios — the trainer's flat-obs builder
        (``flatten_dict_obs``) omits it unconditionally so the obs
        vector width is bit-exact for pre-#252 callers.
        """
        return {
            "signals": self.signals.copy(),
            "locations": self.locations.copy(),
            "houses": self.houses.copy(),
            "last_actions": self.last_actions.copy(),
            "scenario_info": self.scenario.to_feature_vector(),
            # Issue #252: round-1 non-binding commitment signals exposed
            # to the round-2 policy forward. Zeros in simultaneous mode.
            "round1_signals": self.round1_signals.copy(),
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
        """Gym-compatible observation space (placeholder)."""
        return None

    @property
    def action_space(self) -> Any:
        """Gym-compatible action space (placeholder)."""
        return None
