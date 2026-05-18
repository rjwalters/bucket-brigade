"""Macro-action wrapper around :class:`BucketBrigadeEnv` (issue #286).

The PPO basin-trap reading from #270/#271/#272 motivates coarsening the
decision horizon: instead of choosing a primitive ``[house, mode, signal]``
each base step, the agent commits to a multi-step **option** for
``commit_steps`` base steps (or until the env terminates). Macro-actions
shrink the effective horizon ``T_eff = T / N`` and change the search
neighborhood — a single macro-step is N primitive steps, potentially
crossing basin boundaries that primitive-action gradients can't traverse.

This is Sutton's options framework specialized to the credit-assignment
failure described in
``research_notebook/2026-05-17_thesis_misaligned_gradients.md`` (item 4).

Wrapper placement
-----------------

``BucketBrigadeEnv`` is *not* a Gym/Gymnasium/PettingZoo env at the base
layer — it exposes a custom dict-obs / numpy-action contract. There is a
Gymnasium-style wrapper at :mod:`bucket_brigade.envs.puffer_env` for
single-agent PufferLib training, but the joint multi-agent trainer
:class:`bucket_brigade.training.joint_trainer.JointPPOTrainer` drives
``BucketBrigadeEnv`` directly. The macro-action wrapper therefore sits at
the ``BucketBrigadeEnv`` level (wrap, don't subclass), exposing the same
``reset()`` and ``step(actions)`` contract so it is a drop-in replacement
in ``env_fn``.

Action space
------------

Each agent emits a single discrete macro-action index per macro-step. With
``num_agents=N`` the option set is::

    0: PATROL           — rotate around the ring; WORK at fires.
    1: DEFEND_OWN       — park at agent_home_positions[i]; WORK if needed.
    2: REST_UNTIL_FIRE  — REST in place; switch to PATROL on first fire.
    3..3+(N-2): FOLLOW_j — copy agent j's previous primitive action.

There are ``num_options = 3 + (num_agents - 1)`` options total (self is
excluded from FOLLOW). ``action_dims=[num_options]`` is what the trainer
should pass to ``JointPPOTrainer``.

Reward accumulation
-------------------

The undiscounted sum of per-agent rewards across the committed window is
returned as the macro-step reward. Discounted accumulation is available
via ``discount_gamma`` (default ``None`` = undiscounted, per the issue
spec).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .bucket_brigade_env import BucketBrigadeEnv


# Option indices. Keep these in sync with ``MacroActionEnv.option_index_*``
# helpers below.
OPT_PATROL = 0
OPT_DEFEND_OWN = 1
OPT_REST_UNTIL_FIRE = 2
# FOLLOW options start at index 3 and bundle the (num_agents - 1) other
# agents in ascending order (self is skipped).
OPT_FOLLOW_BASE = 3


class MacroActionEnv:
    """Wrap :class:`BucketBrigadeEnv` with a discrete macro-action space.

    Args:
        base_env: An instantiated :class:`BucketBrigadeEnv`.
        commit_steps: How many base-env steps each macro-action commits to
            (subject to early termination on env-done).
        discount_gamma: If ``None`` (default), reward accumulation is
            undiscounted. If a float in ``(0, 1]``, the reward returned at
            macro-step boundary k is ``sum_{t=0..n-1} gamma^t * r_t``
            where ``n`` is the actual number of base steps executed
            (``<= commit_steps``).

    Attributes mirror :class:`BucketBrigadeEnv` where useful:

    - ``num_agents``: forwarded.
    - ``num_houses``: forwarded.
    - ``num_options``: ``3 + (num_agents - 1)``.
    - ``done``: forwarded from the base env (so existing callers that
      poll ``trainer.env.done`` keep working).
    """

    # Re-export base-env constants for callers that want them off the wrapper.
    SAFE = BucketBrigadeEnv.SAFE
    BURNING = BucketBrigadeEnv.BURNING
    RUINED = BucketBrigadeEnv.RUINED
    REST = BucketBrigadeEnv.REST
    WORK = BucketBrigadeEnv.WORK

    def __init__(
        self,
        base_env: BucketBrigadeEnv,
        commit_steps: int = 5,
        discount_gamma: Optional[float] = None,
    ) -> None:
        if commit_steps < 1:
            raise ValueError(
                f"commit_steps must be >= 1; got {commit_steps}"
            )
        if discount_gamma is not None and not (0.0 < discount_gamma <= 1.0):
            raise ValueError(
                f"discount_gamma must be None or in (0, 1]; got {discount_gamma}"
            )

        self.base_env = base_env
        self.commit_steps = int(commit_steps)
        self.discount_gamma = discount_gamma

        self.num_agents = base_env.num_agents
        self.num_houses = base_env.num_houses
        # 3 fixed options (PATROL, DEFEND_OWN, REST_UNTIL_FIRE) + (N - 1)
        # FOLLOW_j options bundled in ascending j-order (skipping self at
        # action time). When num_agents == 1, there are no FOLLOW options.
        # Issue #286 spec: ``num_options = 3 + (num_agents - 1)``.
        self.num_options = 3 + max(0, self.num_agents - 1)

        # Per-option-execution state: tracks whether REST_UNTIL_FIRE has
        # seen a fire yet within the *current* commit window. Indexed by
        # agent. Reset at the start of every macro-step.
        self._fire_seen: np.ndarray = np.zeros(self.num_agents, dtype=bool)

    # ------------------------------------------------------------------
    # Public API mirroring BucketBrigadeEnv
    # ------------------------------------------------------------------

    @property
    def done(self) -> bool:
        return bool(self.base_env.done)

    @property
    def scenario(self):
        return self.base_env.scenario

    @property
    def night(self) -> int:
        return int(self.base_env.night)

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Reset the wrapped env. Returns the base env's obs dict unchanged."""
        obs = self.base_env.reset(seed=seed)
        self._fire_seen = np.zeros(self.num_agents, dtype=bool)
        return obs

    def step(
        self, macro_actions: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict]:
        """Execute one macro-step.

        Args:
            macro_actions: Integer array of shape ``(num_agents,)`` or
                ``(num_agents, 1)``. Each entry is an option index in
                ``[0, num_options)``. The trainer passes ``[N, A]`` with
                ``A=1`` (single discrete head); shape ``(N,)`` is also
                accepted for direct callers.

        Returns:
            ``(obs, rewards, dones, info)`` matching the base env contract.
            ``rewards`` is the undiscounted (or discounted) sum across the
            commit window. ``info["base_steps"]`` reports the actual
            number of base steps executed within the macro-step (``<=
            commit_steps``); ``info["primitive_actions"]`` is a list of
            the ``[N, 3]`` primitive arrays emitted (for tests/debug).
        """
        macro_actions = np.asarray(macro_actions)
        if macro_actions.ndim == 2 and macro_actions.shape[1] == 1:
            macro_actions = macro_actions[:, 0]
        if macro_actions.ndim != 1 or macro_actions.shape[0] != self.num_agents:
            raise ValueError(
                f"macro_actions must have shape ({self.num_agents},) or "
                f"({self.num_agents}, 1); got {macro_actions.shape}"
            )
        # Validate option indices.
        bad = (macro_actions < 0) | (macro_actions >= self.num_options)
        if bad.any():
            raise ValueError(
                f"macro_actions contains out-of-range option indices "
                f"(num_options={self.num_options}): {macro_actions.tolist()}"
            )
        macro_actions = macro_actions.astype(np.int64)

        # Reset per-window REST_UNTIL_FIRE state for agents holding that option.
        # (Agents that are not running REST_UNTIL_FIRE this window will not
        # consult ``_fire_seen``; resetting for all is cheap and harmless.)
        self._fire_seen = np.zeros(self.num_agents, dtype=bool)

        # Reward accumulator across the commit window.
        accumulated_rewards = np.zeros(self.num_agents, dtype=np.float32)
        primitive_log: List[np.ndarray] = []
        base_steps_done = 0
        last_obs: Optional[Dict[str, np.ndarray]] = None
        last_dones: Optional[np.ndarray] = None
        last_info: Dict = {}

        for k in range(self.commit_steps):
            # Update REST_UNTIL_FIRE bookkeeping *before* building the
            # primitive for this step: if any house is BURNING right now,
            # REST_UNTIL_FIRE agents should already start PATROL this step.
            # Also keep ``_fire_seen`` sticky once flipped (so transient
            # burn-out -> RUINED transitions still leave the seen flag set
            # for the rest of the window).
            if np.any(self.base_env.houses == BucketBrigadeEnv.BURNING):
                self._fire_seen[:] = True

            # Build primitive [N, 3] action from current obs + option indices.
            primitive = self._build_primitive_actions(macro_actions)
            primitive_log.append(primitive.copy())

            obs, rewards, dones, info = self.base_env.step(primitive)

            # Accumulate reward (undiscounted by default; discounted optional).
            if self.discount_gamma is None:
                accumulated_rewards += rewards.astype(np.float32)
            else:
                accumulated_rewards += (self.discount_gamma ** k) * rewards.astype(
                    np.float32
                )

            base_steps_done += 1
            last_obs = obs
            last_dones = dones
            last_info = info

            # Also re-check post-step: a fire that ignited via spread or
            # spontaneous ignition is visible on the next iteration's obs.
            if np.any(obs["houses"] == BucketBrigadeEnv.BURNING):
                self._fire_seen[:] = True

            # Early termination: if the base env is done, stop the commit
            # window here. Remaining option steps are dropped.
            if bool(np.asarray(dones).any()):
                break

        # ``last_obs`` is guaranteed set because commit_steps >= 1.
        assert last_obs is not None and last_dones is not None
        info_out = dict(last_info)
        info_out["base_steps"] = base_steps_done
        info_out["primitive_actions"] = primitive_log
        return last_obs, accumulated_rewards, last_dones, info_out

    # ------------------------------------------------------------------
    # Option -> primitive action translation
    # ------------------------------------------------------------------

    def _build_primitive_actions(self, macro_actions: np.ndarray) -> np.ndarray:
        """Translate per-agent option indices into a ``[N, 3]`` primitive."""
        # ``last_actions`` is [N, 2]; ``locations`` is [N]; ``houses`` is [H].
        last_actions = self.base_env.last_actions  # [N, 2]
        locations = self.base_env.locations  # [N]
        houses = self.base_env.houses  # [H]
        home_positions = self.base_env.agent_home_positions  # [N]

        primitive = np.zeros((self.num_agents, 3), dtype=np.int64)

        for i in range(self.num_agents):
            opt = int(macro_actions[i])
            if opt == OPT_PATROL:
                target = (int(locations[i]) + 1) % self.num_houses
                mode = (
                    BucketBrigadeEnv.WORK
                    if houses[target] == BucketBrigadeEnv.BURNING
                    else BucketBrigadeEnv.REST
                )
                primitive[i, 0] = target
                primitive[i, 1] = mode
                primitive[i, 2] = mode  # honest signal mirrors mode bit.
            elif opt == OPT_DEFEND_OWN:
                target = int(home_positions[i])
                # WORK if home or either neighbor is burning, else REST.
                left = (target - 1) % self.num_houses
                right = (target + 1) % self.num_houses
                burning_nearby = (
                    houses[target] == BucketBrigadeEnv.BURNING
                    or houses[left] == BucketBrigadeEnv.BURNING
                    or houses[right] == BucketBrigadeEnv.BURNING
                )
                mode = BucketBrigadeEnv.WORK if burning_nearby else BucketBrigadeEnv.REST
                primitive[i, 0] = target
                primitive[i, 1] = mode
                primitive[i, 2] = mode
            elif opt == OPT_REST_UNTIL_FIRE:
                if not self._fire_seen[i]:
                    # No fire observed yet inside this commit window: REST
                    # at current location.
                    primitive[i, 0] = int(locations[i])
                    primitive[i, 1] = BucketBrigadeEnv.REST
                    primitive[i, 2] = BucketBrigadeEnv.REST
                else:
                    # Transitioned to PATROL after first fire was seen.
                    target = (int(locations[i]) + 1) % self.num_houses
                    mode = (
                        BucketBrigadeEnv.WORK
                        if houses[target] == BucketBrigadeEnv.BURNING
                        else BucketBrigadeEnv.REST
                    )
                    primitive[i, 0] = target
                    primitive[i, 1] = mode
                    primitive[i, 2] = mode
            else:
                # FOLLOW_j options: bundled in ascending j-order, skipping
                # self. opt - OPT_FOLLOW_BASE indexes into the (num_agents - 1)
                # "other agents" list for agent i.
                j = self._follow_target(agent_i=i, opt=opt)
                # Copy j's previous primitive [house, mode]; honest signal.
                target = int(last_actions[j, 0])
                mode = int(last_actions[j, 1])
                # Defensive clamps: ``last_actions`` may have out-of-range
                # entries at the very first step (zeros), which are valid
                # primitive actions; nothing to do.
                primitive[i, 0] = target
                primitive[i, 1] = mode
                primitive[i, 2] = mode

        return primitive

    # ------------------------------------------------------------------
    # FOLLOW bundle index <-> target agent
    # ------------------------------------------------------------------

    def _follow_target(self, agent_i: int, opt: int) -> int:
        """Resolve a FOLLOW option index to a target agent j (j != i).

        FOLLOW options 3..3+(N-2) bundle the (N-1) other agents in ascending
        agent-id order, skipping ``agent_i``.

        Example with N=4, agent_i=2:
          opt=3 -> j=0, opt=4 -> j=1, opt=5 -> j=3
        """
        bundle_idx = opt - OPT_FOLLOW_BASE
        if bundle_idx < 0 or bundle_idx >= self.num_agents - 1:
            raise ValueError(
                f"FOLLOW bundle index {bundle_idx} out of range for "
                f"num_agents={self.num_agents}"
            )
        # The bundle skips ``agent_i``. So bundle_idx maps to the
        # bundle_idx-th element of ``[0..N-1] \ {agent_i}``.
        j = bundle_idx if bundle_idx < agent_i else bundle_idx + 1
        return int(j)

    # ------------------------------------------------------------------
    # Pass-throughs (for callers that probe the env directly)
    # ------------------------------------------------------------------

    @property
    def houses(self) -> np.ndarray:
        return self.base_env.houses

    @property
    def locations(self) -> np.ndarray:
        return self.base_env.locations

    @property
    def signals(self) -> np.ndarray:
        return self.base_env.signals

    @property
    def last_actions(self) -> np.ndarray:
        return self.base_env.last_actions

    @property
    def agent_home_positions(self) -> np.ndarray:
        return self.base_env.agent_home_positions

    def render(self) -> None:
        self.base_env.render()


__all__ = [
    "MacroActionEnv",
    "OPT_PATROL",
    "OPT_DEFEND_OWN",
    "OPT_REST_UNTIL_FIRE",
    "OPT_FOLLOW_BASE",
]
