"""Live observation audit for the joint multi-agent PPO trainer (issue #274).

This module provides a *passive* audit harness that records per-(step, agent)
samples of the exact tensor the policy consumed at training time, alongside:

- the env-side dict that the flattener was built from,
- the raw multi-discrete action the policy emitted,
- the sanitized action the env actually applied (post-:func:`sanitize_actions`,
  PR #316),
- the per-agent instantaneous reward,
- the per-agent identity one-hot tail recovered from ``flat_obs``,
- scenario info needed to interpret the sample (``action_validity_mode``,
  ``macro_actions``, ``extinguish_mode``).

The auditor is **observe-only**: when wired into
:meth:`JointPPOTrainer.collect_rollout` it must not alter PPO behavior in
any way. The bit-identity test (``tests/test_obs_audit.py``) pins that
property against a fixed seed.

Scope (curator-set v1):

- Output is JSON-Lines, one record per (step, agent). Numpy arrays are
  serialized as lists; dtype tags accompany every array.
- Reservoir sampling picks ``N`` records uniformly over the ``total_steps``
  the trainer will visit during the whole run, *not* per-rollout: it would
  bias toward the start of training otherwise. The picked global step
  indices are computed up-front from ``(num_samples, total_steps, seed)``
  so the audit is deterministic.
- ``N = 0`` is a no-op (no overhead, no file written).
- ``N >= total_steps`` records every step (uniform reservoir collapses).

Out of scope for v1:

- Post-hoc analysis / Markdown verdict report (follow-up issue).
- Vectorized puffer rollout path (single-agent, different obs flow).
- Auditing macro-action expansion inside :class:`MacroActionEnv`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


__all__ = ["ObsAuditSample", "ObsAuditor"]


def _to_listish(arr) -> List:
    """Convert a numpy array / scalar / list into a JSON-serializable list."""
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    if isinstance(arr, (np.integer, np.floating)):
        return arr.item()
    if isinstance(arr, (list, tuple)):
        return [_to_listish(x) for x in arr]
    if isinstance(arr, dict):
        return {k: _to_listish(v) for k, v in arr.items()}
    return arr


@dataclass
class ObsAuditSample:
    """One audit record: what agent ``agent_id`` saw / did at global step ``t``.

    Attributes:
        t: Global timestep (rollout step counter, not iteration). Recorded
            as the trainer's monotonically-increasing step index.
        agent_id: Per-agent index (0..num_agents-1).
        flat_obs: ``[obs_dim]`` float32 — exact tensor fed to the policy.
        env_state: Dict snapshot of the pre-flatten env observation for the
            *same* step. Keys: ``houses``, ``signals``, ``locations``,
            ``last_actions``, ``scenario_info`` (mirrors :class:`AgentObservation`
            in ``bucket-brigade-core``).
        raw_action: Length-``len(action_dims)`` int64 — the action the policy
            emitted. For multi-discrete this is ``[house, mode, signal]``.
            For macro-actions this is a single option index.
        action_taken: Length-2 int64 — the action the env actually applied
            after :func:`sanitize_actions`, read back from
            ``env.last_actions[agent_id]`` *after* :meth:`step`. Width 2
            (``[house, mode]``) because the env strips the signal dim from
            ``last_actions``. For pre-#316 scenarios
            (``action_validity_mode="always_valid"``) this is bit-equal to
            ``raw_action[:2]``.
        reward: Per-agent instantaneous reward at step ``t``.
        identity_tail: Length-``num_agents`` float32 — the trailing one-hot
            slice ``flat_obs[-num_agents:]``. Extracted for the post-hoc
            identity check; bit-equal to a slice of ``flat_obs`` but
            captured separately so the analyzer can verify the slice
            convention without re-reading the full obs.
        scenario_info: Free-form dict capturing scenario knobs the analyzer
            needs to interpret the sample. Required keys at v1:
            ``action_validity_mode``, ``macro_actions``, ``extinguish_mode``
            (the latter two may be ``None`` if not applicable).
    """

    t: int
    agent_id: int
    flat_obs: np.ndarray
    env_state: Dict
    raw_action: np.ndarray
    action_taken: np.ndarray
    reward: float
    identity_tail: np.ndarray
    scenario_info: Dict = field(default_factory=dict)

    def to_json_dict(self) -> Dict:
        """Convert to a JSON-serializable dict (lists, no numpy)."""
        return {
            "t": int(self.t),
            "agent_id": int(self.agent_id),
            "flat_obs": self.flat_obs.astype(np.float32).tolist(),
            "env_state": {k: _to_listish(v) for k, v in self.env_state.items()},
            "raw_action": np.asarray(self.raw_action).astype(np.int64).tolist(),
            "action_taken": np.asarray(self.action_taken).astype(np.int64).tolist(),
            "reward": float(self.reward),
            "identity_tail": np.asarray(self.identity_tail).astype(np.float32).tolist(),
            "scenario_info": _to_listish(self.scenario_info),
        }

    @classmethod
    def from_json_dict(cls, d: Dict) -> "ObsAuditSample":
        """Inverse of :meth:`to_json_dict` --- recover a sample from JSONL."""
        return cls(
            t=int(d["t"]),
            agent_id=int(d["agent_id"]),
            flat_obs=np.asarray(d["flat_obs"], dtype=np.float32),
            env_state={k: np.asarray(v) for k, v in d["env_state"].items()},
            raw_action=np.asarray(d["raw_action"], dtype=np.int64),
            action_taken=np.asarray(d["action_taken"], dtype=np.int64),
            reward=float(d["reward"]),
            identity_tail=np.asarray(d["identity_tail"], dtype=np.float32),
            scenario_info=dict(d.get("scenario_info", {})),
        )


class ObsAuditor:
    """Reservoir-style observation auditor.

    Picks ``num_samples`` global step indices uniformly from
    ``[0, total_steps)`` up-front (seeded for determinism), then records
    every (agent, step) pair whose step is in the picked set.

    Usage::

        auditor = ObsAuditor(num_samples=500, total_steps=50_000, seed=0)
        trainer = JointPPOTrainer(..., obs_auditor=auditor)
        for it in range(num_iterations):
            trainer.collect_rollout(rollout_steps)
            trainer.update(...)
        auditor.dump(Path("audit.jsonl"))

    The trainer-side wire-in is the only mechanism that mutates state on
    this object; the auditor is otherwise inert. ``N=0`` makes
    :meth:`maybe_record` a fast no-op so the audit-disabled path is
    bit-for-bit equivalent to legacy ``collect_rollout``.

    Args:
        num_samples: Number of *global timesteps* to record. Each picked
            timestep contributes ``num_agents`` records (one per agent).
        total_steps: Total number of training timesteps the trainer will
            visit during the run. Used to compute the picked step set;
            if the run exceeds this budget, the extra steps are
            silently skipped.
        seed: Seed for the picked-step RNG.
    """

    def __init__(
        self,
        num_samples: int,
        total_steps: int,
        seed: int = 0,
    ):
        if num_samples < 0:
            raise ValueError(f"num_samples must be >= 0, got {num_samples!r}")
        if total_steps < 0:
            raise ValueError(f"total_steps must be >= 0, got {total_steps!r}")

        self.num_samples = int(num_samples)
        self.total_steps = int(total_steps)
        self.seed = int(seed)

        # Pre-compute the set of global step indices we will record. Picking
        # up-front avoids per-step RNG state and makes the audit deterministic
        # against `seed` regardless of how the trainer schedules rollouts.
        if self.num_samples == 0 or self.total_steps == 0:
            self._picked: set = set()
        elif self.num_samples >= self.total_steps:
            # Record everything.
            self._picked = set(range(self.total_steps))
        else:
            rng = np.random.default_rng(self.seed)
            picks = rng.choice(
                self.total_steps,
                size=self.num_samples,
                replace=False,
            )
            self._picked = set(int(x) for x in picks)

        self.samples: List[ObsAuditSample] = []
        # Monotonic global step counter, advanced by the wire-in's
        # `advance_step()` call after each env step.
        self._global_step: int = 0

    # ------------------------------------------------------------------
    # Trainer-facing hooks
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """``True`` if any step is scheduled to be recorded.

        Allows callers to skip the work of assembling the env_state dict
        when ``num_samples == 0`` --- this is what makes the audit-disabled
        path bit-identical to legacy ``collect_rollout``.
        """
        return bool(self._picked)

    def should_record(self, step: Optional[int] = None) -> bool:
        """Whether the *current* (or specified) global step is in the picked set."""
        s = self._global_step if step is None else int(step)
        return s in self._picked

    def advance_step(self) -> None:
        """Advance the internal global step counter by 1.

        Called by the wire-in once per env step (after all per-agent
        records for that step have been emitted).
        """
        self._global_step += 1

    @property
    def global_step(self) -> int:
        return self._global_step

    def maybe_record(
        self,
        agent_id: int,
        flat_obs: np.ndarray,
        env_state: Dict,
        raw_action: np.ndarray,
        action_taken: np.ndarray,
        reward: float,
        scenario_info: Optional[Dict] = None,
        num_agents: Optional[int] = None,
        step: Optional[int] = None,
    ) -> None:
        """Record one (step, agent) sample if the current step is picked.

        When :attr:`enabled` is ``False`` (``num_samples == 0``) this is
        a true no-op --- no env_state copies, no list appends.

        Args:
            agent_id: Index of the agent that consumed ``flat_obs``.
            flat_obs: ``[obs_dim]`` float32 — exact tensor fed to the policy.
            env_state: Dict snapshot of the pre-flatten env observation.
            raw_action: Raw multi-discrete action the policy emitted.
            action_taken: Sanitized action the env applied (read from
                ``env.last_actions[agent_id]`` after ``step()``).
            reward: Per-agent instantaneous reward at this step.
            scenario_info: Free-form dict for scenario knobs.
            num_agents: Width of the identity one-hot tail. Used to extract
                the ``identity_tail`` slice from ``flat_obs``. If ``None``,
                no slice is recorded.
            step: Optional explicit global step (defaults to the auditor's
                internal counter).
        """
        if not self._picked:
            return
        s = self._global_step if step is None else int(step)
        if s not in self._picked:
            return

        if num_agents is not None and num_agents > 0:
            identity_tail = np.asarray(flat_obs[-num_agents:], dtype=np.float32).copy()
        else:
            identity_tail = np.zeros(0, dtype=np.float32)

        env_state_copy = {
            k: (np.asarray(v).copy() if isinstance(v, (np.ndarray, list, tuple)) else v)
            for k, v in env_state.items()
        }

        self.samples.append(
            ObsAuditSample(
                t=s,
                agent_id=int(agent_id),
                flat_obs=np.asarray(flat_obs, dtype=np.float32).copy(),
                env_state=env_state_copy,
                raw_action=np.asarray(raw_action, dtype=np.int64).copy(),
                action_taken=np.asarray(action_taken, dtype=np.int64).copy(),
                reward=float(reward),
                identity_tail=identity_tail,
                scenario_info=dict(scenario_info or {}),
            )
        )

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def dump(self, path: Path) -> None:
        """Write all collected samples to ``path`` as JSON-Lines.

        One record per line. Numpy arrays are emitted as nested lists; the
        round-trip via :meth:`ObsAuditSample.from_json_dict` is dtype-faithful.

        When :attr:`enabled` is False this is a no-op --- no file is created.
        """
        if not self._picked:
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            for s in self.samples:
                f.write(json.dumps(s.to_json_dict()) + "\n")

    @classmethod
    def load(cls, path: Path) -> List[ObsAuditSample]:
        """Read back a JSONL file into a list of :class:`ObsAuditSample`."""
        out: List[ObsAuditSample] = []
        p = Path(path)
        with p.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(ObsAuditSample.from_json_dict(json.loads(line)))
        return out

    # ------------------------------------------------------------------
    # Helpers for analyzers / tests
    # ------------------------------------------------------------------

    def picked_steps(self) -> Tuple[int, ...]:
        """Return the (sorted) tuple of global step indices that will be recorded."""
        return tuple(sorted(self._picked))
