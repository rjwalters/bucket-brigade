"""Wrap a saved PPO checkpoint as an ``AgentBase``-compatible "trained archetype".

Loads a ``state_dict`` saved by ``experiments/p3_specialization/train.py``
(per-agent ``agent_{0..N-1}.pt`` files in a ``policies/`` directory) and
exposes the resulting :class:`bucket_brigade.training.networks.PolicyNetwork`
through the standard :meth:`AgentBase.act` contract. This makes a trained
policy a drop-in replacement for a :class:`HeuristicAgent` inside a Monte
Carlo rollout for issue #275 (Nash equilibrium of *trained* PPO policies).

The wrapper is deliberately minimal:

* Per-agent checkpoint loading — the trainer saved one ``state_dict`` per
  agent, so each ``TrainedPolicyArchetype`` instance represents one slot
  in a team of ``num_agents`` policies. ``act()`` flattens the dict-obs
  using the same per-agent identity one-hot tail as
  :func:`flatten_dict_obs`, so the network sees the same input shape it
  trained on.
* Deterministic by default — issue #275 wants the Nash of *each* trained
  policy as a fixed strategy; sampling would inject stochasticity that the
  Nash solver doesn't model. The ``deterministic`` flag can be flipped for
  diagnostic rollouts.
* Honest signal — the network outputs a third action head (signal) when
  ``action_dims = [10, 2, 2]``; we pass it through. For two-head legacy
  checkpoints (pre-#236) loading is **refused by default**: synthesizing
  ``signal := mode`` would fabricate an output the original policy never
  produced. Pass ``allow_legacy_2head=True`` to opt in to the synthesis
  (with a runtime warning). See issue #325 for the audit trail.

The wrapper does *not* try to fit a 10-d θ vector to clone a network — see
the Curator notes on issue #275: that approach (Option A) is lossy because
the heuristic family is low-capacity. The mixed payoff evaluator in
``experiments/scripts/compute_nash_trained.py`` calls ``act()`` directly
(Option B).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from bucket_brigade.agents.agent_base import AgentBase
from bucket_brigade.training.joint_trainer import flatten_dict_obs
from bucket_brigade.training.networks import PolicyNetwork


class TrainedPolicyArchetype(AgentBase):
    """An ``AgentBase`` wrapper around a saved :class:`PolicyNetwork` checkpoint.

    Parameters
    ----------
    state_dict_path
        Path to a ``.pt`` file produced by ``torch.save(policy.state_dict(),
        ...)`` in ``experiments/p3_specialization/train.py``.
    agent_id
        Which slot in the team this archetype occupies. The flattened
        observation includes a per-agent identity one-hot tail (#204), so
        the same checkpoint plays very differently in slot 0 vs slot 3.
    num_agents
        Team size. Required for the identity one-hot.
    obs_dim
        Expected flattened-obs dimension. If ``None``, inferred from the
        first linear layer of ``shared`` in the state dict.
    action_dims
        Per-dim action sizes. Defaults to ``[10, 2, 2]`` (the post-#235 BB
        action space). If ``None``, inferred from the last linear-head
        output sizes.
    hidden_size
        Hidden size of the shared trunk. If ``None``, inferred from the
        first linear-layer output size in the state dict.
    deterministic
        If ``True`` (default), pick the argmax action per head. If ``False``,
        sample from the categorical distribution (useful for diagnostics).
    device
        Torch device. Default ``"cpu"`` — Nash payoff Monte Carlo is
        CPU-bound (#198 small env) so GPU launch overhead dominates.
    name
        Optional human-readable name (e.g. ``"ippo_seed42_agent0"``).
    allow_legacy_2head
        If ``False`` (default), refuse to load checkpoints with only 2
        action heads (legacy pre-#236 policies) by raising ``ValueError``.
        If ``True``, accept the legacy checkpoint and synthesize the
        missing signal channel as ``signal := mode`` at act time, emitting
        a ``UserWarning``. See issue #325 — all 244 existing checkpoints
        under ``experiments/p3_specialization/`` on 2026-05-18 were 2-head,
        so this gate is the difference between an honest "no signal" verdict
        and a silently-fabricated one.
    """

    def __init__(
        self,
        state_dict_path: Path,
        agent_id: int,
        num_agents: int,
        obs_dim: Optional[int] = None,
        action_dims: Optional[List[int]] = None,
        hidden_size: Optional[int] = None,
        deterministic: bool = True,
        device: str = "cpu",
        name: Optional[str] = None,
        allow_legacy_2head: bool = False,
    ) -> None:
        super().__init__(agent_id, name or f"trained-{Path(state_dict_path).stem}")
        self.state_dict_path = Path(state_dict_path)
        self.num_agents = int(num_agents)
        self.deterministic = bool(deterministic)
        self.device = torch.device(device)
        self.allow_legacy_2head = bool(allow_legacy_2head)

        if not self.state_dict_path.exists():
            raise FileNotFoundError(
                f"TrainedPolicyArchetype: checkpoint missing: {self.state_dict_path}"
            )

        sd = torch.load(
            self.state_dict_path, map_location=self.device, weights_only=True
        )

        # Infer architecture from the state dict if not supplied explicitly.
        # ``shared.0.weight`` is the first Linear in the 2-layer MLP, so its
        # rows give us hidden_size and its columns give us obs_dim.
        inferred_hidden, inferred_obs = sd["shared.0.weight"].shape
        self.obs_dim = int(obs_dim) if obs_dim is not None else int(inferred_obs)
        self.hidden_size = (
            int(hidden_size) if hidden_size is not None else int(inferred_hidden)
        )

        # action_heads.<i>.weight gives action_dims[i] from rows.
        if action_dims is not None:
            self.action_dims = list(action_dims)
        else:
            inferred: List[int] = []
            i = 0
            while f"action_heads.{i}.weight" in sd:
                inferred.append(int(sd[f"action_heads.{i}.weight"].shape[0]))
                i += 1
            if not inferred:
                # Two-head fallback for very old checkpoints, though the
                # post-#235 default is [10, 2, 2].
                inferred = [10, 2, 2]
            self.action_dims = inferred

        # Gate legacy 2-head (pre-#236) checkpoints (#325). The PolicyNetwork
        # only knows about the heads it was constructed with, so a 2-head
        # checkpoint loaded as a 2-head network produces a 2-d action. The
        # downstream env expects 3-d (house, mode, signal); the historical
        # behavior was to silently append ``signal := mode`` at act time,
        # which fabricates a signal output the original policy never produced.
        # Refuse by default; opt in to the synthesis with a warning.
        if len(self.action_dims) == 2:
            if not self.allow_legacy_2head:
                raise ValueError(
                    f"TrainedPolicyArchetype: checkpoint {self.state_dict_path} "
                    f"has 2 action heads (legacy pre-#236). Loading would "
                    f"silently set `signal := mode`, fabricating a signal "
                    f"output the original policy never produced. Pass "
                    f"`allow_legacy_2head=True` to opt in to the backfill "
                    f"(see issue #325)."
                )
            warnings.warn(
                f"Loading legacy 2-head checkpoint {self.state_dict_path}; "
                f"signal := mode backfill active (see issue #325).",
                UserWarning,
                stacklevel=2,
            )

        self.policy = PolicyNetwork(
            obs_dim=self.obs_dim,
            action_dims=self.action_dims,
            hidden_size=self.hidden_size,
        ).to(self.device)
        self.policy.load_state_dict(sd)
        self.policy.eval()

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Choose an action from the wrapped policy.

        Flattens the obs with the per-agent identity one-hot tail (#204)
        and runs a single deterministic forward pass through the loaded
        :class:`PolicyNetwork`.
        """
        flat = flatten_dict_obs(obs, agent_id=self.agent_id, num_agents=self.num_agents)
        if flat.shape[0] != self.obs_dim:
            raise ValueError(
                f"TrainedPolicyArchetype: flattened obs dim {flat.shape[0]} != "
                f"checkpoint obs_dim {self.obs_dim}. Was the policy trained on a "
                "different scenario / num_agents?"
            )
        x = torch.from_numpy(flat).to(self.device).unsqueeze(0)  # [1, obs_dim]
        with torch.no_grad():
            actions, _, _ = self.policy.get_action(x, deterministic=self.deterministic)
        # actions is a list of length len(action_dims); each entry shape [1].
        out = [int(a[0].item()) for a in actions]

        # Backfill signal for legacy two-head checkpoints (signal := mode).
        # Only reachable when the caller opted into ``allow_legacy_2head=True``
        # at construction time; otherwise ``__init__`` raised (#325).
        if len(out) == 2:
            out.append(out[1])

        return np.array(out, dtype=np.int8)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"TrainedPolicyArchetype(id={self.agent_id}, "
            f"path={self.state_dict_path.name}, obs_dim={self.obs_dim}, "
            f"hidden={self.hidden_size}, action_dims={self.action_dims})"
        )
