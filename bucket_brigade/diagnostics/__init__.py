"""Diagnostic helpers shared between standalone scripts and pytest regressions.

This module exposes hermetic, importable building blocks used by the H1/H2/H3
diagnostics under ``experiments/p3_specialization/diagnostics/`` and by the
slow regression test in ``tests/test_env_health_diagnostics.py`` (issue #201).

Why this module exists
----------------------
The standalone scripts under ``experiments/p3_specialization/diagnostics/``
were originally built for ad-hoc investigation and pull data from on-disk
artifacts (e.g. ``inspect_rollout_rewards.py`` rsyncs a trained cell into
``/tmp/h1_cell``). For the env-health regression test to run hermetically in
CI/local sweeps it needs the same per-rollout statistics computed against a
freshly-constructed, random-init ``JointPPOTrainer`` — no on-disk dependency.

The functions here factor out that "rollout → per-agent stats" path so both
callers share a single source of truth.

Public API
----------
- :func:`random_init_rollout_stats` — H1 hermetic: random-init MLP rollout
  returning per-agent CV and action-reward R².
- :func:`per_agent_reward_stats` — shared CV/percentile helper.
- :func:`conditional_mean_r2` — class-mean regressor R² (action ↔ reward).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Pure-numpy reward statistics (no torch dependency)
# ---------------------------------------------------------------------------


def per_agent_reward_stats(r: np.ndarray) -> Dict[str, float]:
    """Distributional stats for a single agent's per-step reward series.

    Mirrors ``inspect_rollout_rewards.per_agent_stats`` so the hermetic
    pytest path and the on-disk diagnostic path return identical numbers.
    """
    mean = float(r.mean())
    std = float(r.std())
    cv = std / (abs(mean) + 1e-9)
    p10, p50, p90 = (float(x) for x in np.percentile(r, [10, 50, 90]))
    return {"mean": mean, "std": std, "cv": cv, "p10": p10, "p50": p50, "p90": p90}


def conditional_mean_r2(r: np.ndarray, labels: np.ndarray) -> float:
    """R² of the optimal class-mean regressor of ``r`` on ``labels``.

    Equivalent to 1 − SS_within / SS_total — i.e. the fraction of per-step
    reward variance explained by the discrete action label. Returns NaN if
    ``r`` has zero variance (degenerate case).
    """
    ss_total = float(((r - r.mean()) ** 2).sum())
    if ss_total <= 1e-12:
        return float("nan")
    pred = np.zeros_like(r, dtype=np.float64)
    for lab in np.unique(labels):
        mask = labels == lab
        pred[mask] = r[mask].mean()
    ss_res = float(((r - pred) ** 2).sum())
    return 1.0 - ss_res / ss_total


# ---------------------------------------------------------------------------
# Hermetic random-init rollout (H1)
# ---------------------------------------------------------------------------


def random_init_rollout_stats(
    scenario_name: str = "default",
    *,
    num_agents: int = 4,
    hidden_size: int = 64,
    rollout_steps: int = 2048,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Run one rollout under a fresh random-init ``JointPPOTrainer``.

    Hermetic version of the H1 diagnostic — no ``/tmp/h1_cell`` dependency.
    Builds a ``JointPPOTrainer`` with the canonical phase-3 hyperparameters
    (``hidden_size=64``, ``num_agents=4``, ``action_dims=[10, 2]``) and runs
    ``trainer.collect_rollout(rollout_steps)``. Returns the per-step reward
    matrix, action matrix, and per-agent stats dicts.

    This mirrors the trained-cell path in
    ``experiments/p3_specialization/diagnostics/inspect_rollout_rewards.py``
    but skips the on-disk policy load, so it is safe to call from pytest.

    Parameters
    ----------
    scenario_name : str
        Name in ``SCENARIO_REGISTRY`` (e.g. ``"default"``,
        ``"minimal_specialization"``).
    num_agents : int
        Number of agents in the env (default 4, matching phase-3 cells).
    hidden_size : int
        Policy MLP hidden size (default 64, matching ``CellConfig``).
    rollout_steps : int
        Number of synchronized env steps to collect. 2048 matches the
        default ``rollout_steps`` used in #190's diagnostic.
    seed : int
        RNG seed for both torch and numpy.

    Returns
    -------
    R : ndarray ``[N, T]``
        Per-step rewards per agent.
    A : ndarray ``[N, T, 2]``
        Per-step ``[house_index, work_flag]`` actions per agent.
    per_agent : list of dict
        One dict per agent with keys: ``mean``, ``std``, ``cv``, ``p10``,
        ``p50``, ``p90``, ``r2_packed`` (20-class action), ``r2_work``
        (binary work/rest only).
    """
    # Local imports — torch may be unavailable in CI lite installs, and this
    # module otherwise has no heavy deps.
    import torch

    from bucket_brigade.envs.bucket_brigade_env import BucketBrigadeEnv
    from bucket_brigade.envs.scenarios_generated import get_scenario_by_name
    from bucket_brigade.training.joint_trainer import (
        JointPPOTrainer,
        flatten_dict_obs,
    )

    scenario = get_scenario_by_name(scenario_name, num_agents=num_agents)

    def env_fn() -> BucketBrigadeEnv:
        return BucketBrigadeEnv(scenario=scenario)

    probe = env_fn()
    obs_dim = flatten_dict_obs(probe.reset(seed=seed)).shape[0]

    trainer = JointPPOTrainer(
        env_fn=env_fn,
        num_agents=num_agents,
        obs_dim=obs_dim,
        action_dims=[10, 2],
        hidden_size=hidden_size,
        seed=seed,
    )

    rollout = trainer.collect_rollout(rollout_steps)

    R = torch.stack([rollout.rewards[i] for i in range(num_agents)]).cpu().numpy()
    A = torch.stack([rollout.actions[i] for i in range(num_agents)]).cpu().numpy()

    per_agent: List[Dict[str, Any]] = []
    for i in range(num_agents):
        r = R[i]
        a = A[i]
        stats = per_agent_reward_stats(r)
        packed = a[:, 0] * 2 + a[:, 1]  # 10 houses × {rest, work}
        stats["r2_packed"] = float(conditional_mean_r2(r, packed))
        stats["r2_work"] = float(conditional_mean_r2(r, a[:, 1]))
        per_agent.append(stats)

    return R, A, per_agent


__all__ = [
    "conditional_mean_r2",
    "per_agent_reward_stats",
    "random_init_rollout_stats",
]
