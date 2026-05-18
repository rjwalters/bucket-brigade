"""Random-init best-of-N with K=20 → K=200 stability re-eval (issue #292).

Adapts ``experiments/p3_specialization/diagnostics/random_mlp_search.py``
(issue #271) to the minimal-dilemma env. The K=20 → K=200 two-phase protocol
is **mandatory** per the issue spec — it's the safeguard that flagged the
false-positive anti-attractor signal in the bucket-brigade version. Failing
to re-eval at K=200 would let a lucky 20-episode draw masquerade as a stable
cooperative basin.

Protocol:

1. **Phase 1 (K=20):** sample ``--seeds`` (default 1000) random-init policy
   bundles. For each seed evaluate ``--episodes-per-seed`` (default 20)
   episodes on the dilemma env and record mean per-step per-agent reward.
2. **Phase 2 (K=200):** take the top ``--top-frac`` (default 1%) of seeds
   and re-evaluate with ``--restability-episodes`` (default 200) **disjoint**
   episodes (different episode seeds than phase 1).
3. **Stability gate (verdict input):** the random-init best-of-N arm is
   "honest" only if top-1% phase-2 mean ≤ phase-1 mean + slack. The
   verdict classifier (``verdict.py``) consumes this to flag basin-trap
   vs anti-attractor.

Per-agent init protocol: ``--protocol independent`` (default) uses
``seed * NUM_AGENTS + i`` for agent ``i`` so each policy net is drawn from a
distinct slice of parameter space. ``--protocol shared`` matches the BC-init
experiment's single-seed broadcast.

Episodes are short (50 steps) and policies are tiny (a 7-d input MLP), so the
per-seed cost is microscopic and we run sequentially — no multiprocessing
needed for the minimal env. (The bucket-brigade version forks because each
episode is ~3-12 nights of Rust env.)

Usage::

    # smoke (~15 sec on laptop)
    uv run python -m experiments.p3_specialization.minimal_dilemma.best_of_n \\
        --seeds 50 --episodes-per-seed 5 --restability-episodes 10 \\
        --output-dir /tmp/dilemma_bon_smoke

    # full run on COMPUTE_HOST_PRIMARY (~5-15 min)
    uv run python -m experiments.p3_specialization.minimal_dilemma.best_of_n \\
        --seeds 1000 --episodes-per-seed 20 --restability-episodes 200 \\
        --output-dir experiments/p3_specialization/minimal_dilemma/results/best_of_n
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from bucket_brigade.training.joint_trainer import flatten_dict_obs
from bucket_brigade.training.networks import PolicyNetwork

from experiments.p3_specialization.minimal_dilemma.env import (
    ACTION_COOPERATE,
    EPISODE_LENGTH,
    MULTIPLIER,
    MinimalDilemmaEnv,
    NUM_AGENTS,
    REWARD_MUTUAL_COOPERATE,
    REWARD_MUTUAL_DEFECT,
)


HIDDEN_SIZE = 64
ACTION_DIMS: List[int] = [2]


def gap_closed(per_agent_per_step: float) -> float:
    """Fraction of the defect→cooperate self-play gap closed."""
    span = REWARD_MUTUAL_COOPERATE - REWARD_MUTUAL_DEFECT
    if span <= 0:
        return float("nan")
    return float((per_agent_per_step - REWARD_MUTUAL_DEFECT) / span)


def _build_policies(seed: int, obs_dim: int, protocol: str) -> List[PolicyNetwork]:
    """Instantiate one PolicyNetwork per agent under the chosen init protocol."""
    policies: List[PolicyNetwork] = []
    if protocol == "shared":
        torch.manual_seed(seed)
        np.random.seed(seed)
        for _ in range(NUM_AGENTS):
            policies.append(
                PolicyNetwork(
                    obs_dim=obs_dim,
                    action_dims=ACTION_DIMS,
                    hidden_size=HIDDEN_SIZE,
                )
            )
    elif protocol == "independent":
        for i in range(NUM_AGENTS):
            agent_seed = seed * NUM_AGENTS + i
            torch.manual_seed(agent_seed)
            np.random.seed(agent_seed)
            policies.append(
                PolicyNetwork(
                    obs_dim=obs_dim,
                    action_dims=ACTION_DIMS,
                    hidden_size=HIDDEN_SIZE,
                )
            )
    else:
        raise ValueError(f"Unknown protocol: {protocol!r}")
    for p in policies:
        p.eval()
    return policies


def _evaluate_seed(
    seed: int,
    n_episodes: int,
    protocol: str,
    episode_seed_base: int,
    multiplier: float = MULTIPLIER,
    episode_length: int = EPISODE_LENGTH,
) -> Dict[str, object]:
    """Evaluate one random-init policy bundle on the dilemma env.

    Returns a dict with the same shape as the worker output in #271's
    ``random_mlp_search.py`` so the verdict consumer is template-compatible.
    """
    env = MinimalDilemmaEnv(multiplier=multiplier, episode_length=episode_length)
    obs_dim = flatten_dict_obs(
        env.reset(seed=0), agent_id=0, num_agents=NUM_AGENTS
    ).shape[0]
    policies = _build_policies(seed, obs_dim, protocol)

    per_step: List[float] = []
    coop_fracs: List[float] = []
    entropy_samples: List[float] = []

    for ep in range(n_episodes):
        # Disjoint episode-seed stream: keyed by (base, seed, episode index).
        ep_seed = episode_seed_base + seed * 100_000 + ep
        obs = env.reset(seed=ep_seed)
        obs_rows = np.stack(
            [
                flatten_dict_obs(obs, agent_id=i, num_agents=NUM_AGENTS)
                for i in range(NUM_AGENTS)
            ],
            axis=0,
        )
        total_reward = 0.0
        n_steps = 0
        n_coop = 0
        first_step = True
        while not env.done:
            obs_t = torch.from_numpy(obs_rows)
            joint_action = np.zeros((NUM_AGENTS, len(ACTION_DIMS)), dtype=np.int64)
            with torch.no_grad():
                for i, policy in enumerate(policies):
                    a, _, ent, _ = policy.get_action_and_value(obs_t[i : i + 1])
                    a_i = int(a[0].cpu().item())
                    joint_action[i] = a_i
                    if first_step:
                        entropy_samples.append(float(ent.item()))
                    if a_i == ACTION_COOPERATE:
                        n_coop += 1
            next_obs, rewards, _, _ = env.step(joint_action)
            total_reward += float(rewards.sum())
            n_steps += 1
            first_step = False
            if not env.done:
                obs_rows = np.stack(
                    [
                        flatten_dict_obs(next_obs, agent_id=i, num_agents=NUM_AGENTS)
                        for i in range(NUM_AGENTS)
                    ],
                    axis=0,
                )
        # Per-agent per-step reward = team_sum / (n_steps * num_agents).
        per_step.append(total_reward / max(1, n_steps * NUM_AGENTS))
        coop_fracs.append(n_coop / max(1, n_steps * NUM_AGENTS))

    arr = np.asarray(per_step, dtype=np.float64)
    return {
        "seed": int(seed),
        "protocol": protocol,
        "per_episode": [float(x) for x in arr],
        "mean_per_step": float(arr.mean()),
        "std_per_step": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "coop_fraction_mean": float(np.mean(coop_fracs)) if coop_fracs else 0.0,
        "entropy": float(np.mean(entropy_samples)) if entropy_samples else float("nan"),
    }


def _bootstrap_ci(
    arr: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, rng_seed: int = 0
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean. Adapted from #271's implementation."""
    rng = np.random.default_rng(rng_seed)
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan")
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boots[i] = arr[rng.integers(0, n, size=n)].mean()
    return (
        float(np.percentile(boots, 100 * alpha / 2)),
        float(np.percentile(boots, 100 * (1 - alpha / 2))),
    )


def stability_gate(
    top_phase1: Sequence[Dict[str, object]],
    top_phase2: Sequence[Dict[str, object]],
    slack: float = 0.05,
) -> Dict[str, object]:
    """K=20 → K=200 stability gate (the #271 false-positive safeguard).

    The gate **passes** (= no random-net bridge to cooperation, basin-trap
    consistent) iff the top-1% phase-2 mean is no higher than the phase-1
    mean plus ``slack``. A failure (phase-2 mean >> phase-1) would mean a
    random net stably reaches cooperative reward — i.e. cooperation is
    structurally accessible without gradient descent, falsifying basin-trap.

    Args:
        top_phase1: phase-1 results for the seeds re-evaluated in phase 2.
        top_phase2: corresponding phase-2 results (same seeds, K=200).
        slack: tolerance for noise (per-step reward units). Default 0.05.

    Returns:
        Dict with ``passed``, ``phase1_mean``, ``phase2_mean``,
        ``phase2_max``, ``drift`` (phase2_mean − phase1_mean), and the
        verdict-ready ``random_bridge_top1_mean`` (phase-2 estimate of
        random-net best-of-N reward, used by the 4-gate verdict).
    """
    p1_means = np.asarray([r["mean_per_step"] for r in top_phase1], dtype=np.float64)
    p2_means = np.asarray([r["mean_per_step"] for r in top_phase2], dtype=np.float64)
    phase1_mean = float(p1_means.mean()) if p1_means.size else float("nan")
    phase2_mean = float(p2_means.mean()) if p2_means.size else float("nan")
    phase2_max = float(p2_means.max()) if p2_means.size else float("nan")
    drift = phase2_mean - phase1_mean
    # Drift is *negative* when phase-2 confirms phase-1 was lucky. We want
    # ``phase2_mean <= phase1_mean + slack``: i.e. no upward drift beyond
    # slack. The bridge fails (= basin-trap consistent) when this holds.
    bridge_failed = bool(drift <= slack)
    return {
        "passed": bridge_failed,  # "passed" = random bridge did NOT happen
        "phase1_mean": phase1_mean,
        "phase2_mean": phase2_mean,
        "phase2_max": phase2_max,
        "drift": float(drift),
        "slack": float(slack),
        "random_bridge_top1_mean": phase2_mean,
        "random_bridge_top1_max": phase2_max,
        "random_bridge_top1_gap_closed": gap_closed(phase2_mean),
    }


def _run_protocol(
    protocol: str,
    seeds: Sequence[int],
    episodes_per_seed: int,
    restability_episodes: int,
    top_frac: float,
    multiplier: float,
    episode_length: int,
    out_dir: Path,
) -> Dict[str, object]:
    """Execute the full best-of-N + stability re-eval for one init protocol."""
    print(f"\n=== Protocol: {protocol} ===")
    print(f"Phase 1: {len(seeds)} seeds x {episodes_per_seed} episodes")

    phase1_results: List[Dict[str, object]] = []
    for idx, s in enumerate(seeds, 1):
        phase1_results.append(
            _evaluate_seed(
                seed=int(s),
                n_episodes=episodes_per_seed,
                protocol=protocol,
                episode_seed_base=0,
                multiplier=multiplier,
                episode_length=episode_length,
            )
        )
        if idx % max(1, len(seeds) // 10) == 0:
            print(f"  [{protocol}] phase-1 progress: {idx}/{len(seeds)}")

    phase1_results.sort(key=lambda r: r["mean_per_step"], reverse=True)
    means = np.asarray([r["mean_per_step"] for r in phase1_results])
    n_top = max(1, int(round(top_frac * len(phase1_results))))
    top = phase1_results[:n_top]
    print(
        f"Phase 1 complete. mean={means.mean():.3f} std={means.std():.3f}, "
        f"top-{int(top_frac * 100)}% threshold={top[-1]['mean_per_step']:.3f}"
    )

    print(f"Phase 2: re-eval top-{n_top} seeds with {restability_episodes} episodes")
    phase2_results: List[Dict[str, object]] = []
    for idx, r in enumerate(top, 1):
        phase2_results.append(
            _evaluate_seed(
                seed=int(r["seed"]),
                n_episodes=restability_episodes,
                protocol=protocol,
                episode_seed_base=7_000_000,
                multiplier=multiplier,
                episode_length=episode_length,
            )
        )
        if idx % max(1, len(top) // 5) == 0:
            print(f"  [{protocol}] phase-2 progress: {idx}/{len(top)}")

    # Attach bootstrap CIs.
    for r in phase2_results:
        arr = np.asarray(r["per_episode"], dtype=np.float64)
        lo, hi = _bootstrap_ci(arr)
        r["ci95_lo"] = lo
        r["ci95_hi"] = hi
    phase2_results.sort(key=lambda r: r["mean_per_step"], reverse=True)

    gate = stability_gate(top, phase2_results)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"results_{protocol}.json").open("w") as f:
        json.dump(
            [
                {
                    "seed": r["seed"],
                    "mean_per_step": r["mean_per_step"],
                    "std_per_step": r["std_per_step"],
                    "coop_fraction_mean": r["coop_fraction_mean"],
                    "entropy": r["entropy"],
                }
                for r in phase1_results
            ],
            f,
            indent=2,
        )
    with (out_dir / f"top_candidates_{protocol}.json").open("w") as f:
        json.dump(top, f, indent=2)
    with (out_dir / f"stability_{protocol}.json").open("w") as f:
        json.dump(phase2_results, f, indent=2)

    summary = {
        "protocol": protocol,
        "n_seeds": len(phase1_results),
        "episodes_per_seed_phase1": int(episodes_per_seed),
        "episodes_per_seed_phase2": int(restability_episodes),
        "n_top": int(n_top),
        "population_mean": float(means.mean()),
        "population_std": float(means.std(ddof=1)) if len(means) > 1 else 0.0,
        "stability_gate": gate,
    }
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=1000)
    p.add_argument("--episodes-per-seed", type=int, default=20)
    p.add_argument("--restability-episodes", type=int, default=200)
    p.add_argument("--top-frac", type=float, default=0.01)
    p.add_argument(
        "--protocol",
        choices=["independent", "shared", "both"],
        default="independent",
        help="Per-agent init protocol; 'both' runs each in turn.",
    )
    p.add_argument("--multiplier", type=float, default=MULTIPLIER)
    p.add_argument("--episode-length", type=int, default=EPISODE_LENGTH)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_offset, args.seed_offset + args.seeds))

    protocols = (
        ["independent", "shared"] if args.protocol == "both" else [args.protocol]
    )

    summaries: Dict[str, object] = {}
    for proto in protocols:
        summaries[proto] = _run_protocol(
            protocol=proto,
            seeds=seeds,
            episodes_per_seed=args.episodes_per_seed,
            restability_episodes=args.restability_episodes,
            top_frac=args.top_frac,
            multiplier=args.multiplier,
            episode_length=args.episode_length,
            out_dir=args.output_dir,
        )

    with (args.output_dir / "summary.json").open("w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nWrote {args.output_dir}/summary.json")
    for proto, s in summaries.items():
        g = s["stability_gate"]
        print(
            f"  [{proto}] stability_gate.passed={g['passed']} "
            f"phase1_mean={g['phase1_mean']:.3f} "
            f"phase2_mean={g['phase2_mean']:.3f} "
            f"drift={g['drift']:+.3f}"
        )


if __name__ == "__main__":
    main()
