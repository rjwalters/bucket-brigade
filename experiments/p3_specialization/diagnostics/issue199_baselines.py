"""Issue #199 sanity-check baselines.

Measures three reward numbers on the ``minimal_specialization`` scenario:

1. **Uniform-random** mean per-step team reward over 50 episodes.
2. **Specialist** (``bucket_brigade.baselines.specialist_action_joint``) mean
   per-step team reward over 50 episodes.
3. Same two measurements on the ``default`` scenario as a comparison baseline.

These three numbers feed into the issue #199 verdict decision tree::

    PPO iter-49 on minimal_specialization vs. (specialist - random) gap:
      >= 50% closed -> existing scenarios are the issue, not the algorithm.
      < 50% closed  -> algorithm fundamentally unsuited, need MAPPO/COMA.

The script is intentionally small and CPU-only: 50 episodes of <= 200 nights
each at ~4 agents is < 1 minute total. Safe to run locally per the CLAUDE.md
compute guidelines.

Results land at::

    experiments/p3_specialization/diagnostics/results/issue199_minspec/baselines.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np

from bucket_brigade.baselines import specialist_action_joint
from bucket_brigade.envs.bucket_brigade_env import BucketBrigadeEnv
from bucket_brigade.envs.scenarios_generated import get_scenario_by_name


def _run_random_episode(
    env: BucketBrigadeEnv, rng: np.random.Generator
) -> Tuple[float, int]:
    """One uniform-random episode. Returns (team_reward, nights_played).

    The "team reward" is the sum of per-agent env rewards summed across the
    episode, matching the convention used by ``random_baseline.py`` and the
    P3 specialization analysis scripts.
    """
    env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    total_reward = 0.0
    while not env.done:
        # MultiDiscrete([10, 2, 2]) per agent post-#236 (issue #235). Third
        # column is the broadcast signal channel; without this row the env
        # silently drops signals and the baseline measures a pre-#236 policy
        # (the bug class issue #237 / PR #244 fixed in two sibling sites).
        actions = np.stack(
            [
                rng.integers(0, 10, size=env.num_agents),
                rng.integers(0, 2, size=env.num_agents),
                rng.integers(0, 2, size=env.num_agents),
            ],
            axis=-1,
        ).astype(np.int64)
        _, rewards, _, _ = env.step(actions)
        total_reward += float(rewards.sum())
    return total_reward, int(env.night)


def _run_specialist_episode(env: BucketBrigadeEnv, seed: int) -> Tuple[float, int]:
    """One specialist-policy episode. Returns (team_reward, nights_played)."""
    obs = env.reset(seed=seed)
    total_reward = 0.0
    while not env.done:
        actions = specialist_action_joint(obs, env.num_agents, num_houses=10)
        obs, rewards, _, _ = env.step(actions)
        total_reward += float(rewards.sum())
    return total_reward, int(env.night)


def _bootstrap_ci(
    arr: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05
) -> Tuple[float, float]:
    rng = np.random.default_rng(0)
    n = len(arr)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = arr[idx].mean()
    return float(np.percentile(boots, 100 * alpha / 2)), float(
        np.percentile(boots, 100 * (1 - alpha / 2))
    )


def _summarize(
    label: str, per_step: np.ndarray, per_ep: np.ndarray, lengths: np.ndarray
) -> dict:
    lo, hi = _bootstrap_ci(per_step)
    return {
        "label": label,
        "n_episodes": int(len(per_step)),
        "per_step": {
            "mean": float(per_step.mean()),
            "ci95_lo": lo,
            "ci95_hi": hi,
            "std": float(per_step.std(ddof=1)) if len(per_step) > 1 else 0.0,
        },
        "per_episode_mean": float(per_ep.mean()),
        "episode_length": {
            "mean": float(lengths.mean()),
            "median": float(np.median(lengths)),
            "min": int(lengths.min()),
            "max": int(lengths.max()),
        },
    }


def _measure(scenario_name: str, num_agents: int, episodes: int, seed: int) -> dict:
    scenario = get_scenario_by_name(scenario_name, num_agents=num_agents)
    env = BucketBrigadeEnv(scenario=scenario)

    # ----- Uniform random -----
    rng = np.random.default_rng(seed)
    rand_per_ep, rand_per_step, rand_lengths = [], [], []
    for _ in range(episodes):
        ep, nights = _run_random_episode(env, rng)
        rand_per_ep.append(ep)
        rand_per_step.append(ep / nights)
        rand_lengths.append(nights)
    rand_summary = _summarize(
        "random",
        np.asarray(rand_per_step),
        np.asarray(rand_per_ep),
        np.asarray(rand_lengths),
    )

    # ----- Specialist -----
    spec_per_ep, spec_per_step, spec_lengths = [], [], []
    for ep_idx in range(episodes):
        ep, nights = _run_specialist_episode(env, seed=seed * 10_000 + ep_idx)
        spec_per_ep.append(ep)
        spec_per_step.append(ep / nights)
        spec_lengths.append(nights)
    spec_summary = _summarize(
        "specialist",
        np.asarray(spec_per_step),
        np.asarray(spec_per_ep),
        np.asarray(spec_lengths),
    )

    return {
        "scenario": scenario_name,
        "num_agents": num_agents,
        "episodes_per_baseline": episodes,
        "seed": seed,
        "random": rand_summary,
        "specialist": spec_summary,
        "specialist_minus_random_per_step": (
            spec_summary["per_step"]["mean"] - rand_summary["per_step"]["mean"]
        ),
        "ratio_specialist_over_random_per_step": (
            spec_summary["per_step"]["mean"] / rand_summary["per_step"]["mean"]
            if rand_summary["per_step"]["mean"] != 0
            else None
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of episodes per baseline. #199 spec asks for 50.",
    )
    ap.add_argument("--num-agents", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--scenarios",
        nargs="+",
        default=["minimal_specialization", "default"],
        help="Scenarios to measure. Defaults to the primary (#199 target) and "
        "the legacy default for comparison.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "issue199_minspec",
    )
    args = ap.parse_args()

    results = {}
    for scenario_name in args.scenarios:
        print("=" * 70)
        print(f"Measuring random vs. specialist on '{scenario_name}'")
        print("=" * 70)
        block = _measure(scenario_name, args.num_agents, args.episodes, args.seed)
        results[scenario_name] = block

        rs = block["random"]["per_step"]
        ss = block["specialist"]["per_step"]
        print(
            f"  random     per-step: mean={rs['mean']:.2f}  "
            f"95% CI=[{rs['ci95_lo']:.2f}, {rs['ci95_hi']:.2f}]  n={block['random']['n_episodes']}"
        )
        print(
            f"  specialist per-step: mean={ss['mean']:.2f}  "
            f"95% CI=[{ss['ci95_lo']:.2f}, {ss['ci95_hi']:.2f}]  n={block['specialist']['n_episodes']}"
        )
        gap = block["specialist_minus_random_per_step"]
        ratio = block["ratio_specialist_over_random_per_step"]
        ratio_str = f"{ratio:.2f}x" if ratio is not None else "n/a"
        print(f"  specialist - random: {gap:+.2f}  (specialist/random = {ratio_str})")
        print()

    # Sanity check: specialist must beat random on minimal_specialization, else
    # the scenario design is broken (per the issue spec's honest-reporting clause).
    ms = results.get("minimal_specialization")
    if ms is not None:
        gap = ms["specialist_minus_random_per_step"]
        if gap <= 0:
            print(
                "WARNING: specialist does NOT beat random on minimal_specialization. "
                "Scenario design may be broken — flag back to curator."
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "baselines.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
