"""Issue #271: random-init MLP best-of-N vs trained PPO on ``minimal_specialization``.

Question: does naive best-of-N random search over policy networks beat trained
PPO on ``minimal_specialization``? If yes, the gradient is actively misleading
— strong evidence for an *active anti-attractor* (vs a plain basin trap).

Protocol (matches the curator-enriched issue spec)
--------------------------------------------------

1. Sample ``--seeds`` (default 1000) random-init policy bundles. For each
   seed, evaluate ``--episodes-per-seed`` (default 20) episodes on
   ``minimal_specialization`` and record mean per-step team reward
   (``ep_reward / env.night``; see ``random_baseline.run_mlp_episode``).
2. Per-agent init protocol: ``--protocol`` ∈ {``shared``, ``independent``}.
   ``shared``: all 4 agents start from the same seed (broadcast). Matches
   the BC-init experiment (#270).
   ``independent`` (default per issue spec): agent ``i`` uses
   ``seed * 4 + i`` so each of the 4 PolicyNetworks is drawn from a distinct
   slice of the parameter-product space.
3. Top-1% noise-stability re-eval (MANDATORY per spec): take the top
   ``--top-frac`` (default 1%) of seeds and re-evaluate each with
   ``--restability-episodes`` (default 200) episodes. Without this step we
   can't tell "stably good" from "lucky 20-episode draw" (per-episode std on
   ``minimal_specialization`` is ~95 — see ``issue199_minspec/baselines.json``).
4. Parallelism: ``multiprocessing`` spawn context + ``torch.set_num_threads(1)``
   per worker so the per-worker MLP forward doesn't oversubscribe cores.

Output (under ``--out-dir``, default
``experiments/p3_specialization/diagnostics/results/issue271_random_mlp_search/``)
---------------------------------------------------------------------------------

- ``results_<protocol>.json``: per-seed dict
  ``{seed: {mean_per_step, std_per_step, per_episode: [...]}}`` for all
  ``--seeds`` bundles.
- ``top_candidates_<protocol>.json``: sorted top-``--top-frac`` seeds with full
  per-episode arrays from the initial 20-episode pass.
- ``stability_<protocol>.json``: for each top candidate, the re-eval
  ``mean_per_step`` and ``std_per_step`` over ``--restability-episodes``
  fresh episodes (different episode seeds than the initial pass) plus a
  bootstrap 95% CI.
- ``distribution_<protocol>.png``: histogram of all per-seed means, with
  vertical lines for uniform-random (-87.72), PPO best (~-75), specialist
  (-22.07), and the stability-confirmed top-1% mean.
- ``summary.md``: combined verdict block covering both protocols; reports
  gap_closed at top-1% (stability-confirmed) and applies the verdict table
  from the issue body.

Reference values (per-step team reward on ``minimal_specialization``,
4 agents; see ``SCENARIO_CITED_VALUES`` in ``random_baseline.py`` and PR
#244 / #243):

| Policy class | per-step team reward |
| --- | --- |
| Uniform-random action sampling | -87.72 |
| Trained IPPO (best of 5 interventions) | ~-75 (gap_closed ≈ 0.18) |
| Specialist | -22.07 |

Denominator: ``specialist - random = 65.65``.

Usage
-----

::

    # smoke test (~30 sec on laptop)
    uv run python experiments/p3_specialization/diagnostics/random_mlp_search.py \\
        --seeds 10 --episodes-per-seed 5 --restability-episodes 10 \\
        --protocol independent

    # full run (run on COMPUTE_HOST_PRIMARY in tmux; ~15-30 min)
    uv run python experiments/p3_specialization/diagnostics/random_mlp_search.py \\
        --seeds 1000 --episodes-per-seed 20 --restability-episodes 200 \\
        --protocol both
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# These imports are also used by the worker, but workers re-import them under
# the ``spawn`` context (fresh interpreter). Keep them top-level for the
# orchestrator process.

# ----------------------------------------------------------------------------
# Constants — kept in sync with random_baseline.py:71-75
# ----------------------------------------------------------------------------
HIDDEN_SIZE = 64
NUM_AGENTS = 4
ACTION_DIMS = [10, 2, 2]  # [house, mode, signal] (issue #235)
SCENARIO_NAME = "minimal_specialization"

# Reference values for verdict (per-step team reward on minimal_specialization).
# Sources: random_baseline.SCENARIO_CITED_VALUES['minimal_specialization'] for
# random; experiments/p3_specialization/analyze_261_calibration.py:53-54 for
# specialist and denominator; PPO best ≈ -75 across the 5 tier-3 interventions.
REF_RANDOM = -87.72
REF_PPO_BEST = -75.0  # approximate PPO ceiling across the 5 interventions
REF_SPECIALIST = -22.07
REF_DENOMINATOR = REF_SPECIALIST - REF_RANDOM  # 65.65


def gap_closed(team_per_step: float) -> float:
    """Fraction of the specialist−random gap that ``team_per_step`` closes."""
    return float((team_per_step - REF_RANDOM) / REF_DENOMINATOR)


# ----------------------------------------------------------------------------
# Worker — evaluates a single (seed → episodes) bundle.
# ----------------------------------------------------------------------------
def _evaluate_seed_worker(args: Tuple[int, int, str, int]) -> Dict[str, object]:
    """Evaluate one random-init MLP bundle on ``minimal_specialization``.

    Args:
        args: tuple of (seed, n_episodes, protocol, episode_seed_base). The
            episode_seed_base offsets the per-episode env seeds so the
            restability pass can use disjoint episode draws.

    Returns:
        ``{"seed": int, "protocol": str, "per_episode": list[float],
        "mean_per_step": float, "std_per_step": float, "lengths": list[int],
        "entropy": float}``.
    """
    seed, n_episodes, protocol, episode_seed_base = args

    # Pin per-worker thread count so multiprocessing doesn't oversubscribe.
    # Imports are inside the worker so spawn doesn't drag the orchestrator's
    # torch state.
    import torch  # noqa: F401  (re-imported intentionally for spawn workers)

    torch.set_num_threads(1)

    from bucket_brigade.envs.bucket_brigade_env import (  # noqa: F811
        BucketBrigadeEnv as _Env,
    )
    from bucket_brigade.envs.scenarios_generated import (  # noqa: F811
        get_scenario_by_name as _get_scenario,
    )
    from bucket_brigade.training.joint_trainer import (  # noqa: F811
        flatten_dict_obs as _flatten,
    )
    from bucket_brigade.training.networks import PolicyNetwork as _Policy

    scenario = _get_scenario(SCENARIO_NAME, num_agents=NUM_AGENTS)
    env = _Env(scenario=scenario)

    # obs_dim from a probe reset; matches random_baseline.py:352-355.
    obs_dim = _flatten(env.reset(seed=0), agent_id=0, num_agents=NUM_AGENTS).shape[0]

    # Build per-agent policies under the chosen init protocol.
    policies: List[_Policy] = []
    if protocol == "shared":
        # All agents instantiated from the same RNG state, in sequence.
        torch.manual_seed(seed)
        np.random.seed(seed)
        for _ in range(NUM_AGENTS):
            policies.append(
                _Policy(
                    obs_dim=obs_dim, action_dims=ACTION_DIMS, hidden_size=HIDDEN_SIZE
                )
            )
    elif protocol == "independent":
        # Distinct seed per agent (seed * 4 + agent_id) → independent init draws.
        for i in range(NUM_AGENTS):
            agent_seed = seed * 4 + i
            torch.manual_seed(agent_seed)
            np.random.seed(agent_seed)
            policies.append(
                _Policy(
                    obs_dim=obs_dim, action_dims=ACTION_DIMS, hidden_size=HIDDEN_SIZE
                )
            )
    else:
        raise ValueError(f"Unknown protocol: {protocol!r}")

    for p in policies:
        p.eval()

    # ------------------------------------------------------------------
    # Per-episode loop — mirrors random_baseline.run_mlp_episode.
    # ------------------------------------------------------------------
    per_step: List[float] = []
    lengths: List[int] = []
    entropy_samples: List[float] = []  # iter-0 action-dist entropy sanity check.

    for ep in range(n_episodes):
        ep_seed = episode_seed_base + seed * 100_000 + ep
        obs_dict = env.reset(seed=ep_seed)
        obs_rows = np.stack(
            [
                _flatten(obs_dict, agent_id=i, num_agents=NUM_AGENTS)
                for i in range(NUM_AGENTS)
            ],
            axis=0,
        )
        total_reward = 0.0
        first_step = True
        while not env.done:
            obs_t = torch.from_numpy(obs_rows)
            joint_action = np.zeros((NUM_AGENTS, len(ACTION_DIMS)), dtype=np.int64)
            with torch.no_grad():
                for i, policy in enumerate(policies):
                    a, _, ent, _ = policy.get_action_and_value(obs_t[i : i + 1])
                    joint_action[i] = a[0].cpu().numpy()
                    if first_step:
                        entropy_samples.append(float(ent.item()))
            next_obs_dict, rewards, _, _ = env.step(joint_action)
            total_reward += float(rewards.sum())
            first_step = False
            if not env.done:
                obs_rows = np.stack(
                    [
                        _flatten(next_obs_dict, agent_id=i, num_agents=NUM_AGENTS)
                        for i in range(NUM_AGENTS)
                    ],
                    axis=0,
                )
        nights = int(env.night)
        per_step.append(total_reward / nights)
        lengths.append(nights)

    per_step_arr = np.asarray(per_step, dtype=np.float64)
    return {
        "seed": int(seed),
        "protocol": protocol,
        "per_episode": [float(x) for x in per_step_arr],
        "mean_per_step": float(per_step_arr.mean()),
        "std_per_step": float(per_step_arr.std(ddof=1))
        if len(per_step_arr) > 1
        else 0.0,
        "lengths": lengths,
        "entropy": float(np.mean(entropy_samples)) if entropy_samples else float("nan"),
    }


# ----------------------------------------------------------------------------
# Driver helpers
# ----------------------------------------------------------------------------
def _bootstrap_ci(
    arr: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05
) -> Tuple[float, float]:
    rng = np.random.default_rng(0)
    boots = np.empty(n_boot, dtype=np.float64)
    n = len(arr)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = arr[idx].mean()
    return (
        float(np.percentile(boots, 100 * alpha / 2)),
        float(np.percentile(boots, 100 * (1 - alpha / 2))),
    )


def _percentiles(arr: np.ndarray) -> Dict[str, float]:
    return {
        "p01": float(np.percentile(arr, 1)),
        "p05": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def _run_protocol(
    protocol: str,
    seeds: Sequence[int],
    episodes_per_seed: int,
    restability_episodes: int,
    top_frac: float,
    out_dir: Path,
    n_workers: int,
) -> Dict[str, object]:
    """Execute the full best-of-N + stability re-eval for one protocol."""
    print(f"\n=== Protocol: {protocol} ===")
    print(
        f"Phase 1: {len(seeds)} seeds x {episodes_per_seed} episodes "
        f"with {n_workers} workers"
    )

    ctx = mp.get_context("spawn")
    # Phase 1: initial best-of-N. Use episode_seed_base=0 here.
    phase1_args = [(int(s), episodes_per_seed, protocol, 0) for s in seeds]
    with ctx.Pool(processes=n_workers) as pool:
        phase1_results: List[Dict[str, object]] = []
        last_progress = 0
        for idx, res in enumerate(
            pool.imap_unordered(_evaluate_seed_worker, phase1_args), 1
        ):
            phase1_results.append(res)
            # Print a progress dot every ~5% of total work.
            progress_pct = int(100 * idx / len(seeds))
            if progress_pct >= last_progress + 5:
                last_progress = progress_pct
                print(
                    f"  [{protocol}] phase-1 progress: {idx}/{len(seeds)} "
                    f"({progress_pct}%)"
                )

    # Sort by mean_per_step descending (best first).
    phase1_results.sort(key=lambda r: r["mean_per_step"], reverse=True)
    means = np.asarray([r["mean_per_step"] for r in phase1_results])
    n_top = max(1, int(round(top_frac * len(phase1_results))))
    top = phase1_results[:n_top]

    print(
        f"Phase 1 complete. Population mean={means.mean():.2f}, std={means.std():.2f}"
    )
    print(
        f"Top-{int(top_frac * 100)}% threshold: {top[-1]['mean_per_step']:.2f}, n_top={n_top}"
    )
    print(f"Best phase-1 seed: {top[0]['seed']} mean={top[0]['mean_per_step']:.2f}")

    # Phase 2: noise-stability re-eval over top-k seeds with K episodes each.
    print(
        f"Phase 2: re-eval top-{n_top} seeds with {restability_episodes} episodes "
        f"(disjoint episode-seed base = 7_000_000)"
    )
    # Use a disjoint episode_seed_base so the restability draws are independent
    # of phase-1 draws (no shared per-episode env seeds).
    phase2_args = [
        (int(r["seed"]), restability_episodes, protocol, 7_000_000) for r in top
    ]
    with ctx.Pool(processes=n_workers) as pool:
        phase2_results: List[Dict[str, object]] = []
        last_progress = 0
        for idx, res in enumerate(
            pool.imap_unordered(_evaluate_seed_worker, phase2_args), 1
        ):
            phase2_results.append(res)
            progress_pct = int(100 * idx / max(1, len(phase2_args)))
            if progress_pct >= last_progress + 20:
                last_progress = progress_pct
                print(
                    f"  [{protocol}] phase-2 progress: {idx}/{len(phase2_args)} "
                    f"({progress_pct}%)"
                )

    # Attach bootstrap CI to each stability result.
    for r in phase2_results:
        arr = np.asarray(r["per_episode"], dtype=np.float64)
        lo, hi = _bootstrap_ci(arr)
        r["ci95_lo"] = lo
        r["ci95_hi"] = hi
    phase2_results.sort(key=lambda r: r["mean_per_step"], reverse=True)

    # ------------------------------------------------------------------
    # Persist artifacts.
    # ------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)

    # results_<protocol>.json (compact: drop per_episode for the full 1000;
    # keep summary fields for analysis).
    results_path = out_dir / f"results_{protocol}.json"
    with open(results_path, "w") as f:
        json.dump(
            [
                {
                    "seed": r["seed"],
                    "mean_per_step": r["mean_per_step"],
                    "std_per_step": r["std_per_step"],
                    "entropy": r["entropy"],
                }
                for r in phase1_results
            ],
            f,
            indent=2,
        )
    print(f"Wrote {results_path}")

    # top_candidates_<protocol>.json — full per-episode arrays for the top set.
    top_path = out_dir / f"top_candidates_{protocol}.json"
    with open(top_path, "w") as f:
        json.dump(top, f, indent=2)
    print(f"Wrote {top_path}")

    # stability_<protocol>.json — restability re-eval results.
    stability_path = out_dir / f"stability_{protocol}.json"
    with open(stability_path, "w") as f:
        json.dump(phase2_results, f, indent=2)
    print(f"Wrote {stability_path}")

    # Histogram.
    _plot_distribution(means, top, phase2_results, protocol, out_dir)

    # ------------------------------------------------------------------
    # Aggregate stats for the verdict.
    # ------------------------------------------------------------------
    top_means_p1 = np.asarray([r["mean_per_step"] for r in top])
    top_means_p2 = np.asarray([r["mean_per_step"] for r in phase2_results])
    top_p1_mean = float(top_means_p1.mean())
    top_p2_mean = float(top_means_p2.mean())
    top_p1_max = float(top_means_p1.max())
    top_p2_max = float(top_means_p2.max())

    # Best stability-confirmed seed: max mean across phase-2 re-evals.
    best_p2 = max(phase2_results, key=lambda r: r["mean_per_step"])

    population_pcts = _percentiles(means)

    return {
        "protocol": protocol,
        "n_seeds": int(len(phase1_results)),
        "episodes_per_seed_phase1": int(episodes_per_seed),
        "episodes_per_seed_phase2": int(restability_episodes),
        "n_top": int(n_top),
        "population_mean": float(means.mean()),
        "population_std": float(means.std(ddof=1)) if len(means) > 1 else 0.0,
        "population_percentiles": population_pcts,
        "top_p1_mean": top_p1_mean,
        "top_p2_mean": top_p2_mean,
        "top_p1_max": top_p1_max,
        "top_p2_max": top_p2_max,
        "top_p1_gap_closed": gap_closed(top_p1_mean),
        "top_p2_gap_closed": gap_closed(top_p2_mean),
        "top_p2_max_gap_closed": gap_closed(top_p2_max),
        "best_phase2_seed": int(best_p2["seed"]),
        "best_phase2_mean": float(best_p2["mean_per_step"]),
        "best_phase2_ci": (float(best_p2["ci95_lo"]), float(best_p2["ci95_hi"])),
        "mean_entropy": float(np.nanmean([r["entropy"] for r in phase1_results])),
    }


def _plot_distribution(
    population_means: np.ndarray,
    top_phase1: Sequence[Dict[str, object]],
    phase2_results: Sequence[Dict[str, object]],
    protocol: str,
    out_dir: Path,
) -> None:
    """Histogram of per-seed mean rewards + reference lines."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping histogram")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(population_means, bins=40, color="steelblue", alpha=0.7, edgecolor="white")

    # Reference lines.
    ax.axvline(
        REF_RANDOM, color="grey", linestyle="--", label=f"random ({REF_RANDOM:.2f})"
    )
    ax.axvline(
        REF_PPO_BEST,
        color="orange",
        linestyle="--",
        label=f"PPO best (~{REF_PPO_BEST:.0f})",
    )
    ax.axvline(
        REF_SPECIALIST,
        color="green",
        linestyle="--",
        label=f"specialist ({REF_SPECIALIST:.2f})",
    )

    top_p1_mean = float(np.mean([r["mean_per_step"] for r in top_phase1]))
    ax.axvline(
        top_p1_mean,
        color="red",
        linestyle="-",
        linewidth=1.5,
        label=f"top-{len(top_phase1)} phase-1 mean ({top_p1_mean:.2f})",
    )

    top_p2_mean = float(np.mean([r["mean_per_step"] for r in phase2_results]))
    ax.axvline(
        top_p2_mean,
        color="purple",
        linestyle="-",
        linewidth=1.5,
        label=f"top phase-2 mean ({top_p2_mean:.2f})",
    )

    ax.set_xlabel("per-step team reward (mean over phase-1 episodes)")
    ax.set_ylabel("# seeds")
    ax.set_title(
        f"Issue #271 — random-init MLP best-of-N on minimal_specialization "
        f"({protocol}-init, N={len(population_means)})"
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    plot_path = out_dir / f"distribution_{protocol}.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {plot_path}")


def _classify_verdict(top_p2_mean: float) -> Tuple[str, str]:
    """Apply the verdict table from the issue body.

    Returns (tier_name, interpretation).
    """
    g = gap_closed(top_p2_mean)
    if top_p2_mean >= -30.0 or g >= 0.88:
        return (
            "stunning_near_specialist",
            "Random init lives near specialist. PPO's failure becomes mysterious.",
        )
    if top_p2_mean >= -55.0 or g >= 0.49:
        return (
            "anti_attractor_confirmed",
            "Best-of-N >> PPO. PPO is doing worse than naive random search → active anti-attractor.",
        )
    if -75.0 <= top_p2_mean < -55.0 or 0.20 <= g < 0.49:
        return (
            "basin_trap_consistent",
            "Best-of-N ≈ PPO ceiling. Basin trap consistent; specialist basin not proven reachable.",
        )
    return (
        "random_play_basin",
        "Random MLPs ≈ uniform random play. PPO sits in same basin random init lives in.",
    )


def _write_summary(
    per_protocol_summaries: Dict[str, Dict[str, object]],
    out_dir: Path,
    args_repr: str,
) -> None:
    """Combined summary.md with both protocols + verdict block."""
    lines: List[str] = []
    lines.append(
        "# Issue #271 — Random-init MLP best-of-N on `minimal_specialization`\n"
    )
    lines.append(
        "Question: does naive best-of-N random search over policy networks beat "
        "trained PPO? If yes, the gradient is actively misleading → active "
        "anti-attractor.\n"
    )
    lines.append(f"Invocation: `{args_repr}`\n")
    lines.append(
        f"\n## Reference values\n\n"
        f"- Uniform-random per-step team reward: **{REF_RANDOM:.2f}** "
        "(from `random_baseline.SCENARIO_CITED_VALUES['minimal_specialization']`)\n"
        f"- PPO best across 5 tier-3 interventions: **~{REF_PPO_BEST:.0f}** "
        "(gap_closed ≈ 0.18)\n"
        f"- Specialist: **{REF_SPECIALIST:.2f}** (from `issue199_baselines.json`)\n"
        f"- Denominator (specialist − random): **{REF_DENOMINATOR:.2f}**\n"
    )

    for protocol, s in per_protocol_summaries.items():
        tier, interp = _classify_verdict(s["top_p2_mean"])
        lines.append(f"\n## Protocol: `{protocol}`\n")
        lines.append(
            f"- n_seeds = {s['n_seeds']}, "
            f"phase-1 episodes/seed = {s['episodes_per_seed_phase1']}, "
            f"phase-2 episodes/seed = {s['episodes_per_seed_phase2']}\n"
            f"- Population mean = **{s['population_mean']:.2f}** "
            f"(std = {s['population_std']:.2f})\n"
        )
        p = s["population_percentiles"]
        lines.append(
            "- Percentiles (per-step team reward):\n"
            f"  - p01 = {p['p01']:.2f}, p05 = {p['p05']:.2f}, p10 = {p['p10']:.2f}\n"
            f"  - p50 = {p['p50']:.2f}\n"
            f"  - p90 = {p['p90']:.2f}, p95 = {p['p95']:.2f}, p99 = {p['p99']:.2f}\n"
        )
        lines.append(
            f"- Top-{s['n_top']} phase-1 mean = **{s['top_p1_mean']:.2f}** "
            f"(gap_closed = {s['top_p1_gap_closed']:.3f})\n"
            f"- Top-{s['n_top']} phase-2 (stability-re-eval) mean = "
            f"**{s['top_p2_mean']:.2f}** "
            f"(gap_closed = {s['top_p2_gap_closed']:.3f})\n"
            f"- Best phase-2 seed: {s['best_phase2_seed']} "
            f"→ mean = {s['best_phase2_mean']:.2f} "
            f"(gap_closed = {gap_closed(s['best_phase2_mean']):.3f}), "
            f"95% CI = [{s['best_phase2_ci'][0]:.2f}, {s['best_phase2_ci'][1]:.2f}]\n"
            f"- Phase-1 → phase-2 drift (mean): "
            f"{s['top_p2_mean'] - s['top_p1_mean']:+.2f}\n"
            f"- Mean iter-0 action-distribution entropy: {s['mean_entropy']:.3f} nats "
            f"(uniform 10×2×2 ≈ {np.log(10) + 2 * np.log(2):.3f} nats)\n"
        )
        lines.append(f"\n### Verdict — `{protocol}`: `{tier}`\n\n{interp}\n")

    lines.append("\n## Verdict table (from issue body, restated)\n")
    lines.append(
        "| Best-of-N outcome (top-1% phase-2 mean) | gap_closed | Interpretation |\n"
        "| --- | --- | --- |\n"
        "| ≥ -55 (clearly above PPO ceiling) | ≥ 0.49 | "
        "**Anti-attractor confirmed.** PPO is worse than naive search. |\n"
        "| -75 to -55 | 0.20 to 0.49 | "
        "PPO ≈ best-of-N random. Basin trap consistent. |\n"
        "| -85 to -75 | < 0.20 | "
        "Random MLPs ≈ uniform random play; PPO sits in same basin. |\n"
        "| ≥ -30 (near specialist) | ≥ 0.88 | "
        "Stunning — random init lives near specialist. |\n"
    )

    summary_path = out_dir / "summary.md"
    with open(summary_path, "w") as f:
        f.write("".join(lines))
    print(f"Wrote {summary_path}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=1000, help="Number of seed bundles.")
    ap.add_argument(
        "--episodes-per-seed",
        type=int,
        default=20,
        help="Phase-1 episodes per seed (the initial sweep).",
    )
    ap.add_argument(
        "--restability-episodes",
        type=int,
        default=200,
        help="Phase-2 episodes per top candidate (re-eval).",
    )
    ap.add_argument(
        "--top-frac",
        type=float,
        default=0.01,
        help="Top fraction of seeds to re-evaluate in phase 2.",
    )
    ap.add_argument(
        "--protocol",
        choices=["independent", "shared", "both"],
        default="both",
        help="Per-agent init protocol. 'both' runs each in sequence.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "experiments/p3_specialization/diagnostics/results/issue271_random_mlp_search"
        ),
        help="Output directory (created if missing).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="Number of multiprocessing workers.",
    )
    ap.add_argument(
        "--start-seed",
        type=int,
        default=1,
        help="First seed in the sweep (seeds = start_seed..start_seed+N-1).",
    )
    args = ap.parse_args()

    args_repr = " ".join(sys.argv)

    seeds = list(range(args.start_seed, args.start_seed + args.seeds))

    protocols: List[str]
    if args.protocol == "both":
        protocols = ["independent", "shared"]
    else:
        protocols = [args.protocol]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_protocol_summaries: Dict[str, Dict[str, object]] = {}
    for protocol in protocols:
        per_protocol_summaries[protocol] = _run_protocol(
            protocol=protocol,
            seeds=seeds,
            episodes_per_seed=args.episodes_per_seed,
            restability_episodes=args.restability_episodes,
            top_frac=args.top_frac,
            out_dir=args.out_dir,
            n_workers=args.workers,
        )

    _write_summary(per_protocol_summaries, args.out_dir, args_repr)

    # Print final verdicts to stdout for tmux log capture.
    print("\n=== Final verdicts ===")
    for protocol, s in per_protocol_summaries.items():
        tier, interp = _classify_verdict(s["top_p2_mean"])
        print(
            f"  [{protocol}] top-{s['n_top']} phase-2 mean = {s['top_p2_mean']:.2f} "
            f"(gap_closed = {s['top_p2_gap_closed']:.3f}) → {tier}"
        )
        print(f"    {interp}")


if __name__ == "__main__":
    main()
