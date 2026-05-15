"""Analysis for the issue #183 P3 plateau Phase 3 long-horizon retest.

Reads per-cell ``metrics.json`` files under
``experiments/p3_specialization/runs/p3_183_phase3/<condition>/<scenario>/lambda_0e0/seed_<N>/``
(gitignored) and writes aggregated artifacts to
``experiments/p3_specialization/results_183_phase3/`` (committed):

- ``summary.json``            — per (condition, scenario) mean/CI/slope/gradient-ratio dict
- ``summary.md``              — markdown table mirroring the notebook headline
- ``reward_trajectories.png`` — per-iter mean team reward; one panel per scenario,
                                with both Phase 2 (#174, 50 iters) and Phase 3
                                (#183, 500 iters) trajectories overlaid where available.
- ``gradient_ratios.png``     — ``value_coef * value_loss / |policy_loss|`` per
                                (condition, scenario) (log y), one panel per scenario.

The 3 conditions are the L1-anchored configurations from #174:

    L1_norm  (normalize_returns=True, value_coef=0.5,  entropy_coef=0.01)
    L1L2     (normalize_returns=True, value_coef=0.05, entropy_coef=0.01)
    L1L3     (normalize_returns=True, value_coef=0.5,  entropy_coef=0.10)

Each (condition, scenario) cell has 5 seeds (42..46). All cells use lambda_red=0,
num_iterations=500, rollout_steps=2048.

Random per-step team-reward baseline = 308 (reported in #145).
Acceptance bar (#183) = CI lower bound > 320 at iter 499.

Usage::

    uv run python experiments/p3_specialization/analyze_183.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RAND_BASELINE = 308.0
ACCEPTANCE_BAR = 320.0
SEEDS = [42, 43, 44, 45, 46]
NUM_AGENTS = 4
SCENARIOS = ["default", "chain_reaction"]
FINAL_ITER = 499  # 0-based index of last iter in a 500-iter run

# (name, normalize_returns, value_coef, entropy_coef)
CONDITIONS: List[Tuple[str, bool, float, float]] = [
    ("L1_norm", True, 0.5, 0.01),
    ("L1L2", True, 0.05, 0.01),
    ("L1L3", True, 0.5, 0.1),
]

# Optional: Phase 2 #174 results to overlay on the trajectory plot for visual
# continuity. We only overlay the same three L1-anchored conditions that are
# being retested here. Falls back gracefully if the Phase 2 artifact is missing.
PHASE2_SUMMARY = Path("experiments/p3_specialization/results_174_ablation/summary.json")


def per_agent_mean(row: dict, key: str) -> float:
    return float(np.mean([row[f"{key}/agent_{i}"] for i in range(NUM_AGENTS)]))


def per_agent_abs_mean(row: dict, key: str) -> float:
    return float(np.mean([abs(row[f"{key}/agent_{i}"]) for i in range(NUM_AGENTS)]))


def gradient_ratio(row: dict, vc: float) -> float:
    v = per_agent_mean(row, "value_loss")
    p = max(per_agent_abs_mean(row, "policy_loss"), 1e-12)
    return vc * v / p


def bootstrap_ci(
    x: np.ndarray, n: int = 10000, alpha: float = 0.05, seed: int = 0
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _linear_slope(reward_mean_traj: np.ndarray, start: int, end: int) -> float:
    """Linear regression slope of mean-trajectory over [start, end)."""
    end = min(end, reward_mean_traj.shape[0])
    if end - start < 2:
        return float("nan")
    x = np.arange(start, end)
    return float(np.polyfit(x, reward_mean_traj[start:end], 1)[0])


def load_cell(
    base: Path, condition: str, scenario: str, vc: float
) -> Optional[Dict]:
    """Load all seeds for a (condition, scenario) cell. Returns None if any seed
    is missing on disk (so we can surface partial-sweep state clearly)."""
    rewards: List[List[float]] = []  # [seed][iter]
    entropies_final: List[float] = []
    grad_ratio_iter0: List[float] = []
    grad_ratio_final: List[float] = []
    grad_ratio_traj: List[List[float]] = []  # [seed][iter]
    for s in SEEDS:
        mfile = (
            base / condition / scenario / "lambda_0e0" / f"seed_{s}" / "metrics.json"
        )
        if not mfile.exists():
            print(f"  MISSING: {mfile}")
            return None
        m = json.loads(mfile.read_text())
        rewards.append([row["mean_step_reward_team"] for row in m])
        entropies_final.append(m[-1]["action_entropy/mean"])
        grad_ratio_iter0.append(gradient_ratio(m[0], vc))
        grad_ratio_final.append(gradient_ratio(m[-1], vc))
        grad_ratio_traj.append([gradient_ratio(row, vc) for row in m])

    R = np.array(rewards)
    r_final = R[:, -1]
    lo, hi = bootstrap_ci(r_final)
    mean_traj = R.mean(axis=0)

    # Mid-horizon slope on the same window as #174 (iters 25-50) for direct
    # comparison; plus the long-horizon slope (iters 250-500) requested by
    # the issue's secondary check.
    slope_25_50 = _linear_slope(mean_traj, 25, 50)
    slope_250_500 = _linear_slope(mean_traj, 250, mean_traj.shape[0])
    # Also report the broader "later half" slope to characterize plateau drift.
    half = mean_traj.shape[0] // 2
    slope_late = _linear_slope(mean_traj, half, mean_traj.shape[0])

    return {
        "num_iters_observed": int(R.shape[1]),
        "reward_per_seed": r_final.tolist(),
        "reward_mean": float(r_final.mean()),
        "reward_ci": [lo, hi],
        "crosses_bar": lo > ACCEPTANCE_BAR,
        "crosses_random": lo > RAND_BASELINE,
        "delta_vs_random": float(r_final.mean() - RAND_BASELINE),
        "reward_trajectory_mean": mean_traj.tolist(),
        "reward_slope_iters_25_to_50": slope_25_50,
        "reward_slope_iters_250_to_500": slope_250_500,
        "reward_slope_late_half": slope_late,
        "entropy_final_mean": float(np.mean(entropies_final)),
        "gradient_ratio_iter0_mean": float(np.mean(grad_ratio_iter0)),
        "gradient_ratio_final_mean": float(np.mean(grad_ratio_final)),
        "gradient_ratio_trajectory_mean": np.array(grad_ratio_traj)
        .mean(axis=0)
        .tolist(),
    }


def write_markdown_table(results: Dict[str, Dict], out: Path) -> None:
    lines = [
        "# Issue #183 — P3 plateau Phase 3 long-horizon summary",
        "",
        f"Random baseline = {RAND_BASELINE}; acceptance bar = CI lower bound > {ACCEPTANCE_BAR}.",
        f"Final-iter target = iter {FINAL_ITER} (0-based); 5 seeds; bootstrap 95% CI.",
        "",
        "| condition | scenario | reward iter_final (95% CI) | Δ vs rand | crosses bar | "
        "ent_final | vc·v_loss/|p_loss| 0→final | slope 25→50 | slope 250→500 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for cond_name, _, _, _ in CONDITIONS:
        for scenario in SCENARIOS:
            key = f"{cond_name}__{scenario}"
            r = results.get(key)
            if r is None:
                lines.append(
                    f"| {cond_name} | {scenario} | n/a (incomplete) | – | – | – | – | – | – |"
                )
                continue
            lo, hi = r["reward_ci"]
            crosses = "**YES**" if r["crosses_bar"] else "no"
            lines.append(
                f"| {cond_name} | {scenario} | "
                f"{r['reward_mean']:.2f} [{lo:.2f}, {hi:.2f}] | "
                f"{r['delta_vs_random']:+.2f} | {crosses} | "
                f"{r['entropy_final_mean']:.3f} | "
                f"{r['gradient_ratio_iter0_mean']:.1e} → {r['gradient_ratio_final_mean']:.1e} | "
                f"{r['reward_slope_iters_25_to_50']:+.3f} | "
                f"{r['reward_slope_iters_250_to_500']:+.3f} |"
            )

    # Verdict footer
    crossed_any = any(
        v is not None and v.get("crosses_bar", False) for v in results.values()
    )
    lines.append("")
    if crossed_any:
        winners = [
            k for k, v in results.items() if v is not None and v.get("crosses_bar")
        ]
        lines.append(
            f"**Verdict: ACCEPT.** Cells crossing the {ACCEPTANCE_BAR} bar: "
            + ", ".join(sorted(winners))
        )
    else:
        lines.append(
            f"**Verdict: REJECT.** No (condition, scenario) cell crosses the "
            f"{ACCEPTANCE_BAR} bar at iter {FINAL_ITER}."
        )
    out.write_text("\n".join(lines) + "\n")


def _load_phase2_overlay() -> Dict[str, List[float]]:
    """Return Phase 2 mean reward trajectories for the same three conditions
    on scenario=default (the only one Phase 2 covered). Empty dict on miss."""
    if not PHASE2_SUMMARY.exists():
        return {}
    try:
        p2 = json.loads(PHASE2_SUMMARY.read_text())
    except json.JSONDecodeError:
        return {}
    return {
        name: p2[name]["reward_trajectory_mean"]
        for name, _, _, _ in CONDITIONS
        if name in p2 and "reward_trajectory_mean" in p2[name]
    }


def plot_reward_trajectories(results: Dict[str, Dict], out: Path) -> None:
    """One subplot per scenario, with Phase 3 trajectories per condition and
    (on the default scenario) Phase 2 trajectories overlaid for continuity."""
    phase2 = _load_phase2_overlay()
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(12, 5), sharey=True)
    if len(SCENARIOS) == 1:
        axes = [axes]  # type: ignore[assignment]

    for ax, scenario in zip(axes, SCENARIOS):
        for cond_name, _, _, _ in CONDITIONS:
            key = f"{cond_name}__{scenario}"
            r = results.get(key)
            if r is None:
                continue
            traj = r["reward_trajectory_mean"]
            ax.plot(traj, label=f"{cond_name} (P3)", linewidth=1.6)

        if scenario == "default" and phase2:
            for cond_name, traj in phase2.items():
                ax.plot(
                    traj,
                    label=f"{cond_name} (P2 #174)",
                    linewidth=1.0,
                    linestyle="--",
                    alpha=0.6,
                )

        ax.axhline(
            RAND_BASELINE,
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"random ({RAND_BASELINE:.0f})",
        )
        ax.axhline(
            ACCEPTANCE_BAR,
            color="green",
            linestyle=":",
            linewidth=1,
            label=f"bar ({ACCEPTANCE_BAR:.0f})",
        )
        ax.set_xlabel("iteration")
        ax.set_title(f"scenario = {scenario}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=7)

    axes[0].set_ylabel("mean per-step team reward (5 seeds)")
    fig.suptitle("Issue #183 (P3 Phase 3): reward trajectories, λ=0, 500 iters")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_gradient_ratios(results: Dict[str, Dict], out: Path) -> None:
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(12, 5), sharey=True)
    if len(SCENARIOS) == 1:
        axes = [axes]  # type: ignore[assignment]

    for ax, scenario in zip(axes, SCENARIOS):
        for cond_name, _, _, _ in CONDITIONS:
            key = f"{cond_name}__{scenario}"
            r = results.get(key)
            if r is None:
                continue
            ax.plot(r["gradient_ratio_trajectory_mean"], label=cond_name, linewidth=1.5)
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_title(f"scenario = {scenario}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[0].set_ylabel("value_coef · value_loss / |policy_loss|  (log)")
    fig.suptitle(
        "Issue #183 (P3 Phase 3): PPO gradient-term ratio per (condition, scenario)"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    cells = Path("experiments/p3_specialization/runs/p3_183_phase3")
    out = Path("experiments/p3_specialization/results_183_phase3")
    if not cells.exists():
        raise SystemExit(f"missing sweep output: {cells}")
    out.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict] = {}
    for cond_name, norm, vc, ec in CONDITIONS:
        for scenario in SCENARIOS:
            key = f"{cond_name}__{scenario}"
            cell = load_cell(cells, cond_name, scenario, vc)
            if cell is None:
                print(f"skipping {key}: missing seeds")
                continue
            cell["config"] = {
                "normalize_returns": norm,
                "value_coef": vc,
                "entropy_coef": ec,
                "scenario": scenario,
            }
            results[key] = cell

    (out / "summary.json").write_text(json.dumps(results, indent=2))
    write_markdown_table(results, out / "summary.md")
    plot_reward_trajectories(results, out / "reward_trajectories.png")
    plot_gradient_ratios(results, out / "gradient_ratios.png")

    print(f"wrote {out}/summary.json")
    print(f"wrote {out}/summary.md")
    print(f"wrote {out}/reward_trajectories.png")
    print(f"wrote {out}/gradient_ratios.png")


if __name__ == "__main__":
    main()
