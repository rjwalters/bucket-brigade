"""Analysis for the issue #174 P3 plateau ablation sweep.

Reads per-cell ``metrics.json`` files under
``experiments/p3_specialization/runs/p3_174_ablation/<condition>/default/lambda_0e0/seed_<N>/``
(gitignored) and writes aggregated artifacts to
``experiments/p3_specialization/results_174_ablation/`` (committed):

- ``summary.json``            — per-condition mean/CI/slope/gradient-ratio dict
- ``summary.md``              — markdown table mirroring the notebook headline
- ``reward_trajectories.png`` — per-iter mean team reward per condition
- ``gradient_ratios.png``     — ``value_coef * value_loss / |policy_loss|`` per condition (log y)

The 6 conditions are the (normalize_returns, value_coef, entropy_coef) tuples
specified in issue #174:

    baseline    (False, 0.5,  0.01)
    L1_norm     (True,  0.5,  0.01)
    L2_low_vf   (False, 0.05, 0.01)
    L3_high_ent (False, 0.5,  0.1)
    L1L2        (True,  0.05, 0.01)
    L1L3        (True,  0.5,  0.1)

Each condition has 5 seeds (42..46). All cells use scenario=default,
lambda_red=0, num_iterations=50, rollout_steps=2048.

Random per-step team-reward baseline on default = 251.23 (re-derived in
issue #237 on post-#236 main at commit ``dffe1060``, n=1000 episodes,
seeds 42..46; 95% CI [244.86, 257.51]). PR #218's previous value of 247.58
measured the pre-#236 action space (``MultiDiscrete([10, 2])``) and is
therefore stale — PR #236 promoted the broadcast signal to a first-class
action dimension (``MultiDiscrete([10, 2, 2])``), and uniform-random play
now exercises a strictly larger joint-action manifold. The +3.65 shift
on ``default`` is the small-but-real consequence of teammate envelopes
observing independently-randomized signals from teammates.

Acceptance bar = CI upper bound of the re-derived random baseline (257.51).
"Crosses bar" therefore means "statistically distinguishable from random,"
not "exceeds an arbitrary buffer above a noisy single-sample baseline."
The original #174 acceptance bar of 320 (= 308 + 12) was derived from the
incorrect baseline; see issue #202 for the audit and policy decision.
Earlier audits (each superseded by the next): #218 (308 → 247.58 post-#197/#198),
#237 (247.58 → 251.23 post-#236).

Usage::

    uv run python experiments/p3_specialization/analyze_174.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RAND_BASELINE = 251.23  # post-#236 (issue #237); 95% CI [244.86, 257.51]
ACCEPTANCE_BAR = 257.51  # = CI upper bound of post-#236 baseline (issue #237)
SEEDS = [42, 43, 44, 45, 46]
NUM_AGENTS = 4

# (name, normalize_returns, value_coef, entropy_coef)
CONDITIONS: List[Tuple[str, bool, float, float]] = [
    ("baseline", False, 0.5, 0.01),
    ("L1_norm", True, 0.5, 0.01),
    ("L2_low_vf", False, 0.05, 0.01),
    ("L3_high_ent", False, 0.5, 0.1),
    ("L1L2", True, 0.05, 0.01),
    ("L1L3", True, 0.5, 0.1),
]


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


def load_condition(base: Path, name: str, vc: float) -> Dict:
    rewards: List[List[float]] = []  # [seed][iter]
    entropies_iter49: List[float] = []
    grad_ratio_iter0: List[float] = []
    grad_ratio_iter49: List[float] = []
    grad_ratio_traj: List[List[float]] = []  # [seed][iter]
    for s in SEEDS:
        mfile = base / name / "default" / "lambda_0e0" / f"seed_{s}" / "metrics.json"
        m = json.loads(mfile.read_text())
        rewards.append([row["mean_step_reward_team"] for row in m])
        entropies_iter49.append(m[-1]["action_entropy/mean"])
        grad_ratio_iter0.append(gradient_ratio(m[0], vc))
        grad_ratio_iter49.append(gradient_ratio(m[-1], vc))
        grad_ratio_traj.append([gradient_ratio(row, vc) for row in m])

    R = np.array(rewards)
    r49 = R[:, -1]
    lo, hi = bootstrap_ci(r49)
    x = np.arange(25, R.shape[1])
    slope = float(np.polyfit(x, R[:, 25:].mean(axis=0), 1)[0])
    return {
        "reward_per_seed": r49.tolist(),
        "reward_mean": float(r49.mean()),
        "reward_ci": [lo, hi],
        "crosses_bar": lo > ACCEPTANCE_BAR,
        "crosses_random": lo > RAND_BASELINE,
        "delta_vs_random": float(r49.mean() - RAND_BASELINE),
        "reward_trajectory_mean": R.mean(axis=0).tolist(),
        "reward_slope_iters_25_to_50": slope,
        "entropy_iter49_mean": float(np.mean(entropies_iter49)),
        "gradient_ratio_iter0_mean": float(np.mean(grad_ratio_iter0)),
        "gradient_ratio_iter49_mean": float(np.mean(grad_ratio_iter49)),
        "gradient_ratio_trajectory_mean": np.array(grad_ratio_traj)
        .mean(axis=0)
        .tolist(),
    }


def write_markdown_table(results: Dict[str, Dict], out: Path) -> None:
    lines = [
        "# Issue #174 — P3 plateau ablation summary",
        "",
        f"Random baseline = {RAND_BASELINE}; acceptance bar = CI lower bound > {ACCEPTANCE_BAR}.",
        "",
        "| condition | reward iter49 (95% CI) | Δ vs rand | crosses bar | ent49 | vc·v_loss/|p_loss| 0→49 | slope iter25→50 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, _, _, _ in CONDITIONS:
        r = results[name]
        lo, hi = r["reward_ci"]
        crosses = "**YES**" if r["crosses_bar"] else "no"
        lines.append(
            f"| {name} | {r['reward_mean']:.2f} [{lo:.2f}, {hi:.2f}] | {r['delta_vs_random']:+.2f} | {crosses} | "
            f"{r['entropy_iter49_mean']:.3f} | {r['gradient_ratio_iter0_mean']:.1e} → {r['gradient_ratio_iter49_mean']:.1e} | "
            f"{r['reward_slope_iters_25_to_50']:+.3f} |"
        )
    out.write_text("\n".join(lines) + "\n")


def plot_reward_trajectories(results: Dict[str, Dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, _, _, _ in CONDITIONS:
        traj = results[name]["reward_trajectory_mean"]
        ax.plot(traj, label=name, linewidth=1.5)
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
        label=f"acceptance bar ({ACCEPTANCE_BAR:.0f})",
    )
    ax.set_xlabel("iteration")
    ax.set_ylabel("mean per-step team reward (across 5 seeds)")
    ax.set_title("Issue #174 ablation: reward trajectories on default, λ=0")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_gradient_ratios(results: Dict[str, Dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, _, _, _ in CONDITIONS:
        traj = results[name]["gradient_ratio_trajectory_mean"]
        ax.plot(traj, label=name, linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("value_coef · value_loss / |policy_loss|  (log)")
    ax.set_title("Issue #174 ablation: PPO gradient-term ratio per condition")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    cells = Path("experiments/p3_specialization/runs/p3_174_ablation")
    out = Path("experiments/p3_specialization/results_174_ablation")
    if not cells.exists():
        raise SystemExit(f"missing sweep output: {cells}")
    out.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict] = {}
    for name, norm, vc, ec in CONDITIONS:
        results[name] = load_condition(cells, name, vc)
        results[name]["config"] = {
            "normalize_returns": norm,
            "value_coef": vc,
            "entropy_coef": ec,
        }

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
