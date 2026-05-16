"""Phase 1 plateau diagnostics for P3 specialization sweep.

This script loads existing ``metrics.json`` files under
``experiments/p3_specialization/runs/`` and produces a small set of
matplotlib plots that characterize the PPO learning plateau reported in
issue #145. No new training is run — this is a pure read-existing-data
exercise.

Outputs go to ``experiments/p3_specialization/diagnostics/``. Each plot
aggregates across seeds (per-iteration mean and inter-quartile shading)
for the ``lambda_red = 0`` cells in each scenario, which is the cleanest
read on whether PPO is learning anything at all.

The plots produced are:

- ``reward_vs_baseline.png``       per-iter team reward vs random baseline
- ``loss_decomposition.png``       policy vs value vs entropy contributions
- ``entropy_per_agent.png``        per-agent action distribution entropy
- ``value_loss_log.png``           value loss magnitude (log scale)
- ``loss_term_scales.png``         scaled loss terms (coefs applied) for comparison

These plots support the diagnosis in research_notebook/
2026-05-14_p3_specialization_results.md and the issue body.

Usage::

    uv run python experiments/p3_specialization/analyze_plateau.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np


# Hard-coded baselines (random-policy and heuristic per-step team reward).
# These are not in the metrics files; they were measured separately.
#
# The ``default`` random baseline was re-derived in PR #196 / issue #192 using
# ``diagnostics/random_baseline.py`` (n=1000 episodes, 5 seeds): 293.39 with
# 95% bootstrap CI [288.87, 297.78]. The previous value (308.0, from #145 at
# n=50) was an outlier on the right tail of a high-variance distribution and
# sits outside the n=1000 CI. See ``research_notebook/2026-05-15_h3_random_baseline.md``.
#
# The ``trivial_cooperation`` and ``chain_reaction`` random values, and the
# three ``heuristic`` values, share the same uncommitted #145 provenance and
# remain flagged for sibling re-derivation (see issue #202 follow-up).
BASELINES: Dict[str, Dict[str, float]] = {
    "trivial_cooperation": {"random": 400.0, "heuristic": 400.0},
    "default": {"random": 293.4, "heuristic": 307.0},
    "chain_reaction": {"random": 233.0, "heuristic": 226.0},
}

# Coefficients used in joint_trainer (default values from CellConfig).
# These are needed to compare loss term *contributions* (value_coef *
# value_loss, etc.) on the same axes.
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01


def _load_seed_metrics(seed_dir: Path) -> List[dict] | None:
    metrics_path = seed_dir / "metrics.json"
    config_path = seed_dir / "config.json"
    if not metrics_path.exists() or not config_path.exists():
        return None
    with metrics_path.open() as f:
        metrics = json.load(f)
    return metrics


def collect_scenario_lambda0(
    sweep_root: Path,
) -> Dict[str, List[List[dict]]]:
    """Return ``{scenario: [seed_metrics_list, ...]}`` for lambda_red = 0."""
    out: Dict[str, List[List[dict]]] = defaultdict(list)
    for cfg_path in sweep_root.rglob("config.json"):
        with cfg_path.open() as f:
            cfg = json.load(f)
        if cfg.get("lambda_red", 0) != 0:
            continue
        m = _load_seed_metrics(cfg_path.parent)
        if m is None:
            continue
        out[cfg["scenario"]].append(m)
    return out


def _series(metrics_list: List[List[dict]], key: str) -> np.ndarray:
    """Stack the per-iteration ``key`` over seeds. Shape = (n_seeds, n_iters)."""
    arrs = []
    for m in metrics_list:
        vals = [row.get(key, np.nan) for row in m]
        arrs.append(vals)
    if not arrs:
        return np.zeros((0, 0))
    # All cells should have the same number of iterations; if not, pad-NaN.
    max_len = max(len(a) for a in arrs)
    padded = []
    for a in arrs:
        if len(a) < max_len:
            a = a + [np.nan] * (max_len - len(a))
        padded.append(a)
    return np.asarray(padded, dtype=float)


def _mean_iqr(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if arr.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    mean = np.nanmean(arr, axis=0)
    lo = np.nanpercentile(arr, 25, axis=0)
    hi = np.nanpercentile(arr, 75, axis=0)
    return mean, lo, hi


def _scenario_color(scenario: str) -> str:
    return {
        "trivial_cooperation": "tab:green",
        "default": "tab:blue",
        "chain_reaction": "tab:red",
    }.get(scenario, "tab:gray")


def plot_reward_vs_baseline(
    data: Dict[str, List[List[dict]]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    scenarios = ["trivial_cooperation", "default", "chain_reaction"]
    for ax, scen in zip(axes, scenarios):
        if scen not in data:
            ax.set_title(f"{scen} (no data)")
            continue
        arr = _series(data[scen], "mean_step_reward_team")
        if arr.size == 0:
            ax.set_title(f"{scen} (empty)")
            continue
        iters = np.arange(arr.shape[1])
        mean, lo, hi = _mean_iqr(arr)
        c = _scenario_color(scen)
        ax.plot(iters, mean, color=c, label=f"PPO mean (n={arr.shape[0]} seeds)")
        ax.fill_between(iters, lo, hi, color=c, alpha=0.2, label="IQR")

        bl = BASELINES.get(scen, {})
        if "random" in bl:
            ax.axhline(
                bl["random"],
                color="black",
                linestyle="--",
                alpha=0.6,
                label=f"random = {bl['random']:.0f}",
            )
        if "heuristic" in bl and bl["heuristic"] != bl.get("random"):
            ax.axhline(
                bl["heuristic"],
                color="dimgray",
                linestyle=":",
                alpha=0.6,
                label=f"heuristic = {bl['heuristic']:.0f}",
            )

        ax.set_title(f"{scen} (lambda_red=0)")
        ax.set_xlabel("iteration")
        ax.set_ylabel("mean_step_reward_team")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(
        "PPO learning curve vs random/heuristic baseline\n"
        "(issue #145: PPO matches but does not exceed random)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_entropy_per_agent(
    data: Dict[str, List[List[dict]]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    scenarios = ["trivial_cooperation", "default", "chain_reaction"]
    agent_colors = ["tab:purple", "tab:orange", "tab:cyan", "tab:olive"]
    for ax, scen in zip(axes, scenarios):
        if scen not in data:
            ax.set_title(f"{scen} (no data)")
            continue
        for ai in range(4):
            arr = _series(data[scen], f"action_entropy/agent_{ai}")
            if arr.size == 0:
                continue
            iters = np.arange(arr.shape[1])
            mean, lo, hi = _mean_iqr(arr)
            ax.plot(iters, mean, color=agent_colors[ai], label=f"agent_{ai}")
            ax.fill_between(iters, lo, hi, color=agent_colors[ai], alpha=0.15)
        ax.set_title(f"{scen} (lambda_red=0)")
        ax.set_xlabel("iteration")
        ax.set_ylabel("action_entropy (nats)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(
        "Per-agent action-distribution entropy over training\n"
        "(low values = collapsed/deterministic policy)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_value_loss_log(
    data: Dict[str, List[List[dict]]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    scenarios = ["trivial_cooperation", "default", "chain_reaction"]
    for ax, scen in zip(axes, scenarios):
        if scen not in data:
            ax.set_title(f"{scen} (no data)")
            continue
        # Mean of per-agent value loss for clarity.
        per_agent = []
        for ai in range(4):
            arr = _series(data[scen], f"value_loss/agent_{ai}")
            if arr.size:
                per_agent.append(arr)
        if not per_agent:
            continue
        stacked = np.stack(per_agent, axis=0)  # (4, n_seeds, n_iters)
        mean_over_agents = stacked.mean(axis=0)
        iters = np.arange(mean_over_agents.shape[1])
        mean, lo, hi = _mean_iqr(mean_over_agents)
        c = _scenario_color(scen)
        ax.plot(iters, mean, color=c, label="mean across agents")
        ax.fill_between(iters, lo, hi, color=c, alpha=0.2, label="IQR (seeds)")
        ax.set_yscale("log")
        ax.set_title(f"{scen} (lambda_red=0)")
        ax.set_xlabel("iteration")
        ax.set_ylabel("value_loss (log scale)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(
        "Value loss magnitude over training (log scale)\n"
        "(initial values O(10^5) — see issue hypothesis: value loss dominates)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_loss_decomposition(
    data: Dict[str, List[List[dict]]],
    out_path: Path,
) -> None:
    """Compare scaled loss terms: |policy|, value_coef*value, entropy_coef*entropy.

    Plotted on log scale because value_coef * value_loss is many orders of
    magnitude above the other two terms.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    scenarios = ["trivial_cooperation", "default", "chain_reaction"]
    for ax, scen in zip(axes, scenarios):
        if scen not in data:
            ax.set_title(f"{scen} (no data)")
            continue

        # Aggregate over agents (mean) then over seeds.
        def agg(prefix: str) -> np.ndarray:
            per_agent = []
            for ai in range(4):
                arr = _series(data[scen], f"{prefix}/agent_{ai}")
                if arr.size:
                    per_agent.append(arr)
            if not per_agent:
                return np.zeros(0)
            return np.stack(per_agent, axis=0).mean(axis=0)

        pol = np.abs(agg("policy_loss"))
        val = agg("value_loss") * VALUE_COEF
        ent = agg("entropy") * ENTROPY_COEF

        if pol.size == 0:
            continue
        iters = np.arange(pol.shape[1])
        for series, name, color in (
            (pol, "|policy_loss|", "tab:blue"),
            (val, "value_coef * value_loss", "tab:red"),
            (ent, "entropy_coef * entropy", "tab:green"),
        ):
            mean, lo, hi = _mean_iqr(series)
            # Replace any zeros with a small floor for log plotting.
            mean = np.where(mean <= 0, np.nan, mean)
            ax.plot(iters, mean, color=color, label=name)
        ax.set_yscale("log")
        ax.set_title(f"{scen} (lambda_red=0)")
        ax.set_xlabel("iteration")
        ax.set_ylabel("scaled loss term (log)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(
        "Loss-term magnitudes (after coefficient scaling)\n"
        "(value_coef * value_loss dominates the gradient; entropy bonus negligible)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_loss_term_scales(
    data: Dict[str, List[List[dict]]],
    out_path: Path,
) -> None:
    """Bar chart of mean magnitude per scaled loss term at iter 0 and iter 49.

    Companion to plot_loss_decomposition for a quick visual on dominance.
    """
    scenarios = [
        s for s in ["trivial_cooperation", "default", "chain_reaction"] if s in data
    ]
    if not scenarios:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    _xs = np.arange(len(scenarios) * 2)  # iter0 then iter_last per scenario
    width = 0.25

    def agg_term(metrics_list: List[List[dict]], prefix: str, iter_idx: int) -> float:
        vals = []
        for m in metrics_list:
            if iter_idx >= len(m):
                continue
            row = m[iter_idx]
            agents = [abs(row.get(f"{prefix}/agent_{ai}", 0.0)) for ai in range(4)]
            vals.append(np.mean(agents))
        return float(np.mean(vals)) if vals else 0.0

    labels = []
    pol_bars = []
    val_bars = []
    ent_bars = []
    for scen in scenarios:
        ms = data[scen]
        if not ms:
            continue
        n_iters = len(ms[0])
        for idx, tag in ((0, "iter 0"), (n_iters - 1, f"iter {n_iters - 1}")):
            labels.append(f"{scen}\n{tag}")
            pol_bars.append(agg_term(ms, "policy_loss", idx))
            val_bars.append(agg_term(ms, "value_loss", idx) * VALUE_COEF)
            ent_bars.append(agg_term(ms, "entropy", idx) * ENTROPY_COEF)

    x = np.arange(len(labels))
    ax.bar(x - width, pol_bars, width, label="|policy_loss|", color="tab:blue")
    ax.bar(x, val_bars, width, label="value_coef * value_loss", color="tab:red")
    ax.bar(
        x + width, ent_bars, width, label="entropy_coef * entropy", color="tab:green"
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=0)
    ax.set_ylabel("|term magnitude| (log)")
    ax.set_title("Scaled loss-term magnitudes at start and end of training")
    ax.grid(alpha=0.3, which="both", axis="y")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


def write_summary(
    data: Dict[str, List[List[dict]]],
    out_path: Path,
) -> None:
    """Write a human-readable text summary of the diagnostics."""
    lines: List[str] = []
    lines.append("# P3 Specialization Plateau Diagnostics (Phase 1)")
    lines.append("")
    lines.append("Per-scenario summary at lambda_red=0 across all seeds.")
    lines.append("All values are means across seeds.")
    lines.append("")
    for scen in ["trivial_cooperation", "default", "chain_reaction"]:
        if scen not in data:
            continue
        ms = data[scen]
        n_seeds = len(ms)
        if n_seeds == 0:
            continue
        n_iters = len(ms[0])
        # Reward
        rewards = _series(ms, "mean_step_reward_team")
        # Per-agent value loss and entropy (mean over agents)
        val_losses = np.stack(
            [_series(ms, f"value_loss/agent_{ai}") for ai in range(4)], axis=0
        ).mean(axis=0)
        entropies = np.stack(
            [_series(ms, f"entropy/agent_{ai}") for ai in range(4)], axis=0
        ).mean(axis=0)
        pol_losses = np.stack(
            [_series(ms, f"policy_loss/agent_{ai}") for ai in range(4)], axis=0
        ).mean(axis=0)

        bl = BASELINES.get(scen, {})
        lines.append(f"## {scen}")
        lines.append(f"- n_seeds = {n_seeds}, n_iters = {n_iters}")
        lines.append(
            f"- reward iter 0 -> iter {n_iters - 1}: "
            f"{np.nanmean(rewards[:, 0]):.2f} -> "
            f"{np.nanmean(rewards[:, -1]):.2f}"
            f"  (baseline random = {bl.get('random', float('nan'))})"
        )
        lines.append(
            f"- mean value_loss iter 0 -> iter {n_iters - 1}: "
            f"{np.nanmean(val_losses[:, 0]):.2e} -> "
            f"{np.nanmean(val_losses[:, -1]):.2e}"
            f"  (scaled by value_coef={VALUE_COEF}: "
            f"{VALUE_COEF * np.nanmean(val_losses[:, 0]):.2e})"
        )
        lines.append(
            f"- mean |policy_loss| iter 0 -> iter {n_iters - 1}: "
            f"{np.nanmean(np.abs(pol_losses[:, 0])):.3e} -> "
            f"{np.nanmean(np.abs(pol_losses[:, -1])):.3e}"
        )
        lines.append(
            f"- mean entropy iter 0 -> iter {n_iters - 1}: "
            f"{np.nanmean(entropies[:, 0]):.3e} -> "
            f"{np.nanmean(entropies[:, -1]):.3e}"
            f"  (scaled by entropy_coef={ENTROPY_COEF}: "
            f"{ENTROPY_COEF * np.nanmean(entropies[:, -1]):.3e})"
        )
        # Dominance ratio at iter 0
        v0 = VALUE_COEF * np.nanmean(val_losses[:, 0])
        p0 = np.nanmean(np.abs(pol_losses[:, 0]))
        e0 = ENTROPY_COEF * np.nanmean(entropies[:, 0])
        lines.append(
            f"- iter 0 dominance: "
            f"value_term / policy_term = {v0 / max(p0, 1e-12):.1e}, "
            f"value_term / entropy_term = {v0 / max(e0, 1e-12):.1e}"
        )
        lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")


def _default_sweep_root() -> Path:
    """Find the sweep root.

    Order:
        1. ``experiments/p3_specialization/runs`` relative to CWD.
        2. ``experiments/p3_specialization/runs`` in the main repo, when
           this script is invoked from a worktree (where ``runs/`` is
           gitignored and lives only in the canonical checkout).
    """
    here = Path("experiments/p3_specialization/runs")
    if here.exists():
        return here
    # Walk up from this file looking for the main-repo runs/ dir.
    this = Path(__file__).resolve()
    for parent in this.parents:
        # Detect the main repo by the presence of a non-worktree .git/ dir.
        candidate = parent / "experiments" / "p3_specialization" / "runs"
        if candidate.exists() and not str(candidate).count("/worktrees/"):
            return candidate
    return here  # last resort; will be reported as not-found below


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sweep-root",
        type=Path,
        default=None,
        help="root containing {scenario}/lambda_*/seed_*/metrics.json "
        "(auto-detected if not provided)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/p3_specialization/diagnostics"),
    )
    args = p.parse_args()

    if args.sweep_root is None:
        args.sweep_root = _default_sweep_root()
    print(f"Using sweep_root = {args.sweep_root}")

    if not args.sweep_root.exists():
        raise SystemExit(
            f"sweep_root not found: {args.sweep_root}\n"
            "Existing metrics live under experiments/p3_specialization/runs/."
            " This Phase 1 script does not run new training."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = collect_scenario_lambda0(args.sweep_root)
    n_total = sum(len(v) for v in data.values())
    print(
        f"Loaded {n_total} lambda_red=0 cells across "
        f"{len(data)} scenarios: {sorted(data.keys())}"
    )
    if n_total == 0:
        raise SystemExit("No lambda_red=0 cells found; cannot produce plots.")

    plot_reward_vs_baseline(data, args.out_dir / "reward_vs_baseline.png")
    plot_entropy_per_agent(data, args.out_dir / "entropy_per_agent.png")
    plot_value_loss_log(data, args.out_dir / "value_loss_log.png")
    plot_loss_decomposition(data, args.out_dir / "loss_decomposition.png")
    plot_loss_term_scales(data, args.out_dir / "loss_term_scales.png")
    write_summary(data, args.out_dir / "summary.md")
    print(f"\nAll diagnostics written to {args.out_dir}")


if __name__ == "__main__":
    main()
