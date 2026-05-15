"""P3 sweep analysis: bootstrap CIs across seeds and falsifier evaluation.

Walks an ``experiments/p3_specialization/runs/`` tree, aggregates per-cell
metrics across seeds, and reports the answer to each pre-registered
falsifier in :data:`FALSIFIERS`. The output is machine-readable JSON plus a
short human summary printed to stdout.

The falsifiers (from ``research_notebook/2026-05-13_p3_specialization_plan.md``):

    F1: monotone decrease in conditional redundancy as lambda_red increases.
    F2: reward strictly worse at every lambda_red > 0 (penalty just hurts).

If F1 fails or F2 holds, P3 is falsified --- which is itself a publishable
result.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float, float]:
    rng = rng or np.random.default_rng(0)
    n = len(values)
    point = float(values.mean())
    if n < 2:
        return point, point, point
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = values[idx].mean()
    alpha = (1.0 - confidence) / 2.0
    return (
        point,
        float(np.quantile(boots, alpha)),
        float(np.quantile(boots, 1.0 - alpha)),
    )


def _load_cell(cell_dir: Path) -> Dict[str, object] | None:
    metrics_path = cell_dir / "metrics.json"
    config_path = cell_dir / "config.json"
    if not metrics_path.exists() or not config_path.exists():
        return None
    with metrics_path.open() as f:
        metrics = json.load(f)
    with config_path.open() as f:
        cfg = json.load(f)

    # Final-iteration metrics are what we report for each cell.
    if not metrics:
        return None
    final = metrics[-1]
    out: Dict[str, object] = {
        "scenario": cfg["scenario"],
        "lambda_red": cfg["lambda_red"],
        "seed": cfg["seed"],
        "team_reward": final["mean_step_reward_team"],
        "mi_mean": final["mi/mean_pair"],
        "cmi_mean": final["cmi/mean_pair"],
        "action_entropy_mean": final["action_entropy/mean"],
    }
    # Option 3 sensitivity-check CMI (architect decision 2026-05-15, #172).
    # Older runs predate this metric; treat as optional so the analysis stays
    # tolerant of mixed-vintage sweep trees.
    #
    # Per the post-PR-#180 amendment to the architect notebook, the aggregate
    # ``cmi_action/mean_pair`` is masked over non-degenerate per-pair CMIs and
    # can be NaN (written as JSON ``null``) when every conditioning agent is
    # degenerate. Carry through ``cmi_action_n_valid_pairs`` so the
    # aggregator can report the explicit denominator alongside the mean.
    if "cmi_action/mean_pair" in final:
        out["cmi_action_mean"] = final["cmi_action/mean_pair"]
    if "cmi_action/n_valid_pairs" in final:
        out["cmi_action_n_valid_pairs"] = final["cmi_action/n_valid_pairs"]

    # Optional dropout robustness if it has been evaluated.
    dropout_path = cell_dir / "dropout_results.json"
    if dropout_path.exists():
        with dropout_path.open() as f:
            dr = json.load(f)
        base = dr["none"]["mean"]
        agent_means = [dr[k]["mean"] for k in dr if k.startswith("agent_")]
        out["dropout_baseline"] = base
        # Mean drop in team reward when any single agent is removed.
        out["dropout_mean_drop"] = float(base - np.mean(agent_means))
    return out


def aggregate(sweep_root: Path) -> Dict[Tuple[str, float], Dict[str, dict]]:
    """Group cell results by (scenario, lambda) and bootstrap across seeds."""
    cells = []
    for cfg_path in sweep_root.rglob("config.json"):
        loaded = _load_cell(cfg_path.parent)
        if loaded is not None:
            cells.append(loaded)
    print(f"Loaded {len(cells)} cells from {sweep_root}")

    by_key = defaultdict(list)
    for c in cells:
        by_key[(c["scenario"], c["lambda_red"])].append(c)

    out: Dict[Tuple[str, float], Dict[str, dict]] = {}
    metrics_to_aggregate = [
        "team_reward",
        "mi_mean",
        "cmi_mean",
        "cmi_action_mean",
        "action_entropy_mean",
        "dropout_baseline",
        "dropout_mean_drop",
    ]
    for key, group in by_key.items():
        agg: Dict[str, dict] = {"n_seeds": len(group)}
        for metric in metrics_to_aggregate:
            # Skip null/NaN cells. ``cmi_action_mean`` is masked to NaN (JSON
            # ``null``) when every conditioning agent is degenerate (notebook
            # Amendment, post-PR #180); other metrics shouldn't contain
            # ``None``/``NaN`` in practice, but filter defensively. We honour
            # NaN-skip semantics so a single fully-collapsed cell doesn't
            # poison the cross-seed mean.
            vals = []
            for c in group:
                if metric not in c:
                    continue
                v = c[metric]
                if v is None:
                    continue
                fv = float(v)
                if np.isnan(fv):
                    continue
                vals.append(fv)
            if not vals:
                continue
            point, lo, hi = _bootstrap_mean_ci(np.asarray(vals, dtype=float))
            agg[metric] = {"mean": point, "ci_lo": lo, "ci_hi": hi, "n": len(vals)}
        # Surface the n_valid_pairs denominator for the Option 3 aggregate so
        # consumers can interpret the masked mean correctly. We report the
        # min across seeds (worst case — "this many pairs were valid in at
        # least one seed of this cell"). Skipped if no seed in the cell
        # logged the field (mixed-vintage trees).
        n_valid = [
            int(c["cmi_action_n_valid_pairs"])
            for c in group
            if "cmi_action_n_valid_pairs" in c
        ]
        if n_valid:
            agg["cmi_action_n_valid_pairs"] = {
                "min": min(n_valid),
                "max": max(n_valid),
                "n": len(n_valid),
            }
        out[key] = agg
    return out


def evaluate_falsifiers(
    agg: Dict[Tuple[str, float], Dict[str, dict]],
) -> Dict[str, Dict[str, str]]:
    """Apply the pre-registered falsifiers to the aggregated results.

    Returns a dict ``{scenario: {falsifier: verdict}}`` where verdict is one
    of ``"supported"``, ``"partial"``, ``"falsified"``, ``"insufficient"``.
    """
    # Group by scenario, lambda-sorted within each.
    by_scenario: Dict[str, List[Tuple[float, dict]]] = defaultdict(list)
    for (scenario, lam), data in agg.items():
        by_scenario[scenario].append((lam, data))
    for s in by_scenario:
        by_scenario[s].sort(key=lambda x: x[0])

    verdicts: Dict[str, Dict[str, str]] = {}
    for scenario, rows in by_scenario.items():
        v: Dict[str, str] = {}

        # F1: monotone decrease in CMI as lambda increases.
        cmis = [r[1].get("cmi_mean", {}).get("mean") for r in rows]
        if any(c is None for c in cmis) or len(cmis) < 2:
            v["F1_cmi_monotone_decrease"] = "insufficient"
        else:
            diffs = np.diff(cmis)
            n_decreases = int((diffs < 0).sum())
            n_total = len(diffs)
            if n_decreases == n_total:
                v["F1_cmi_monotone_decrease"] = "supported"
            elif n_decreases >= n_total - 1:
                v["F1_cmi_monotone_decrease"] = "partial"
            else:
                v["F1_cmi_monotone_decrease"] = "falsified"

        # F2: reward strictly worse at every lambda_red > 0.
        rewards = {lam: data.get("team_reward", {}).get("mean") for lam, data in rows}
        baseline = rewards.get(0.0)
        if baseline is None:
            v["F2_reward_always_worse"] = "insufficient"
        else:
            penalised = {
                lam: r for lam, r in rewards.items() if lam > 0.0 and r is not None
            }
            if not penalised:
                v["F2_reward_always_worse"] = "insufficient"
            else:
                # F2 falsified if any penalised setting matches or beats baseline.
                any_helps = any(r >= baseline for r in penalised.values())
                all_worse = all(r < baseline for r in penalised.values())
                if all_worse:
                    # The penalty really does just hurt --- the proposed method
                    # is falsified for this scenario.
                    v["F2_reward_always_worse"] = "falsified-method"
                elif any_helps:
                    v["F2_reward_always_worse"] = "supported"
                else:
                    v["F2_reward_always_worse"] = "partial"

        verdicts[scenario] = v
    return verdicts


def _print_summary(
    agg: Dict[Tuple[str, float], Dict[str, dict]],
    verdicts: Dict[str, Dict[str, str]],
) -> None:
    print("\n=== Per-cell summary ===")
    # Group by scenario for printable output.
    rows = sorted(agg.items(), key=lambda x: (x[0][0], x[0][1]))
    cur_scenario = None
    for (scenario, lam), data in rows:
        if scenario != cur_scenario:
            print(f"\n[{scenario}]")
            cur_scenario = scenario
        n = data.get("n_seeds", 0)
        tr = data.get("team_reward", {})
        cmi = data.get("cmi_mean", {})
        cmi_action = data.get("cmi_action_mean", {})
        drop = data.get("dropout_mean_drop", {})
        print(
            f"  lambda={lam:<7g} n={n:2d} | "
            f"team_reward={tr.get('mean', float('nan')):8.3f} "
            f"[{tr.get('ci_lo', float('nan')):.2f}, {tr.get('ci_hi', float('nan')):.2f}] | "
            f"cmi={cmi.get('mean', float('nan')):.3f} "
            f"[{cmi.get('ci_lo', float('nan')):.2f}, {cmi.get('ci_hi', float('nan')):.2f}] | "
            f"cmi_action={cmi_action.get('mean', float('nan')):.3f} "
            f"[{cmi_action.get('ci_lo', float('nan')):.2f}, {cmi_action.get('ci_hi', float('nan')):.2f}] | "
            f"dropout_drop={drop.get('mean', float('nan')):.3f}"
        )

    print("\n=== Pre-registered falsifiers ===")
    for scenario, v in verdicts.items():
        print(f"  {scenario}: {v}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sweep-root", type=Path, default=Path("experiments/p3_specialization/runs")
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/p3_specialization/analysis.json"),
    )
    args = p.parse_args()

    agg = aggregate(args.sweep_root)
    verdicts = evaluate_falsifiers(agg)

    serializable = {f"{scenario}__{lam}": data for (scenario, lam), data in agg.items()}
    with args.output.open("w") as f:
        json.dump(
            {"aggregate": serializable, "falsifiers": verdicts},
            f,
            indent=2,
        )

    _print_summary(agg, verdicts)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
