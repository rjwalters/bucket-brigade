"""Analysis for issue #281 — TransformerPolicyNetwork BC-fit vs MLP baseline.

Aggregates per-seed BC-fit JSONs produced by
``run_issue281_transformer_bc.sh`` and applies the verdict ladder from the
curated issue body:

| Transformer house-on-WORK | Interpretation | Next step |
|---|---|---|
| ≥ 0.90 *and* clearly beats MLP | Attention solved indexed lookup. PPO's MLP was wrong inductive bias. | File PPO-with-Transformer production issue. |
| Within ~0.05 of best MLP | Architecture wasn't the gap. Specialist may use info not in obs — task reformulation needed. | File obs/reward redesign issue. |
| Modest improvement (0.05–0.15) | Combined approach plausible but inconclusive. | Decide based on cost; default = file env-side issue. |

Reference MLP baseline from **PR #278** on the same task:
    house_acc_on_work_subset (MLP @ hidden_size=64, 40k pairs / 30 epochs) ≈ 0.93+
    gap_closed (BC eval) ≈ 0.934
(These constants are pulled from PR #278's bc_summary.json. If your local
MLP baseline disagrees materially, pass --mlp-baseline-* overrides.)

Usage::

    python experiments/p3_specialization/analyze_281.py \
        --results-dir experiments/p3_specialization/diagnostics/results/issue281_transformer_bc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# Reference baselines from PR #278 (MLP @ hidden_size=64, 40k pairs/30 epochs).
# These are the numbers to beat for the Transformer to claim an inductive-bias
# advantage on the WORK-house discriminator sub-task.
MLP_BASELINE_HOUSE_ACC_ON_WORK = 0.93  # approximate; see PR #278 bc_summary.json
MLP_BASELINE_GAP_CLOSED = 0.934  # exact: PR #278 stated 0.934 in body

# Verdict thresholds from the curator spec.
ATTENTION_SOLVED_THRESHOLD = 0.90  # transformer must clear this AND beat MLP
WITHIN_EPS_OF_MLP = 0.05  # |Δ| ≤ this -> architecture wasn't the gap
MODEST_IMPROVEMENT_LOWER = 0.05  # 0.05-0.15 range -> inconclusive


def _load_result(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _per_seed_summary(seed_dir: Path) -> Optional[Dict]:
    result_path = seed_dir / "bc_fit_only_result.json"
    payload = _load_result(result_path)
    if payload is None:
        return None
    res = payload.get("result", {})
    final = res.get("final", {})
    return {
        "seed": payload.get("args", {}).get("seed"),
        "architecture": res.get("architecture", "unknown"),
        "n_params": res.get("n_params"),
        "house_acc": final.get("house_acc"),
        "mode_acc": final.get("mode_acc"),
        "signal_acc": final.get("signal_acc"),
        "joint_acc": final.get("joint_acc"),
        "eval_loss": final.get("eval_loss"),
        "house_acc_on_work_subset": res.get("house_acc_on_work_subset"),
        "eval_work_rows": res.get("eval_work_rows"),
    }


def aggregate(seed_summaries: List[Dict]) -> Dict:
    """Aggregate per-seed summaries into a single mean ± stdev report."""
    valid = [
        s
        for s in seed_summaries
        if s is not None and s.get("house_acc_on_work_subset") is not None
    ]
    if not valid:
        return {"n_seeds": 0}

    def _mean(key: str) -> float:
        vals = [s[key] for s in valid if s.get(key) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    def _std(key: str) -> float:
        vals = [s[key] for s in valid if s.get(key) is not None]
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    return {
        "n_seeds": len(valid),
        "house_acc_on_work_subset_mean": _mean("house_acc_on_work_subset"),
        "house_acc_on_work_subset_std": _std("house_acc_on_work_subset"),
        "house_acc_mean": _mean("house_acc"),
        "house_acc_std": _std("house_acc"),
        "joint_acc_mean": _mean("joint_acc"),
        "joint_acc_std": _std("joint_acc"),
        "eval_loss_mean": _mean("eval_loss"),
        "eval_loss_std": _std("eval_loss"),
        "n_params": valid[0].get("n_params"),
        "architecture": valid[0].get("architecture"),
    }


def classify_verdict(
    transformer_house_acc_work: float,
    mlp_baseline_house_acc_work: float,
) -> Dict:
    """Apply the curator's 3-tier verdict ladder.

    Returns a dict with the verdict label, the delta vs MLP, and a recommended
    next step.
    """
    delta = transformer_house_acc_work - mlp_baseline_house_acc_work

    if (
        transformer_house_acc_work >= ATTENTION_SOLVED_THRESHOLD
        and delta > WITHIN_EPS_OF_MLP
    ):
        verdict = "ATTENTION_SOLVED"
        next_step = (
            "Attention solved the WORK-house indexed lookup. File a PPO-with-"
            "Transformer production issue."
        )
    elif abs(delta) <= WITHIN_EPS_OF_MLP:
        verdict = "ARCHITECTURE_WAS_NOT_THE_GAP"
        next_step = (
            "Transformer matches MLP. Architecture wasn't the bottleneck — "
            "specialist likely uses info not in the obs. File an obs/reward "
            "redesign issue (env-side intervention per project thesis)."
        )
    elif MODEST_IMPROVEMENT_LOWER < delta < 0.15:
        verdict = "MODEST_IMPROVEMENT"
        next_step = (
            "Modest gain — inconclusive. Default action: file env-side issue. "
            "Optionally try Transformer + larger MLP + class balancing combined."
        )
    elif delta < 0:
        verdict = "TRANSFORMER_WORSE"
        next_step = (
            "Transformer underperforms MLP — likely an optimization/budget "
            "issue. Re-check learning rate, warmup, and class balancing before "
            "concluding."
        )
    else:
        # delta in [0.15, ATTENTION_SOLVED_THRESHOLD - mlp_baseline_house_acc_work)
        # AND transformer < ATTENTION_SOLVED_THRESHOLD
        verdict = "STRONG_IMPROVEMENT_BUT_BELOW_THRESHOLD"
        next_step = (
            "Substantial gain but below the 0.90 threshold. File PPO-with-"
            "Transformer follow-up; track whether attention is the right "
            "primitive or just a capacity boost."
        )

    return {
        "verdict": verdict,
        "delta_vs_mlp": delta,
        "transformer_house_acc_on_work": transformer_house_acc_work,
        "mlp_baseline_house_acc_on_work": mlp_baseline_house_acc_work,
        "next_step": next_step,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path(
            "experiments/p3_specialization/diagnostics/results/issue281_transformer_bc"
        ),
        help="Directory containing seed_<S>/bc_fit_only_result.json files.",
    )
    p.add_argument(
        "--mlp-baseline-house-acc-on-work",
        type=float,
        default=MLP_BASELINE_HOUSE_ACC_ON_WORK,
        help=(
            "Reference MLP baseline for house_acc_on_work_subset (default: "
            f"{MLP_BASELINE_HOUSE_ACC_ON_WORK} from PR #278). Override if your "
            "#279 MLP cell produced a different number."
        ),
    )
    p.add_argument(
        "--mlp-baseline-gap-closed",
        type=float,
        default=MLP_BASELINE_GAP_CLOSED,
        help=(
            f"Reference MLP gap_closed (default: {MLP_BASELINE_GAP_CLOSED} from "
            "PR #278). Currently unused in the verdict but recorded in the "
            "output for cross-reference."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Where to write the aggregated JSON summary. Defaults to "
            "<results-dir>/analyze_281.json."
        ),
    )
    args = p.parse_args()

    seed_dirs = sorted(d for d in args.results_dir.glob("seed_*") if d.is_dir())
    if not seed_dirs:
        raise SystemExit(
            f"No seed_* subdirectories under {args.results_dir}. Did you run "
            "run_issue281_transformer_bc.sh first?"
        )

    seed_summaries = [_per_seed_summary(d) for d in seed_dirs]
    agg = aggregate(seed_summaries)

    if agg.get("n_seeds", 0) == 0:
        raise SystemExit(
            f"No valid bc_fit_only_result.json files found under {args.results_dir}."
        )

    verdict = classify_verdict(
        transformer_house_acc_work=agg["house_acc_on_work_subset_mean"],
        mlp_baseline_house_acc_work=args.mlp_baseline_house_acc_on_work,
    )

    payload = {
        "issue": 281,
        "results_dir": str(args.results_dir),
        "n_seeds": agg["n_seeds"],
        "per_seed": seed_summaries,
        "aggregate": agg,
        "reference_baseline": {
            "source": "PR #278",
            "mlp_house_acc_on_work_subset": args.mlp_baseline_house_acc_on_work,
            "mlp_gap_closed": args.mlp_baseline_gap_closed,
            "note": (
                "PR #278 found PolicyNetwork(hidden_size=64) reached "
                "gap_closed=0.934 at 40k pairs/30 epochs on minimal_"
                "specialization. This contradicts the original INDUCTIVE_BIAS_"
                "GAP premise that motivated #281; the transformer escalation "
                "may be unnecessary."
            ),
        },
        "verdict": verdict,
    }

    print()
    print("=" * 72)
    print("ISSUE #281: TransformerPolicyNetwork BC-fit verdict")
    print("=" * 72)
    print(f"Results directory:      {args.results_dir}")
    print(f"Architecture:           {agg['architecture']}")
    print(f"Param count:            {agg['n_params']}")
    print(f"Seeds aggregated:       {agg['n_seeds']}")
    print()
    print("Transformer (this run):")
    print(
        f"  house_acc_on_work_subset = "
        f"{agg['house_acc_on_work_subset_mean']:.4f} "
        f"± {agg['house_acc_on_work_subset_std']:.4f}"
    )
    print(
        f"  house_acc (overall)      = {agg['house_acc_mean']:.4f} "
        f"± {agg['house_acc_std']:.4f}"
    )
    print(
        f"  joint_acc                = {agg['joint_acc_mean']:.4f} "
        f"± {agg['joint_acc_std']:.4f}"
    )
    print(
        f"  eval_loss                = {agg['eval_loss_mean']:.4f} "
        f"± {agg['eval_loss_std']:.4f}"
    )
    print()
    print(
        f"MLP baseline (PR #278):  house_acc_on_work_subset ≈ "
        f"{args.mlp_baseline_house_acc_on_work:.4f}"
    )
    print(f"Δ (transformer - MLP):   {verdict['delta_vs_mlp']:+.4f}")
    print()
    print(f"VERDICT: {verdict['verdict']}")
    print(f"NEXT STEP: {verdict['next_step']}")
    print("=" * 72)

    out_path = args.out or (args.results_dir / "analyze_281.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
