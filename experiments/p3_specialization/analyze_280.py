"""Analysis for issue #280 — hidden_size capacity probe for BC-fit.

Aggregates per-cell results from
``experiments/p3_specialization/runs/issue280_hidden_size/hs_{HS}/seed_{SEED}/result.json``
and applies the curator's verdict matrix from the issue body.

Headline metric: ``result.house_acc_on_work_subset`` — eval-set accuracy of
the house head on rows where the specialist labeled mode=WORK. Chance level
≈ 0.354 (3 owned houses out of 10). Success threshold ≥ 0.90.

Verdict matrix (capacity-vs-data interpretation, from the issue body):

| Pattern across cells                                  | Verdict                |
|-------------------------------------------------------|------------------------|
| hs=64 already saturates (≥0.90) at short budget       | capacity_not_gap       |
| Monotonic improvement with hidden_size, plateau ≥0.90 | capacity_bound         |
| Plateau below 0.90 even at 512                        | architecture_mismatch  |
| No cell exceeds chance (~0.35)                        | information_gap        |

Usage:

    uv run python experiments/p3_specialization/analyze_280.py
    uv run python experiments/p3_specialization/analyze_280.py \\
        --runs-root experiments/p3_specialization/runs/issue280_hidden_size \\
        --hidden-sizes 64 128 \\
        --output-dir experiments/p3_specialization/diagnostics/results/issue280_hidden_size
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Discriminating-subtask reference points (issue body).
# 3 owned houses out of (typically) 10 -> chance is 3/10 = 0.30 for argmax,
# or ~0.354 if the specialist's label distribution over owned houses is
# non-uniform. We use the issue body's stated chance of 0.354 as the
# floor and 0.90 as the success threshold.
HOUSE_ON_WORK_CHANCE = 0.354
HOUSE_ON_WORK_SUCCESS = 0.90


def _load_cell(path: Path) -> Optional[dict]:
    f = path / "result.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def aggregate_cell(cells: List[Path], hidden_size: int) -> Dict:
    """Average headline metrics across seeds for a single hidden_size."""
    seeds_data = []
    house_acc_work = []
    eval_loss = []
    house_acc = []
    joint_acc = []
    n_params = None

    for cell in cells:
        payload = _load_cell(cell)
        if payload is None:
            seeds_data.append({"cell": str(cell), "missing": True})
            continue
        result = payload.get("result", {})
        final = result.get("final", {})
        h = result.get("house_acc_on_work_subset", float("nan"))
        seeds_data.append(
            {
                "cell": str(cell),
                "eval_loss": float(final.get("eval_loss", float("nan"))),
                "house_acc": float(final.get("house_acc", float("nan"))),
                "mode_acc": float(final.get("mode_acc", float("nan"))),
                "signal_acc": float(final.get("signal_acc", float("nan"))),
                "joint_acc": float(final.get("joint_acc", float("nan"))),
                "house_acc_on_work_subset": float(h),
                "eval_work_rows": int(result.get("eval_work_rows", 0)),
                "n_params": int(result.get("n_params", 0)),
                "verdict_bc": payload.get("verdict", None),
            }
        )
        if not np.isnan(h):
            house_acc_work.append(h)
        if not np.isnan(final.get("eval_loss", float("nan"))):
            eval_loss.append(float(final["eval_loss"]))
        if not np.isnan(final.get("house_acc", float("nan"))):
            house_acc.append(float(final["house_acc"]))
        if not np.isnan(final.get("joint_acc", float("nan"))):
            joint_acc.append(float(final["joint_acc"]))
        if n_params is None:
            n_params = int(result.get("n_params", 0))

    if not house_acc_work:
        return {
            "hidden_size": hidden_size,
            "n_seeds": 0,
            "seeds": seeds_data,
        }

    return {
        "hidden_size": hidden_size,
        "n_seeds": len(house_acc_work),
        "n_params": n_params,
        "house_acc_on_work_subset_mean": float(np.mean(house_acc_work)),
        "house_acc_on_work_subset_std": float(
            np.std(house_acc_work) if len(house_acc_work) > 1 else 0.0
        ),
        "eval_loss_mean": float(np.mean(eval_loss)) if eval_loss else float("nan"),
        "house_acc_mean": float(np.mean(house_acc)) if house_acc else float("nan"),
        "joint_acc_mean": float(np.mean(joint_acc)) if joint_acc else float("nan"),
        "seeds": seeds_data,
    }


def classify_verdict(rows: List[Dict]) -> Tuple[str, str]:
    """Apply the verdict matrix from the issue body to the per-cell rollup."""
    completed = [r for r in rows if r.get("n_seeds", 0) > 0]
    if not completed:
        return "no_data", "No cells produced result.json"

    by_hs = {r["hidden_size"]: r["house_acc_on_work_subset_mean"] for r in completed}
    hs_sorted = sorted(by_hs.keys())
    acc_sorted = [by_hs[h] for h in hs_sorted]

    smallest = hs_sorted[0]
    smallest_acc = by_hs[smallest]
    max_acc = max(acc_sorted)
    largest_hs = hs_sorted[-1]
    largest_acc = by_hs[largest_hs]

    # 1. Smallest hidden_size already saturates -> capacity is not the gap.
    if smallest_acc >= HOUSE_ON_WORK_SUCCESS:
        return "capacity_not_gap", (
            f"hidden_size={smallest} already reaches house-on-WORK="
            f"{smallest_acc:.3f} >= {HOUSE_ON_WORK_SUCCESS}. Capacity is not "
            "the bottleneck. If #279 (class-balanced BC) also passed, this "
            "ladder is resolved by class-balancing. If #279 failed, the "
            "specialist may rely on information not present in the obs — "
            "reframe research direction."
        )

    # 2. No cell exceeds chance -> information gap.
    if max_acc <= HOUSE_ON_WORK_CHANCE + 0.05:
        return "information_gap", (
            f"No cell exceeds chance (max house-on-WORK={max_acc:.3f}, "
            f"chance≈{HOUSE_ON_WORK_CHANCE}). Neither capacity nor "
            "class-balancing (#279) is the limiting factor. Information for "
            "the specialist's argmax over burning-owned houses may not be in "
            "the observation. Reframe: inspect obs structure; specialist may "
            "use privileged info."
        )

    # 3. Monotone improvement and at least one cell crosses success.
    monotone = all(
        acc_sorted[i + 1] >= acc_sorted[i] - 0.01 for i in range(len(acc_sorted) - 1)
    )
    if monotone and max_acc >= HOUSE_ON_WORK_SUCCESS:
        winning_hs = next(h for h, a in zip(hs_sorted, acc_sorted) if a >= HOUSE_ON_WORK_SUCCESS)
        return "capacity_bound", (
            f"Monotone improvement with hidden_size; plateau >= "
            f"{HOUSE_ON_WORK_SUCCESS} reached at hidden_size={winning_hs} "
            f"(house-on-WORK={by_hs[winning_hs]:.3f}). Confirmed capacity "
            "bound at short budget. PPO's default-64 actor is undersized for "
            f"this discriminating sub-task. File production-PPO follow-up "
            f"with hidden_size={winning_hs}."
        )

    # 4. Largest cell still below success -> architecture mismatch.
    if largest_acc < HOUSE_ON_WORK_SUCCESS:
        return "architecture_mismatch", (
            f"Largest hidden_size={largest_hs} only reaches house-on-WORK="
            f"{largest_acc:.3f} < {HOUSE_ON_WORK_SUCCESS}. Architecture "
            "mismatch isn't pure capacity — it's *kind* of capacity (depth? "
            "attention? non-linearity?). Promote #281 "
            "(TransformerPolicyNetwork)."
        )

    return "partial", (
        f"Mixed pattern across cells: house-on-WORK by hidden_size = "
        f"{dict(zip(hs_sorted, [round(a, 3) for a in acc_sorted]))}. "
        "Verdict is inconclusive; consider adding seeds or extending to Path B."
    )


def render_markdown(rows: List[Dict], verdict: str, reasoning: str) -> str:
    lines = [
        "# Issue #280 — hidden_size capacity probe for BC-fit",
        "",
        f"**Verdict**: `{verdict}`",
        "",
        f"**Reasoning**: {reasoning}",
        "",
        "## Per-cell results",
        "",
        "| hidden_size | n_params | n_seeds | eval_loss | house_acc | "
        "house_acc_on_work_subset | joint_acc |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: x["hidden_size"]):
        if r.get("n_seeds", 0) == 0:
            lines.append(
                f"| {r['hidden_size']} | - | 0 | - | - | - | - |"
            )
            continue
        lines.append(
            f"| {r['hidden_size']} | {r.get('n_params', '-')} | "
            f"{r['n_seeds']} | {r['eval_loss_mean']:.4f} | "
            f"{r['house_acc_mean']:.3f} | "
            f"{r['house_acc_on_work_subset_mean']:.3f} | "
            f"{r['joint_acc_mean']:.3f} |"
        )
    lines += [
        "",
        f"Chance level (house-on-WORK): {HOUSE_ON_WORK_CHANCE}",
        f"Success threshold (house-on-WORK): {HOUSE_ON_WORK_SUCCESS}",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-root",
        type=Path,
        default=Path("experiments/p3_specialization/runs/issue280_hidden_size"),
        help="Root containing hs_{HS}/seed_{SEED}/result.json cells.",
    )
    p.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[64, 128],
        help="Cells to load (Path A: 64 128; Path B: 64 128 256 512).",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiments/p3_specialization/diagnostics/results/issue280_hidden_size"
        ),
    )
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for hs in args.hidden_sizes:
        cells = [args.runs_root / f"hs_{hs}" / f"seed_{s}" for s in args.seeds]
        rows.append(aggregate_cell(cells, hs))

    verdict, reasoning = classify_verdict(rows)

    out = {
        "issue": 280,
        "verdict": verdict,
        "reasoning": reasoning,
        "rows": rows,
        "references": {
            "house_on_work_chance": HOUSE_ON_WORK_CHANCE,
            "house_on_work_success": HOUSE_ON_WORK_SUCCESS,
            "hidden_sizes": args.hidden_sizes,
            "seeds": args.seeds,
        },
    }
    (args.output_dir / "analysis.json").write_text(json.dumps(out, indent=2))
    (args.output_dir / "verdict.md").write_text(
        render_markdown(rows, verdict, reasoning)
    )

    print(f"\nverdict: {verdict}")
    print(f"reasoning: {reasoning}")
    print(f"\nartifacts written to {args.output_dir}/")


if __name__ == "__main__":
    main()
