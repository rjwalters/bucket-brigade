"""Analysis for issue #288 — PBT basin-escape verdict.

Mirrors ``analyze_270.py`` structure. Reads each PBT seed's ``gen_<G>/lineage_<L>/
metrics.json``, computes the best-of-population trailing-5 gap_closed at the
final generation per seed, averages across seeds, and applies the curator's
success-criterion table:

| best-of-pop trailing-5 gap_closed (mean over PBT seeds) | Verdict                    |
|---|---|
| ≥ 0.7  | ``population_escape`` (cooperative basin reachable via diversity) |
| 0.2 – 0.7 | ``partial`` (population biases toward basin but doesn't fully land) |
| < 0.2 | ``basin_globally_unreachable`` (re-elevate gradient-direction interventions) |

Baselines (random / specialist team reward per step) are imported from
``analyze_270`` so the gap_closed conversion is bit-identical with the #270
verdict pipeline. The analyzer also exposes a pure
``classify_verdict(best_of_pop_gap_closed)`` function so it can be unit-tested
without a real PBT run on disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Reuse the gap_closed conversion and trailing-N from analyze_270 so the #288
# verdict is directly comparable to #270/#271 outcomes (same scenario, same
# reference baselines). Keeping these constants here as fallbacks lets the
# analyzer be invoked standalone if analyze_270 is ever refactored.
try:
    from experiments.p3_specialization.analyze_270 import (
        MINSPEC_RANDOM,
        MINSPEC_SPECIALIST,
        TRAILING_N,
        gap_closed,
    )
except ImportError:  # pragma: no cover
    # Frozen-at-derivation fallback if analyze_270 is refactored. The
    # ``MINSPEC_SPECIALIST`` value is updated under issue #416 to track
    # the n=10k re-derivation in ``bucket_brigade/baselines/__init__.py`` —
    # keep in sync with that file's ``MINSPEC_SPECIALIST``.
    MINSPEC_RANDOM = -96.07
    MINSPEC_SPECIALIST = -28.38
    TRAILING_N = 5

    def gap_closed(per_step_team: float) -> float:
        return (per_step_team - MINSPEC_RANDOM) / (MINSPEC_SPECIALIST - MINSPEC_RANDOM)


# Verdict thresholds from the curator's success-criterion table. Adjusting
# these is a research-policy decision; do not touch without updating the
# issue body.
VERDICT_ESCAPE_THRESHOLD = 0.7
VERDICT_PARTIAL_THRESHOLD = 0.2


def classify_verdict(best_of_pop_gap_closed: float) -> Tuple[str, str]:
    """Pure function: map a best-of-pop trailing-5 gap_closed to a verdict tier.

    Public for unit testing.
    """
    if best_of_pop_gap_closed >= VERDICT_ESCAPE_THRESHOLD:
        return "population_escape", (
            f"best-of-pop trailing-5 gap_closed = {best_of_pop_gap_closed:.3f} "
            f">= {VERDICT_ESCAPE_THRESHOLD}. Cooperative basin reachable via "
            "population diversity. Phase-2 success — PBT escapes the random-"
            "init basin trap confirmed by #270/#271."
        )
    if best_of_pop_gap_closed >= VERDICT_PARTIAL_THRESHOLD:
        return "partial", (
            f"best-of-pop trailing-5 gap_closed = {best_of_pop_gap_closed:.3f} "
            f"sits in [{VERDICT_PARTIAL_THRESHOLD}, {VERDICT_ESCAPE_THRESHOLD}). "
            "Population biases toward the cooperative basin but doesn't fully "
            "land. Consider larger population, more generations, or stronger "
            "mutation."
        )
    return "basin_globally_unreachable", (
        f"best-of-pop trailing-5 gap_closed = {best_of_pop_gap_closed:.3f} "
        f"< {VERDICT_PARTIAL_THRESHOLD}. Cooperative basin unreachable from "
        "random init under local-search-with-diversity. Re-elevate gradient-"
        "direction interventions (#284 COMA, #287 LOLA, #283 potential "
        "shaping)."
    )


def _trailing5_team(metrics_path: Path) -> Optional[float]:
    if not metrics_path.exists():
        return None
    rows = json.loads(metrics_path.read_text())
    if not rows:
        return None
    tail = rows[-TRAILING_N:]
    return float(sum(r["mean_step_reward_team"] for r in tail) / max(1, len(tail)))


def aggregate_seed(seed_dir: Path) -> Dict:
    """For one PBT seed, find the final generation and report best-of-pop."""
    gen_dirs = sorted(
        [d for d in seed_dir.glob("gen_*") if d.is_dir()],
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not gen_dirs:
        return {"seed_dir": str(seed_dir), "n_generations": 0, "missing": True}

    final_gen = gen_dirs[-1]
    lineage_scores: List[Dict] = []
    for cell in sorted(final_gen.glob("lineage_*")):
        score = _trailing5_team(cell / "metrics.json")
        lineage_scores.append(
            {
                "lineage_id": int(cell.name.split("_")[1]),
                "trailing5_team_reward": score,
                "trailing5_gap_closed": gap_closed(score)
                if score is not None
                else None,
            }
        )

    valid = [
        s["trailing5_gap_closed"]
        for s in lineage_scores
        if s["trailing5_gap_closed"] is not None
    ]
    best = max(valid) if valid else None

    return {
        "seed_dir": str(seed_dir),
        "n_generations": len(gen_dirs),
        "final_gen": final_gen.name,
        "lineage_scores": lineage_scores,
        "best_of_pop_gap_closed": best,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-root",
        type=Path,
        default=Path("experiments/p3_specialization/runs/issue288_pbt"),
        help="Root containing seed_<N>/ PBT trial directories.",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/p3_specialization/diagnostics/results/issue288_pbt"),
    )
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_summaries = [aggregate_seed(args.runs_root / f"seed_{s}") for s in args.seeds]
    valid_bests = [
        s["best_of_pop_gap_closed"]
        for s in seed_summaries
        if s.get("best_of_pop_gap_closed") is not None
    ]
    if not valid_bests:
        verdict, reasoning = "no_data", "No completed PBT seeds found."
        mean_best = float("nan")
    else:
        mean_best = float(np.mean(valid_bests))
        verdict, reasoning = classify_verdict(mean_best)

    out = {
        "issue": 288,
        "verdict": verdict,
        "reasoning": reasoning,
        "mean_best_of_pop_gap_closed": mean_best,
        "per_seed": seed_summaries,
        "references": {
            "minspec_random": MINSPEC_RANDOM,
            "minspec_specialist": MINSPEC_SPECIALIST,
            "trailing_n": TRAILING_N,
            "verdict_escape_threshold": VERDICT_ESCAPE_THRESHOLD,
            "verdict_partial_threshold": VERDICT_PARTIAL_THRESHOLD,
        },
    }
    (args.output_dir / "analysis.json").write_text(json.dumps(out, indent=2))

    md = [
        "# Issue #288 — PBT basin-escape verdict",
        "",
        f"**Verdict**: `{verdict}`",
        "",
        f"**Reasoning**: {reasoning}",
        "",
        f"- mean best-of-pop trailing-5 gap_closed: {mean_best:.3f}",
        f"- n PBT seeds with data: {len(valid_bests)} / {len(seed_summaries)}",
        "",
        "## Per-seed best-of-pop",
    ]
    for s in seed_summaries:
        md.append(
            f"- `{Path(s['seed_dir']).name}` "
            f"({s.get('n_generations', 0)} gens): "
            f"best_gap_closed = {s.get('best_of_pop_gap_closed')}"
        )
    (args.output_dir / "verdict.md").write_text("\n".join(md))

    print(f"\nverdict: {verdict}")
    print(f"reasoning: {reasoning}")
    print(f"\nartifacts written to {args.output_dir}/")


if __name__ == "__main__":
    main()
