"""Diff pre-#240 vs post-#240 Nash equilibria for the 12 v1 scenarios.

Reads `equilibrium.json` from `experiments/nash/v1_results_python/{scenario}/`
(pre-#240) and `experiments/nash/v1_results_python_post240/{scenario}/`
(post-#240), then emits a markdown table summarizing the verdict diff.

Usage:
    uv run python experiments/nash/scripts/diff_post240.py

Paste the output into `docs/NASH_BENCHMARKS.md`, replacing the stub rows
under the "Expected verdict diff" header.
"""

import json
from pathlib import Path

SCENARIOS = [
    "chain_reaction",
    "deceptive_calm",
    "default",
    "early_containment",
    "easy",
    "greedy_neighbor",
    "hard",
    "mixed_motivation",
    "overcrowding",
    "rest_trap",
    "sparse_heroics",
    "trivial_cooperation",
]
PRE = Path("experiments/nash/v1_results_python")
POST = Path("experiments/nash/v1_results_python_post240")


def summarize(eq_json):
    """Return (type, payoff, liar_mass, top_archetype_str)."""
    eq = eq_json["equilibrium"]
    pool = {s["index"]: s["classification"] for s in eq["strategy_pool"]}
    dist = {int(k): v for k, v in eq["distribution"].items()}
    liar_mass = sum(p for i, p in dist.items() if pool.get(i) == "Liar")
    top_idx = max(dist, key=dist.get)
    top = f"{pool.get(top_idx, '?')} ({dist[top_idx]:.2f})"
    return eq["type"], eq["expected_payoff"], liar_mass, top


def main():
    print(
        "| Scenario | Pre-#240 (type, payoff, top) | Post-#240 (type, payoff, top) | Liar mass Δ |"
    )
    print(
        "|----------|------------------------------|-------------------------------|-------------|"
    )
    for s in SCENARIOS:
        pre_path = PRE / s / "equilibrium.json"
        post_path = POST / s / "equilibrium.json"
        if not post_path.exists():
            print(f"| {s} | (pre present) | **MISSING** | — |")
            continue
        pre = json.loads(pre_path.read_text())
        post = json.loads(post_path.read_text())
        t0, p0, _, top0 = summarize(pre)
        t1, p1, l1, top1 = summarize(post)
        _, _, l0, _ = summarize(pre)
        delta = f"{l1 - l0:+.3f}"
        print(f"| {s} | {t0}, {p0:.2f}, {top0} | {t1}, {p1:.2f}, {top1} | {delta} |")


if __name__ == "__main__":
    main()
