"""Analysis for issue #220 — paired baseline/treatment comparison of the
per-agent observation differentiation fix (#216).

Reads per-cell ``metrics.json`` files under

    experiments/p3_specialization/runs/issue220_baseline/<scenario>/lambda_0e0/seed_<N>/
    experiments/p3_specialization/runs/issue220_treatment/<scenario>/lambda_0e0/seed_<N>/

and emits a markdown summary + JSON under

    experiments/p3_specialization/diagnostics/results/issue220_obsfix/

Five metrics per cell, per the issue #220 curator protocol:

1. **Final per-step team reward** — mean of ``mean_step_reward_team`` over
   the last 5 iterations (the trailing-5 convention from #199).
2. **Per-agent action entropy** at iter-final (``action_entropy/agent_i``
   plus ``action_entropy/mean``). Pre-#216 collapse should manifest as
   nearly-identical per-agent entropies.
3. **Per-agent CV + action-reward R²** — populated from
   ``inspect_rollout_rewards`` JSON (if present in the cell dir).
4. **Pairwise action-distribution KL** — read from
   ``pairwise_action_kl.json`` in each cell (produced by
   ``diagnostics/pairwise_action_kl.py``). Pre-#216 → KL ≈ 0; post-#216
   → KL > 0 iff agents specialize.
5. **Gap closed** — for ``minimal_specialization``:
   ``(ppo_trailing5 − random) / (specialist − random)`` with the 2026-05-15
   references (random = −96.07, specialist = −22.07; pre-#216 reference
   was 18.1%). Pass bar: treatment ≥ 50%.

Usage::

    uv run python experiments/p3_specialization/analyze_220.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


ARMS = ["baseline", "treatment"]
SCENARIOS = ["default", "minimal_specialization"]
SEEDS = [42, 43, 44]
LAMBDA_DIR = "lambda_0e0"
NUM_AGENTS = 4
TRAILING_N = 5

# 2026-05-15 references from research_notebook/issue199 entry.
MINSPEC_RANDOM = -96.07
MINSPEC_SPECIALIST = -22.07


def _cell_dir(root: Path, arm: str, scenario: str, seed: int) -> Path:
    return root / f"issue220_{arm}" / scenario / LAMBDA_DIR / f"seed_{seed}"


def _trailing_team(metrics: List[dict], n: int = TRAILING_N) -> float:
    rewards = [row["mean_step_reward_team"] for row in metrics]
    tail = rewards[-n:] if len(rewards) >= n else rewards
    return float(np.mean(tail))


def _per_agent_entropy_final(metrics: List[dict]) -> Tuple[List[float], float]:
    last = metrics[-1]
    per = [float(last[f"action_entropy/agent_{i}"]) for i in range(NUM_AGENTS)]
    mean = float(last["action_entropy/mean"])
    return per, mean


def _load_kl(cell: Path) -> Optional[dict]:
    p = cell / "pairwise_action_kl.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_cell(cell: Path) -> Optional[dict]:
    mfile = cell / "metrics.json"
    if not mfile.exists():
        return None
    metrics = json.loads(mfile.read_text())
    per_ent, mean_ent = _per_agent_entropy_final(metrics)
    kl = _load_kl(cell)
    return {
        "cell": str(cell),
        "trailing5_team_reward": _trailing_team(metrics),
        "final_iter": len(metrics) - 1,
        "per_agent_entropy_final": per_ent,
        "mean_entropy_final": mean_ent,
        "entropy_spread": float(np.max(per_ent) - np.min(per_ent)),
        "kl_off_diag_mean": (kl["kl_off_diag_mean"] if kl else None),
        "kl_off_diag_max": (kl["kl_off_diag_max"] if kl else None),
    }


def gap_closed(ppo_trailing5: float) -> float:
    """Fraction of specialist−random gap closed on minimal_specialization."""
    return (ppo_trailing5 - MINSPEC_RANDOM) / (MINSPEC_SPECIALIST - MINSPEC_RANDOM)


def aggregate_arm(root: Path, arm: str) -> Dict[str, Dict]:
    per_scenario: Dict[str, Dict] = {}
    for scenario in SCENARIOS:
        seeds_data: List[Dict] = []
        for s in SEEDS:
            cell = _cell_dir(root, arm, scenario, s)
            d = load_cell(cell)
            if d is None:
                seeds_data.append({"seed": s, "missing": True})
                continue
            d["seed"] = s
            seeds_data.append(d)
        present = [d for d in seeds_data if not d.get("missing")]
        if not present:
            per_scenario[scenario] = {"seeds": seeds_data, "n_seeds": 0}
            continue
        team = np.array([d["trailing5_team_reward"] for d in present])
        ents = np.array([d["mean_entropy_final"] for d in present])
        spreads = np.array([d["entropy_spread"] for d in present])
        kl_means = [d["kl_off_diag_mean"] for d in present if d["kl_off_diag_mean"] is not None]
        per_scenario[scenario] = {
            "seeds": seeds_data,
            "n_seeds": len(present),
            "team_reward_mean": float(team.mean()),
            "team_reward_std": float(team.std(ddof=1)) if len(team) > 1 else 0.0,
            "team_reward_per_seed": team.tolist(),
            "mean_entropy_mean": float(ents.mean()),
            "entropy_spread_mean": float(spreads.mean()),
            "kl_off_diag_mean": (float(np.mean(kl_means)) if kl_means else None),
        }
        if scenario == "minimal_specialization":
            per_scenario[scenario]["gap_closed_mean"] = float(gap_closed(team.mean()))
            per_scenario[scenario]["gap_closed_per_seed"] = [
                float(gap_closed(x)) for x in team
            ]
    return per_scenario


def render_markdown(results: Dict[str, Dict[str, Dict]]) -> str:
    lines = [
        "# Issue #220 — Obs-fix paired comparison",
        "",
        "Random/specialist references on `minimal_specialization` "
        f"(from 2026-05-15 notebook): random = {MINSPEC_RANDOM:.2f}, "
        f"specialist = {MINSPEC_SPECIALIST:.2f}. Pass bar = treatment closes ≥ 50%.",
        "",
        "## Team reward (trailing-5 mean of `mean_step_reward_team`)",
        "",
        "| scenario | arm | n | mean ± std | per-seed |",
        "|---|---|---|---|---|",
    ]
    for scenario in SCENARIOS:
        for arm in ARMS:
            r = results[arm].get(scenario, {})
            if not r.get("n_seeds"):
                lines.append(f"| {scenario} | {arm} | 0 | — | missing |")
                continue
            lines.append(
                f"| {scenario} | {arm} | {r['n_seeds']} | "
                f"{r['team_reward_mean']:+.2f} ± {r['team_reward_std']:.2f} | "
                f"{r['team_reward_per_seed']} |"
            )
    lines += [
        "",
        "## Policy divergence (final-iter entropy spread + pairwise action KL)",
        "",
        "| scenario | arm | mean_entropy | entropy_spread (max−min across agents) | KL off-diag mean |",
        "|---|---|---|---|---|",
    ]
    for scenario in SCENARIOS:
        for arm in ARMS:
            r = results[arm].get(scenario, {})
            if not r.get("n_seeds"):
                continue
            kl = r.get("kl_off_diag_mean")
            kl_s = f"{kl:.4f}" if kl is not None else "n/a"
            lines.append(
                f"| {scenario} | {arm} | {r['mean_entropy_mean']:.3f} | "
                f"{r['entropy_spread_mean']:.3f} | {kl_s} |"
            )
    lines += [
        "",
        "## Gap-closed on `minimal_specialization`",
        "",
        "| arm | gap_closed (trailing-5 mean) | per-seed | pre-reg verdict |",
        "|---|---|---|---|",
    ]
    for arm in ARMS:
        r = results[arm].get("minimal_specialization", {})
        if "gap_closed_mean" not in r:
            lines.append(f"| {arm} | — | missing | — |")
            continue
        gc = r["gap_closed_mean"]
        if gc >= 0.50:
            verdict = "**≥50% — pass bar met**"
        elif gc >= 0.25:
            verdict = "25–50% — partial / joint-cause"
        else:
            verdict = "< 25% — MAPPO needed"
        per_seed = [f"{x:.3f}" for x in r["gap_closed_per_seed"]]
        lines.append(f"| {arm} | {gc:.3f} | {per_seed} | {verdict} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    runs_root = Path("experiments/p3_specialization/runs")
    out_dir = Path("experiments/p3_specialization/diagnostics/results/issue220_obsfix")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {arm: aggregate_arm(runs_root, arm) for arm in ARMS}
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    (out_dir / "summary.md").write_text(render_markdown(results))
    print(f"wrote {out_dir / 'summary.json'}")
    print(f"wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
