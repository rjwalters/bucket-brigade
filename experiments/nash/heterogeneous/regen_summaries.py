#!/usr/bin/env python3
"""Regenerate summary.json/summary.md from results.json using fixed verdict logic.

Use this when the verdict logic in compute_nash_heterogeneous.py is updated
without re-running the (expensive) sweep.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

EPSILON = 50.0


def is_symmetric(profile_label: str) -> bool:
    parts = [p.split("(")[0].strip() for p in profile_label.split("|")]
    return len(set(parts)) == 1


def regen(scenario_dir: Path) -> None:
    with open(scenario_dir / "results.json") as f:
        r = json.load(f)

    equilibria = r["equilibria"]
    converged = [e for e in equilibria if e["converged"]]
    symmetric = [e for e in equilibria if e["symmetric_profile"]]
    asymmetric = [e for e in equilibria if not e["symmetric_profile"]]
    converged_asymmetric = [e for e in asymmetric if e["converged"]]
    converged_symmetric = [e for e in symmetric if e["converged"]]

    best_conv_asym = (
        max(converged_asymmetric, key=lambda e: e["team_payoff"])
        if converged_asymmetric else None
    )
    best_conv_sym = (
        max(converged_symmetric, key=lambda e: e["team_payoff"])
        if converged_symmetric else None
    )

    sym_best = best_conv_sym["team_payoff"] if best_conv_sym else None

    if best_conv_asym is not None and sym_best is not None:
        if best_conv_asym["team_payoff"] > sym_best + EPSILON:
            verdict = "asymmetric_ne_superior"
            verdict_detail = (
                f"Asymmetric NE (payoff={best_conv_asym['team_payoff']:.2f}) outperforms "
                f"best symmetric NE (payoff={sym_best:.2f}). "
                "Role differentiation is a genuine equilibrium advantage."
            )
        elif sym_best > best_conv_asym["team_payoff"] + EPSILON:
            verdict = "symmetric_ne_superior"
            verdict_detail = (
                f"Symmetric NE (payoff={sym_best:.2f}) outperforms best converged "
                f"asymmetric NE (payoff={best_conv_asym['team_payoff']:.2f}). "
                "Hero-like symmetric play is the dominant equilibrium."
            )
        else:
            verdict = "asymmetric_ne_exists_not_superior"
            verdict_detail = (
                f"Asymmetric NE (payoff={best_conv_asym['team_payoff']:.2f}) and "
                f"symmetric NE (payoff={sym_best:.2f}) are within ε={EPSILON} of each other."
            )
    elif best_conv_asym is not None:
        verdict = "asymmetric_only"
        verdict_detail = (
            f"All {len(converged)} converged equilibria are asymmetric "
            f"(best payoff={best_conv_asym['team_payoff']:.2f}). "
            "No symmetric NE exists — role differentiation is required."
        )
    elif best_conv_sym is not None:
        verdict = "symmetric_only"
        verdict_detail = (
            f"All {len(converged)} converged equilibria are symmetric "
            f"(best payoff={sym_best:.2f}). "
            "Role-differentiated specialization is not a Nash equilibrium."
        )
    elif asymmetric:
        verdict = "asymmetric_profile_found_not_converged"
        verdict_detail = (
            "Asymmetric profiles were found during search but none converged. "
            "Increase restarts/iterations for a definitive answer."
        )
    else:
        verdict = "no_convergence"
        verdict_detail = "No restarts converged within max_iterations."

    out = {
        "scenario": r["scenario"],
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "elapsed_seconds": r["timing"]["elapsed_seconds"],
        "converged": len(converged),
        "total_restarts": len(equilibria),
        "symmetric_profiles": len(symmetric),
        "asymmetric_profiles": len(asymmetric),
        "converged_asymmetric_profiles": len(converged_asymmetric),
        "converged_symmetric_profiles": len(converged_symmetric),
        "best_team_payoff": max(e["team_payoff"] for e in equilibria),
        "best_converged_symmetric_payoff": sym_best,
        "best_converged_asymmetric_payoff": (
            best_conv_asym["team_payoff"] if best_conv_asym else None
        ),
    }

    with open(scenario_dir / "summary.json", "w") as f:
        json.dump(out, f, indent=2)

    # --- Markdown ---
    lines = [
        f"# Heterogeneous Nash — {r['scenario']}\n",
        f"**Verdict**: `{verdict}`\n",
        f"> {verdict_detail}\n",
        "## Run configuration\n",
        f"- restarts={r['parameters']['num_restarts']}, "
        f"simulations={r['parameters']['num_simulations']}, "
        f"max_iter={r['parameters']['max_iterations']}, "
        f"ε={r['parameters']['epsilon']}, seed={r['parameters']['seed']}\n",
        f"- Elapsed: {r['timing']['elapsed_seconds']:.0f}s\n",
        "## Results overview\n",
        "| | Count |\n|---|---|\n"
        f"| Total restarts | {len(equilibria)} |\n"
        f"| Converged | {len(converged)} |\n"
        f"| Symmetric profiles | {len(symmetric)} |\n"
        f"| Asymmetric profiles | {len(asymmetric)} |\n"
        f"| Converged symmetric | {len(converged_symmetric)} |\n"
        f"| Converged asymmetric | {len(converged_asymmetric)} |\n",
    ]

    if best_conv_sym:
        lines += [
            "## Best converged symmetric equilibrium\n",
            f"- Team payoff: **{best_conv_sym['team_payoff']:.2f}**\n",
            f"- Profile: `{best_conv_sym['profile_label']}`\n",
            f"- Iterations: {best_conv_sym['iterations']}\n",
        ]

    if best_conv_asym:
        lines += [
            "## Best converged asymmetric equilibrium\n",
            f"- Team payoff: **{best_conv_asym['team_payoff']:.2f}**\n",
            f"- Profile: `{best_conv_asym['profile_label']}`\n",
            f"- Iterations: {best_conv_asym['iterations']}\n",
            "\n| Position | Payoff | Closest archetype | work_tendency |\n"
            "|---|---|---|---|\n",
        ]
        for i, s in enumerate(best_conv_asym["strategy_profile"]):
            wt = s["parameters"]["work_tendency"]
            payoff = best_conv_asym["per_position_payoffs"][i]
            lines.append(
                f"| {i} | {payoff:.1f} | {s['closest_archetype']} | {wt:.3f} |\n"
            )

    with open(scenario_dir / "summary.md", "w") as f:
        f.writelines(lines)

    print(f"{r['scenario']:30s}  verdict={verdict}")


if __name__ == "__main__":
    base = Path(__file__).parent
    for d in sorted(base.iterdir()):
        if d.is_dir() and (d / "results.json").exists():
            regen(d)
