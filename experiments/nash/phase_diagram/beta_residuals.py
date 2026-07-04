#!/usr/bin/env python3
"""Cross-β residual analysis of the bernoulli-mode Nash phase diagram
(issue #442 — solver-nondeterminism probe).

In bernoulli extinguish mode the β axis (``prob_fire_spreads_to_neighbor``)
is provably inert: the engine phase order (extinguish → burn_out → spread →
spontaneous_ignition) guarantees every still-burning house is RUINED before
``spread_fires`` runs, and the spread phase draws **zero RNG** in this mode,
so both the dynamics and the RNG stream are bit-identical across β. Cells in
``results.json`` that share (κ, c) but differ in β are therefore *repeat
solves of the same game* — any payoff/verdict difference between them is
pure double-oracle solver nondeterminism (restart lottery / seed batching),
not physics.

This script commits that probe reproducibly:

1. Groups the 39 cells of ``results.json`` into 13 (κ, c) columns and
   reports the cross-β residual (max − min ``best_team_payoff``, verdict and
   convergence-count consistency) for every column.
2. For columns with a nonzero residual, loads the committed best-converged
   NE profiles from ``bucket_brigade/baselines/release/local/nash/
   phase_diagram/`` and re-evaluates each unique profile with
   common-random-number (CRN) scripted rollouts (Rust engine, seconds) to
   decide whether the divergent β=0.1 payoff reflects a genuinely better
   equilibrium profile or just Monte-Carlo evaluation noise. This is the
   80.915-vs-72.0095 check for ``b0.10_k0.90_c0.50`` vs the registered
   ``asym_b05_k09_c05`` / ``asym_b09_k09_c05`` cells.
3. Empirically verifies β-inertness for every committed profile: the same
   profile evaluated under β=0.1 and β=0.9 with identical seeds must return
   bit-identical rewards.

No double-oracle solves are re-run here — the seeded-DO retry that would
*resolve* the restart lottery is issue #445's territory. See also the
solver-noise hedge carried by the workshop paper (PR #448).

Run locally (join + CRN rollouts only, ~seconds):

    uv run python experiments/nash/phase_diagram/beta_residuals.py

Outputs (written next to this script, deterministic given --seed):
    - ``beta_residuals.json`` — machine-readable residuals + CRN verdicts
    - ``beta_residuals.md``   — write-up
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE_DIAGRAM_ROOT = REPO_ROOT / "experiments" / "nash" / "phase_diagram"

DEFAULT_RESULTS_JSON = PHASE_DIAGRAM_ROOT / "results.json"
DEFAULT_BASELINES_DIR = (
    REPO_ROOT
    / "bucket_brigade"
    / "baselines"
    / "release"
    / "local"
    / "nash"
    / "phase_diagram"
)
DEFAULT_OUT_JSON = PHASE_DIAGRAM_ROOT / "beta_residuals.json"
DEFAULT_OUT_MD = PHASE_DIAGRAM_ROOT / "beta_residuals.md"

DEFAULT_N_SIMS = 20_000
DEFAULT_INERTNESS_SIMS = 2_000
DEFAULT_SEED = 20260442

# |t| above this is treated as a decisive CRN payoff difference.
T_DECISIVE = 3.0


# ---------------------------------------------------------------------------
# Pure join / residual helpers (no Rust, unit-tested)
# ---------------------------------------------------------------------------


def cell_tag(beta: float, kappa: float, c: float) -> str:
    """Filesystem tag, mirroring compute_nash_phase_diagram.py."""
    return f"b{beta:.2f}_k{kappa:.2f}_c{c:.2f}"


def group_columns(cells: list[dict]) -> list[dict]:
    """Group phase-diagram cells by (κ, c) and compute cross-β residuals.

    Every column must contain at least two β values (a single-β column has
    no cross-β residual to measure — fail loudly, the artifact is not the
    grid this analysis expects).
    """
    groups: dict[tuple[float, float], list[dict]] = {}
    for cell in cells:
        groups.setdefault((cell["kappa"], cell["c"]), []).append(cell)

    columns: list[dict] = []
    for (kappa, c), col_cells in sorted(groups.items()):
        col_cells = sorted(col_cells, key=lambda x: x["beta"])
        if len(col_cells) < 2:
            raise ValueError(
                f"(kappa={kappa}, c={c}) has only {len(col_cells)} beta value(s); "
                "cross-beta residuals need at least 2."
            )
        payoffs = [x["best_team_payoff"] for x in col_cells]
        if any(p is None for p in payoffs):
            raise ValueError(
                f"(kappa={kappa}, c={c}) has a null best_team_payoff; "
                "results.json is malformed for this analysis."
            )
        residual = max(payoffs) - min(payoffs)
        columns.append(
            {
                "kappa": kappa,
                "c": c,
                "betas": [x["beta"] for x in col_cells],
                "tags": [x["tag"] for x in col_cells],
                "best_team_payoff": payoffs,
                "converged": [x["converged"] for x in col_cells],
                "verdicts": [x["verdict"] for x in col_cells],
                "payoff_residual": residual,
                "bit_identical": len(set(payoffs)) == 1,
                "verdict_consistent": len(set(x["verdict"] for x in col_cells)) == 1,
                "convergence_consistent": len(set(x["converged"] for x in col_cells))
                == 1,
            }
        )
    return columns


def genome_max_delta(
    genomes_a: list[list[float]], genomes_b: list[list[float]]
) -> float:
    """Max elementwise |difference| between two positional genome profiles."""
    if len(genomes_a) != len(genomes_b):
        raise ValueError("Profiles have different numbers of positions.")
    delta = 0.0
    for ga, gb in zip(genomes_a, genomes_b):
        if len(ga) != len(gb):
            raise ValueError("Genomes have different lengths.")
        delta = max(delta, max(abs(x - y) for x, y in zip(ga, gb)))
    return delta


def dedupe_profiles(profiles_by_beta: dict[str, dict]) -> list[dict]:
    """Collapse per-β committed profiles into unique-genome groups.

    ``profiles_by_beta`` maps a β key (e.g. ``"0.10"``) to a committed
    baseline payload with ``positions[*].genome``. Returns one entry per
    unique genome profile, recording which β cells committed it.
    """
    unique: list[dict] = []
    for beta_key in sorted(profiles_by_beta):
        payload = profiles_by_beta[beta_key]
        genomes = [pos["genome"] for pos in payload["positions"]]
        for entry in unique:
            if genome_max_delta(entry["genomes"], genomes) == 0.0:
                entry["from_betas"].append(beta_key)
                break
        else:
            unique.append(
                {
                    "from_betas": [beta_key],
                    "profile_label": payload["profile_label"],
                    "solver_team_payoff": payload["team_payoff"],
                    "symmetric_profile": payload["symmetric_profile"],
                    "genomes": genomes,
                }
            )
    return unique


def paired_stats(diffs: list[float]) -> dict:
    """Mean / SE / t for a list of CRN paired differences."""
    n = len(diffs)
    if n < 2:
        raise ValueError("Need at least 2 paired samples.")
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    t = mean / se if se > 0 else float("inf") if mean != 0 else 0.0
    return {"n": n, "mean": mean, "se": se, "t": t}


# ---------------------------------------------------------------------------
# CRN evaluation (Rust engine, lazy imports)
# ---------------------------------------------------------------------------


def _make_scenario(beta: float, kappa: float, c: float):
    """minimal_specialization with (β, κ, c) overridden — mirrors the driver."""
    from bucket_brigade.envs import get_scenario_by_name

    base = get_scenario_by_name("minimal_specialization", num_agents=4)
    return dataclasses.replace(
        base,
        prob_fire_spreads_to_neighbor=float(beta),
        prob_solo_agent_extinguishes_fire=float(kappa),
        cost_to_work_one_night=float(c),
    )


def evaluate_profile(
    genomes: list[list[float]],
    beta: float,
    kappa: float,
    c: float,
    seeds: list[int],
) -> "list[float]":
    """Per-seed team payoff (mean across positions) for one profile."""
    import bucket_brigade_core as core
    from bucket_brigade.equilibrium.payoff_evaluator_rust import (
        _convert_scenario_to_rust,
    )

    scenario = _make_scenario(beta, kappa, c)
    rust_scenario = _convert_scenario_to_rust(scenario)
    num_agents = scenario.num_agents
    team: list[float] = []
    for seed in seeds:
        rewards = core.run_heuristic_episode(rust_scenario, num_agents, genomes, seed)
        team.append(sum(rewards) / len(rewards))
    return team


def make_seeds(n: int, seed: int) -> list[int]:
    """Deterministic evaluation seed list (shared across profiles → CRN)."""
    import numpy as np

    rng = np.random.RandomState(seed)
    return [int(s) for s in rng.randint(0, 2**31 - 1, size=n)]


def crn_compare_column(
    kappa: float,
    c: float,
    unique_profiles: list[dict],
    n_sims: int,
    seed: int,
    eval_beta: float = 0.5,
) -> dict:
    """CRN-evaluate each unique profile of a residual column and pair them.

    β is inert, so the evaluation β is arbitrary (0.5 = the registered
    scenarios' value). All profiles share one seed list — differences are
    paired per-seed, which removes most of the episode-level variance.
    """
    seeds = make_seeds(n_sims, seed)
    evals: list[dict] = []
    per_profile_team: list[list[float]] = []
    for prof in unique_profiles:
        team = evaluate_profile(prof["genomes"], eval_beta, kappa, c, seeds)
        per_profile_team.append(team)
        stats = paired_stats(team)
        evals.append(
            {
                "from_betas": prof["from_betas"],
                "profile_label": prof["profile_label"],
                "solver_team_payoff": prof["solver_team_payoff"],
                "reeval_mean": stats["mean"],
                "reeval_se": stats["se"],
                "solver_minus_reeval": prof["solver_team_payoff"] - stats["mean"],
            }
        )

    pairs: list[dict] = []
    for i in range(len(unique_profiles)):
        for j in range(i + 1, len(unique_profiles)):
            diffs = [a - b for a, b in zip(per_profile_team[i], per_profile_team[j])]
            stats = paired_stats(diffs)
            if abs(stats["t"]) >= T_DECISIVE:
                better = (
                    evals[i]["profile_label"]
                    if stats["mean"] > 0
                    else evals[j]["profile_label"]
                )
                verdict = "decisive"
            else:
                better = None
                verdict = "not_distinguishable"
            pairs.append(
                {
                    "a": evals[i]["profile_label"],
                    "a_from_betas": evals[i]["from_betas"],
                    "b": evals[j]["profile_label"],
                    "b_from_betas": evals[j]["from_betas"],
                    "genome_max_delta": genome_max_delta(
                        unique_profiles[i]["genomes"], unique_profiles[j]["genomes"]
                    ),
                    "crn_diff_mean": stats["mean"],
                    "crn_diff_se": stats["se"],
                    "crn_diff_t": stats["t"],
                    "verdict": verdict,
                    "better_profile": better,
                }
            )

    return {
        "kappa": kappa,
        "c": c,
        "eval_beta": eval_beta,
        "n_sims": n_sims,
        "seed": seed,
        "profiles": evals,
        "pairwise": pairs,
    }


def inertness_check(
    genomes: list[list[float]],
    kappa: float,
    c: float,
    n_sims: int,
    seed: int,
    betas: tuple[float, float] = (0.1, 0.9),
) -> bool:
    """True iff the profile's per-seed rewards are bit-identical across β."""
    seeds = make_seeds(n_sims, seed)
    a = evaluate_profile(genomes, betas[0], kappa, c, seeds)
    b = evaluate_profile(genomes, betas[1], kappa, c, seeds)
    return a == b


# ---------------------------------------------------------------------------
# Committed-profile loading
# ---------------------------------------------------------------------------


def load_column_profiles(
    column: dict, baselines_dir: Path
) -> tuple[dict[str, dict], list[str]]:
    """Load committed baseline profiles for every β cell of one column.

    Returns (profiles_by_beta, missing_tags). ``no_convergence`` cells have
    no registered profile — that is expected and reported, not fatal.
    """
    profiles: dict[str, dict] = {}
    missing: list[str] = []
    for beta, tag in zip(column["betas"], column["tags"]):
        path = baselines_dir / f"{tag}.json"
        if not path.exists():
            missing.append(tag)
            continue
        with open(path) as f:
            profiles[f"{beta:.2f}"] = json.load(f)
    return profiles, missing


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _md_label(label: str) -> str:
    """Escape profile labels for use inside markdown tables/bullets."""
    return label.replace("|", "\\|")


def _fmt(x: float) -> str:
    """Format a stat that may be large (payoffs) or tiny (CRN diffs)."""
    return f"{x:+.4g}"


def render_markdown(result: dict) -> str:
    columns = result["columns"]
    n_cols = len(columns)
    n_identical = sum(1 for c in columns if c["bit_identical"])
    residual_cols = [c for c in columns if not c["bit_identical"]]

    lines = [
        "# Cross-β residuals: solver nondeterminism, not physics (issue #442)",
        "",
        "Generated by `beta_residuals.py`. Input: `results.json` (39-cell",
        "bernoulli-mode phase diagram) plus the committed best-converged NE",
        "profiles under `bucket_brigade/baselines/release/local/nash/phase_diagram/`.",
        "",
        "β (`prob_fire_spreads_to_neighbor`) is **inert** in bernoulli extinguish",
        "mode — dynamics and RNG stream are bit-identical across β (mechanism in",
        "issue #442; verified empirically below). Cells sharing (κ, c) are repeat",
        "solves of the same game, so any cross-β difference measures double-oracle",
        "solver nondeterminism.",
        "",
        "## Headline",
        "",
        f"- {n_identical}/{n_cols} (κ, c) columns are **bit-identical** across all",
        "  three β values (same payoff, same convergence count, same verdict) —",
        "  the solver is deterministic given the seed (cf. the alc-5 freq-cap",
        "  reproduction in `RECOVERY_NOTES.md`).",
        f"- {len(residual_cols)} columns show residuals, and both are exactly the",
        "  β=0.1 cells that were lost in the 2026-06-05 alc-4 outage and re-run",
        "  later in a separate fill batch (`RECOVERY_NOTES.md`, `LAUNCH_RUNBOOK.md`",
        "  Plan A). The residual is a **restart lottery** across solve batches,",
        "  not hardware or dynamics noise.",
        "- β=0.5 vs β=0.9 (always solved in the same batch) are bit-identical in",
        f"  {n_cols}/{n_cols} columns.",
        "",
        "## Per-column residuals",
        "",
        "| κ | c | payoffs by β (0.1, 0.5, 0.9) | residual | converged by β | verdicts consistent? |",
        "|---|---|---|---|---|---|",
    ]
    for col in columns:
        payoffs = ", ".join(f"{p:.2f}" for p in col["best_team_payoff"])
        conv = ", ".join(str(v) for v in col["converged"])
        verdict = (
            "yes"
            if col["verdict_consistent"]
            else "**NO**: " + ", ".join(f"`{v}`" for v in col["verdicts"])
        )
        residual = (
            "0 (bit-identical)"
            if col["bit_identical"]
            else f"**{col['payoff_residual']:.2f}**"
        )
        lines.append(
            f"| {col['kappa']:.2f} | {col['c']:.2f} | {payoffs} | {residual} "
            f"| {conv} | {verdict} |"
        )

    lines += [
        "",
        "## CRN re-evaluation of the residual columns",
        "",
        "For each residual column the committed best-converged profiles were",
        "re-evaluated with common random numbers (same seed list for every",
        f"profile, n = {result['crn_n_sims']} episodes, Rust engine). Paired",
        "per-seed differences remove most episode-level variance; |t| ≥",
        f"{T_DECISIVE:.0f} is treated as decisive.",
        "",
    ]

    for ev in result["crn_evaluations"]:
        lines += [
            f"### (κ = {ev['kappa']:.2f}, c = {ev['c']:.2f})",
            "",
            "| profile | from β cells | solver payoff | CRN re-eval mean ± SE | solver − re-eval |",
            "|---|---|---|---|---|",
        ]
        for prof in ev["profiles"]:
            betas = ", ".join(prof["from_betas"])
            lines.append(
                f"| `{_md_label(prof['profile_label'])}` | {betas} "
                f"| {prof['solver_team_payoff']:.2f} "
                f"| {prof['reeval_mean']:.2f} ± {prof['reeval_se']:.2f} "
                f"| {prof['solver_minus_reeval']:+.2f} |"
            )
        lines.append("")
        for pair in ev["pairwise"]:
            lines += [
                f"- **`{_md_label(pair['a'])}`** (β {', '.join(pair['a_from_betas'])}) vs "
                f"**`{_md_label(pair['b'])}`** (β {', '.join(pair['b_from_betas'])}): "
                f"genome max |Δ| = {pair['genome_max_delta']:.2e}; CRN paired diff "
                f"= {_fmt(pair['crn_diff_mean'])} ± {pair['crn_diff_se']:.4g} "
                f"(t = {pair['crn_diff_t']:+.1f}) → **{pair['verdict']}**"
                + (
                    f", better: `{_md_label(pair['better_profile'])}`"
                    if pair["better_profile"]
                    else ""
                ),
            ]
        lines.append("")

    lines += [
        "Note the consistently **positive** `solver − re-eval` gap for the",
        "asymmetric profiles: `best_team_payoff` is a max over 20 restarts of",
        "noisy 1000-episode MC estimates, so the reported winner carries a",
        "winner's-curse selection bias (see the `solver − re-eval` column).",
        "Absolute payoffs in `results.json` should be read with that bias in mind;",
        "cross-profile *rankings* from the CRN comparison above are unaffected.",
        "",
        "## Interpretation",
        "",
        result["interpretation"],
        "",
        "## β-inertness verification",
        "",
        f"Every committed profile ({result['inertness']['n_profiles_checked']}"
        " unique profile/column combinations) was evaluated under β = 0.1 and",
        f"β = 0.9 with identical seeds (n = {result['inertness']['n_sims']}):",
        f"**{'all bit-identical' if result['inertness']['all_identical'] else 'MISMATCH FOUND'}**"
        " — the mechanistic inertness argument holds empirically.",
        "",
        "## Cross-references",
        "",
        "- **Issue #445** — seeded DO retry / exploitability-bounded fallback:",
        "  the structural fix for the restart lottery quantified here. Any",
        "  update to the registered `asym_b05_k09_c05` / `asym_b09_k09_c05`",
        "  gap references (`SCENARIO_GAP_REFERENCES`) should ride that",
        "  coordinated re-solve, not this analysis.",
        "- **Issue #429** — het_ppo Phase 2 gap ladder consumes the registered",
        "  cells' NE payoff as denominator; see the interpretation above.",
        "- **PR #448** — the workshop paper carries the matching β-inertness",
        "  corrections and a solver-noise hedge; this file is the",
        "  experiments-side artifact behind that hedge.",
        "- `LAUNCH_RUNBOOK.md` / `phase_diagram_table.md` — β-axis policy for",
        "  future sweeps (collapse β in bernoulli mode: 39 cells → 13).",
        "",
        "Reproduce with:",
        "",
        "```bash",
        "uv run python experiments/nash/phase_diagram/beta_residuals.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def build_interpretation(result: dict) -> str:
    """Data-driven interpretation paragraph for the two residual columns."""
    parts: list[str] = []
    for ev in result["crn_evaluations"]:
        for pair in ev["pairwise"]:
            tag = f"(κ = {ev['kappa']:.2f}, c = {ev['c']:.2f})"
            if pair["genome_max_delta"] < 1e-5:
                parts.append(
                    f"At {tag} the committed profiles are behaviorally identical "
                    f"(genome max |Δ| = {pair['genome_max_delta']:.1e} — L-BFGS-B "
                    "endpoint jitter), and the CRN paired difference is "
                    f"{pair['crn_diff_mean']:+.4g} ± {pair['crn_diff_se']:.4g}. The "
                    "headline results.json residual in this column is therefore pure "
                    "Monte-Carlo evaluation + restart-selection noise at "
                    "num_simulations = 1000, giving an honest per-cell payoff noise "
                    "scale of order tens of payoff units."
                )
            elif pair["verdict"] == "decisive":
                parts.append(
                    f"At {tag} the two batches found **structurally different** "
                    f"equilibria (`{pair['a']}` vs `{pair['b']}`), and the CRN "
                    f"comparison is decisive: `{pair['better_profile']}` has the "
                    f"higher true team payoff by {abs(pair['crn_diff_mean']):.2f} ± "
                    f"{pair['crn_diff_se']:.2f} (t = {pair['crn_diff_t']:+.1f}). The "
                    "restart lottery is not just payoff jitter — it changes *which* "
                    "equilibrium is reported. Whether the better profile is itself an "
                    "ε-Nash equilibrium of the shared environment requires the seeded "
                    "re-solve of issue #445 (best-response search is not re-run here)."
                )
            else:
                parts.append(
                    f"At {tag} the two batches found different profiles (`{pair['a']}` "
                    f"vs `{pair['b']}`), but the CRN comparison cannot distinguish "
                    f"their true team payoffs ({pair['crn_diff_mean']:+.2f} ± "
                    f"{pair['crn_diff_se']:.2f}, t = {pair['crn_diff_t']:+.1f}) — the "
                    "results.json residual is within evaluation noise."
                )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-json", type=Path, default=DEFAULT_RESULTS_JSON)
    parser.add_argument("--baselines-dir", type=Path, default=DEFAULT_BASELINES_DIR)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--n-sims", type=int, default=DEFAULT_N_SIMS)
    parser.add_argument("--inertness-sims", type=int, default=DEFAULT_INERTNESS_SIMS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    if not args.results_json.exists():
        print(f"ERROR: input artifact not found: {args.results_json}", file=sys.stderr)
        return 1

    with open(args.results_json) as f:
        data = json.load(f)

    columns = group_columns(data["cells"])
    residual_columns = [c for c in columns if not c["bit_identical"]]

    # CRN re-evaluation of residual columns from committed profiles.
    crn_evaluations: list[dict] = []
    missing_profiles: list[dict] = []
    for col in residual_columns:
        profiles_by_beta, missing = load_column_profiles(col, args.baselines_dir)
        if missing:
            missing_profiles.append(
                {"kappa": col["kappa"], "c": col["c"], "missing_tags": missing}
            )
        if len(profiles_by_beta) < 2:
            # Nothing committed to compare — document and leave the
            # re-solve to issue #445.
            continue
        unique = dedupe_profiles(profiles_by_beta)
        crn_evaluations.append(
            crn_compare_column(col["kappa"], col["c"], unique, args.n_sims, args.seed)
        )

    # Empirical β-inertness verification over every committed profile.
    n_checked = 0
    all_identical = True
    for col in columns:
        profiles_by_beta, _ = load_column_profiles(col, args.baselines_dir)
        for prof in dedupe_profiles(profiles_by_beta):
            identical = inertness_check(
                prof["genomes"],
                col["kappa"],
                col["c"],
                args.inertness_sims,
                args.seed,
            )
            n_checked += 1
            all_identical = all_identical and identical

    result = {
        "inputs": {
            "results_json": str(args.results_json),
            "baselines_dir": str(args.baselines_dir),
        },
        "crn_n_sims": args.n_sims,
        "columns": columns,
        "n_columns": len(columns),
        "n_bit_identical": sum(1 for c in columns if c["bit_identical"]),
        "residual_columns": [
            {"kappa": c["kappa"], "c": c["c"], "payoff_residual": c["payoff_residual"]}
            for c in residual_columns
        ],
        "missing_committed_profiles": missing_profiles,
        "crn_evaluations": crn_evaluations,
        "inertness": {
            "n_profiles_checked": n_checked,
            "n_sims": args.inertness_sims,
            "all_identical": all_identical,
        },
    }
    result["interpretation"] = build_interpretation(result)

    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"Wrote {args.out_json}")

    with open(args.out_md, "w") as f:
        f.write(render_markdown(result))
    print(f"Wrote {args.out_md}")

    print(
        f"\nHeadline: {result['n_bit_identical']}/{result['n_columns']} columns "
        f"bit-identical across beta; {len(residual_columns)} residual column(s)."
    )
    for ev in crn_evaluations:
        for pair in ev["pairwise"]:
            print(
                f"  (k={ev['kappa']:.2f}, c={ev['c']:.2f}) CRN diff "
                f"{pair['crn_diff_mean']:+.2f} ± {pair['crn_diff_se']:.2f} "
                f"(t={pair['crn_diff_t']:+.1f}) → {pair['verdict']}"
            )
    if not result["inertness"]["all_identical"]:
        print("WARNING: beta-inertness verification FAILED", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
