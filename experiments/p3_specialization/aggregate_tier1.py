"""Aggregate Tier-1 cell summaries into a single verdict table.

Walks ``--tier1-root/*/cell_summary.json`` (one per ``<trainer>_<scenario>``
cell) and produces:

* ``tier1_verdict.md`` — a markdown table sorted by mean ``gap_closed``
  descending, with the verdict tier per #343's threshold ladder.
* ``tier1_verdict.json`` — the same data in machine-readable form for
  programmatic consumption.

Verdict thresholds (mirrored from ``run_tier1_cell.VERDICT_THRESHOLDS``;
historical 4-tier ladder from
``experiments/p3_specialization/diagnostics/random_mlp_search.py``):

* ``gap_closed_mean >= 0.88`` -> ``closed`` (stunning_near_specialist)
* ``0.49 <= gap_closed_mean < 0.88`` -> ``partial_upper`` (anti-attractor confirmed)
* ``0.20 <= gap_closed_mean < 0.49`` -> ``partial_lower`` (basin-trap consistent)
* ``gap_closed_mean < 0.20`` -> ``insufficient`` (random-play basin)

Since #434 ``gap_closed_mean`` may be null: scenarios whose reference pair
is degenerate (``verdict_tier = "not_scored_degenerate_reference"``, e.g.
rest_trap's NE-below-random social trap) or missing
(``verdict_tier = "not_scored"``) are never classified on the fraction
ladder. Such rows sort between ``insufficient`` and ``no_data`` and report
the scenario-scale ``uplift_over_random`` column instead.

Since #436 degenerate-reference rows additionally carry a categorical
``trap_verdict`` (``trapped_at_ne`` / ``at_random`` / ``escaped_trap`` /
``above_scripted_best``, computed by ``run_tier1_cell.classify_trap_verdict``
from the seed-bootstrap CI vs the NE / random / scripted_best anchors),
rendered as its own table column (``n/a`` for every other row).

Cells with ``n_seeds_completed == 0`` (or missing ``cell_summary.json``)
are reported as ``no_data`` and sorted to the bottom.

If ``<tier1-root>/tier1_verdict_notes.md`` exists, its contents are appended
verbatim to the generated markdown. This lets caveats about specific rows
(e.g. metric-scale artifacts) survive regeneration instead of being wiped
every time the aggregator reruns.

Usage:

    uv run python experiments/p3_specialization/aggregate_tier1.py \\
        --tier1-root experiments/p3_specialization/tier1_runs/
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TIER1_ROOT = REPO_ROOT / "experiments" / "p3_specialization" / "tier1_runs"

VERDICT_THRESHOLDS = (0.20, 0.49, 0.88)

# Optional operator-maintained notes appended to the generated markdown
# (relative to --tier1-root). Survives regeneration by construction.
NOTES_FILENAME = "tier1_verdict_notes.md"


def verdict_for(mean) -> tuple[str, str]:
    """Classify a scored ``gap_closed_mean`` on the fraction ladder.

    Null/NaN means are never classified on the ladder (#434): rows carrying
    their own ``not_scored*`` verdict keep it (see :func:`_sort_key`); this
    fallback classifier maps null to ``no_data``.
    """
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "no_data", "no completed seeds"
    low, mid, high = VERDICT_THRESHOLDS
    if mean >= high:
        return "closed", f"gap_closed_mean = {mean:.3f} >= {high}"
    if mean >= mid:
        return "partial_upper", f"{mid} <= gap_closed_mean < {high}"
    if mean >= low:
        return "partial_lower", f"{low} <= gap_closed_mean < {mid}"
    return "insufficient", f"gap_closed_mean = {mean:.3f} < {low}"


def _verdict_rank(tier: str) -> int:
    # ``not_scored*`` tiers (#434: degenerate/missing reference pair) sort
    # between ``insufficient`` and ``no_data`` — they have data, just no
    # meaningful fraction.
    return {
        "closed": 0,
        "partial_upper": 1,
        "partial_lower": 2,
        "insufficient": 3,
        "not_scored_degenerate_reference": 4,
        "not_scored": 5,
        "no_data": 6,
    }.get(tier, 7)


def load_cells(tier1_root: Path) -> list[dict]:
    """Walk ``tier1_root/<cell>/cell_summary.json`` and return the dicts.

    Cells without a ``cell_summary.json`` are still surfaced (as
    ``no_data`` placeholders) so the aggregated table makes the missing
    cells obvious instead of silently dropping them.
    """
    rows: list[dict] = []
    if not tier1_root.exists():
        return rows
    for cell_dir in sorted(p for p in tier1_root.iterdir() if p.is_dir()):
        summary_path = cell_dir / "cell_summary.json"
        if not summary_path.exists():
            rows.append(
                {
                    "cell_dir": cell_dir.name,
                    "trainer": cell_dir.name.split("_", 1)[0],
                    "scenario": cell_dir.name.split("_", 1)[-1]
                    if "_" in cell_dir.name
                    else "unknown",
                    "gap_closed_mean": float("nan"),
                    "gap_closed_std": 0.0,
                    "n_seeds_completed": 0,
                    "n_seeds_failed": 0,
                    "verdict_tier": "no_data",
                    "verdict_reason": "cell_summary.json missing",
                }
            )
            continue
        try:
            rows.append(json.loads(summary_path.read_text()))
        except (OSError, json.JSONDecodeError) as exc:
            rows.append(
                {
                    "cell_dir": cell_dir.name,
                    "trainer": cell_dir.name,
                    "scenario": "unknown",
                    "gap_closed_mean": float("nan"),
                    "gap_closed_std": 0.0,
                    "n_seeds_completed": 0,
                    "n_seeds_failed": 0,
                    "verdict_tier": "no_data",
                    "verdict_reason": f"failed to parse cell_summary.json: {exc}",
                }
            )
    return rows


def _sort_key(row: dict) -> tuple[int, float]:
    """Sort closed > partial > insufficient > not_scored* > no_data; within
    a tier, by mean desc.

    Null/NaN-mean rows sort by their own ``verdict_tier`` rank (#434:
    ``not_scored*`` rows carry data but no fraction, so they must not be
    conflated with ``no_data``), using ``-uplift_over_random_mean`` as the
    within-tier secondary key when present.
    """
    tier = row.get("verdict_tier", "no_data")
    mean = row.get("gap_closed_mean", None)
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        uplift = row.get("uplift_over_random_mean", None)
        if uplift is None or (isinstance(uplift, float) and math.isnan(uplift)):
            uplift = 0.0
        return (_verdict_rank(tier), -float(uplift))
    # Negate for descending sort.
    return (_verdict_rank(tier), -float(mean))


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.3f}"
    return str(value)


def _fmt_uplift(mean: object, std: object) -> str:
    """Format ``uplift_over_random`` as ``+m.mmm ± s.sss`` (or ``n/a``).

    Pre-#434 (schema v1) summaries carry no uplift fields; those rows render
    ``n/a`` rather than crashing.
    """
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "n/a"
    out = f"{float(mean):+.3f}"
    if std is not None and not (isinstance(std, float) and math.isnan(std)):
        out += f" ± {float(std):.3f}"
    return out


def build_markdown(rows: Sequence[dict], notes: Optional[str] = None) -> str:
    sorted_rows = sorted(rows, key=_sort_key)
    lines = [
        "# Tier-1 sweep verdict",
        "",
        "Verdict ladder: `gap_closed_mean >= 0.88` -> **closed**; "
        "`0.49 <= mean < 0.88` -> **partial_upper**; "
        "`0.20 <= mean < 0.49` -> **partial_lower**; "
        "`mean < 0.20` -> **insufficient**. "
        "Rows with a null `gap_closed` (**not_scored** / "
        "**not_scored_degenerate_reference**, #434) are never classified on "
        "the ladder; read their `uplift_over_random` (per-step, scenario "
        "scale) and the categorical `trap_verdict` (#436: seed-bootstrap "
        "95% CI vs NE / random / scripted_best anchors -> `trapped_at_ne` / "
        "`at_random` / `escaped_trap` / `above_scripted_best`) instead.",
        "",
        "| Trainer | Scenario | gap_closed (mean ± std) | "
        "uplift_over_random (mean ± std) | Trap verdict | n_seeds | Verdict |",
        "|---------|----------|--------------------------|"
        "---------------------------------|--------------|---------|---------|",
    ]
    for row in sorted_rows:
        mean = row.get("gap_closed_mean", float("nan"))
        std = row.get("gap_closed_std", 0.0)
        uplift_mean = row.get("uplift_over_random_mean", None)
        uplift_std = row.get("uplift_over_random_std", None)
        n_ok = row.get("n_seeds_completed", 0)
        n_fail = row.get("n_seeds_failed", 0)
        n_str = f"{n_ok} ok"
        if n_fail:
            n_str += f", {n_fail} failed"
        gap_str = "n/a" if _fmt(mean) == "n/a" else f"{_fmt(mean)} ± {_fmt(std)}"
        lines.append(
            "| {trainer} | {scenario} | {gap} | {uplift} | {trap} | "
            "{n_str} | {verdict} |".format(
                trainer=row.get("trainer", "?"),
                scenario=row.get("scenario", "?"),
                gap=gap_str,
                uplift=_fmt_uplift(uplift_mean, uplift_std),
                trap=row.get("trap_verdict") or "n/a",
                n_str=n_str,
                verdict=row.get("verdict_tier", "no_data"),
            )
        )
    lines.append("")
    if notes:
        lines.append(notes.rstrip("\n"))
        lines.append("")
    return "\n".join(lines)


def build_json(rows: Sequence[dict]) -> dict:
    sorted_rows = sorted(rows, key=_sort_key)
    return {
        # v2 (#434): added gap_source, uplift_over_random_* and
        # gap_closed_minspec_legacy_mean; gap_closed_mean may be null for
        # not_scored* rows (degenerate/missing reference pair).
        # v3 (#436): added trap_verdict / trap_verdict_reason for
        # degenerate-reference rows (null everywhere else).
        "schema_version": 3,
        "thresholds": {
            "partial_lower": VERDICT_THRESHOLDS[0],
            "partial_upper": VERDICT_THRESHOLDS[1],
            "closed": VERDICT_THRESHOLDS[2],
        },
        "cells": [
            {
                "trainer": r.get("trainer", "?"),
                "scenario": r.get("scenario", "?"),
                "gap_closed_mean": r.get("gap_closed_mean", float("nan")),
                "gap_closed_std": r.get("gap_closed_std", 0.0),
                "gap_closed_per_seed": r.get("gap_closed_per_seed", []),
                "gap_source": r.get("gap_source"),
                "uplift_over_random_mean": r.get("uplift_over_random_mean"),
                "uplift_over_random_std": r.get("uplift_over_random_std"),
                "trap_verdict": r.get("trap_verdict"),
                "trap_verdict_reason": r.get("trap_verdict_reason"),
                "gap_closed_minspec_legacy_mean": r.get(
                    "gap_closed_minspec_legacy_mean"
                ),
                "n_seeds_completed": r.get("n_seeds_completed", 0),
                "n_seeds_failed": r.get("n_seeds_failed", 0),
                "verdict_tier": r.get("verdict_tier", "no_data"),
                "verdict_reason": r.get("verdict_reason", ""),
            }
            for r in sorted_rows
        ],
    }


def _safe_json_dump(obj: object) -> str:
    """``json.dumps`` with NaN -> ``null`` so output is portable."""

    def _replace(o):
        if isinstance(o, float) and math.isnan(o):
            return None
        if isinstance(o, dict):
            return {k: _replace(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_replace(x) for x in o]
        return o

    return json.dumps(_replace(obj), indent=2)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tier1-root",
        type=Path,
        default=DEFAULT_TIER1_ROOT,
        help="Root of tier-1 cell outputs (default: experiments/p3_specialization/tier1_runs/).",
    )
    p.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Markdown verdict path (default: <tier1-root>/tier1_verdict.md).",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="JSON verdict path (default: <tier1-root>/tier1_verdict.json).",
    )
    args = p.parse_args(argv)

    rows = load_cells(args.tier1_root)
    if not rows:
        print(
            f"WARN: no cells found under {args.tier1_root}. "
            "Run run_tier1_cell.py first."
        )

    md_path = args.output_md or (args.tier1_root / "tier1_verdict.md")
    json_path = args.output_json or (args.tier1_root / "tier1_verdict.json")

    notes_path = args.tier1_root / NOTES_FILENAME
    notes = notes_path.read_text() if notes_path.exists() else None

    args.tier1_root.mkdir(parents=True, exist_ok=True)
    md_path.write_text(build_markdown(rows, notes=notes))
    json_path.write_text(_safe_json_dump(build_json(rows)))

    print(f"Wrote {md_path} ({len(rows)} cells)")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
