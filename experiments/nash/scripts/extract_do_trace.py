#!/usr/bin/env python3
"""Extract the per-iteration improvement trace from a ``compute_nash.py`` log.

The Double Oracle driver (``experiments/scripts/compute_nash.py``) prints a
verbose per-iteration block::

    --- Iteration 12 ---
    Computing payoff matrix (20x20)...
    Solving restricted game...
    Equilibrium support size: 6
      Strategy 0: 0.349
      ...
    Equilibrium payoff: 1905.04
    Computing best response...
    Best response payoff: 1943.14
    Improvement: 38.0978
    Added best response to pool (new size: 21)

This script parses that block structure into a tidy per-iteration table
(iteration, pool size, support size, equilibrium payoff, best-response
payoff, improvement, support composition) and writes it as CSV, optionally
also emitting a markdown table. Committed so non-converged runs (issue #445 /
#352: rest_trap Double Oracle cycling) leave an auditable improvement-metric
trace next to their ``equilibrium.json``.

Usage:
    uv run python experiments/nash/scripts/extract_do_trace.py \
        experiments/nash/rest_trap_seeded_do/nash-resttrap-seeded.log \
        --csv experiments/nash/rest_trap_seeded_do/do_trace.csv \
        --markdown
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ITERATION_RE = re.compile(r"^--- Iteration (\d+) ---")
MATRIX_RE = re.compile(r"^Computing payoff matrix \((\d+)x(\d+)\)")
SUPPORT_SIZE_RE = re.compile(r"^Equilibrium support size: (\d+)")
SUPPORT_ENTRY_RE = re.compile(r"^\s+Strategy (\d+): ([0-9.]+)")
EQ_PAYOFF_RE = re.compile(r"^Equilibrium payoff: (-?[0-9.]+)")
BR_PAYOFF_RE = re.compile(r"^Best response payoff: (-?[0-9.]+)")
IMPROVEMENT_RE = re.compile(r"^Improvement: (-?[0-9.]+)")

FIELDS = [
    "iteration",
    "pool_size",
    "support_size",
    "eq_payoff",
    "br_payoff",
    "improvement",
    "support",
]


def parse_do_log(text: str) -> list[dict]:
    """Parse a verbose ``compute_nash.py`` log into per-iteration rows.

    Returns a list of dicts with keys ``iteration`` (int), ``pool_size``
    (int, size of the strategy pool whose payoff matrix was computed),
    ``support_size`` (int), ``eq_payoff`` / ``br_payoff`` / ``improvement``
    (float), and ``support`` (str, ``"idx:prob|idx:prob|..."`` composition
    of the restricted-game equilibrium).

    Incomplete trailing blocks (e.g. a run killed mid-iteration) are
    dropped: a row is emitted only once its ``Improvement:`` line is seen.
    """
    rows: list[dict] = []
    current: dict | None = None
    support_parts: list[str] = []

    for line in text.splitlines():
        m = ITERATION_RE.match(line)
        if m:
            current = {"iteration": int(m.group(1))}
            support_parts = []
            continue
        if current is None:
            continue

        m = MATRIX_RE.match(line)
        if m:
            current["pool_size"] = int(m.group(1))
            continue
        m = SUPPORT_SIZE_RE.match(line)
        if m:
            current["support_size"] = int(m.group(1))
            continue
        m = SUPPORT_ENTRY_RE.match(line)
        if m:
            support_parts.append(f"{m.group(1)}:{m.group(2)}")
            continue
        m = EQ_PAYOFF_RE.match(line)
        if m:
            current["eq_payoff"] = float(m.group(1))
            continue
        m = BR_PAYOFF_RE.match(line)
        if m:
            current["br_payoff"] = float(m.group(1))
            continue
        m = IMPROVEMENT_RE.match(line)
        if m:
            current["improvement"] = float(m.group(1))
            current["support"] = "|".join(support_parts)
            missing = [k for k in FIELDS if k not in current]
            if missing:
                raise ValueError(
                    f"Iteration {current.get('iteration')} block is missing "
                    f"fields {missing} despite an Improvement line"
                )
            rows.append(current)
            current = None
            continue

    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict]) -> str:
    lines = [
        "| Iter | Pool | Support | Eq payoff | BR payoff | Improvement |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['iteration']} | {r['pool_size']} | {r['support_size']} "
            f"| {r['eq_payoff']:.2f} | {r['br_payoff']:.2f} "
            f"| {r['improvement']:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Double Oracle improvement trace from a compute_nash.py log."
    )
    parser.add_argument("log", type=Path, help="Path to the verbose run log")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV output path (default: <log dir>/do_trace.csv)",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Also print the trace as a markdown table on stdout",
    )
    args = parser.parse_args()

    text = args.log.read_text()
    rows = parse_do_log(text)
    if not rows:
        print(f"ERROR: no complete Double Oracle iterations found in {args.log}")
        sys.exit(2)

    csv_path = args.csv if args.csv is not None else args.log.parent / "do_trace.csv"
    write_csv(rows, csv_path)

    improvements = [r["improvement"] for r in rows]
    eq_payoffs = [r["eq_payoff"] for r in rows]
    best = min(rows, key=lambda r: r["improvement"])
    print(f"Parsed {len(rows)} iterations from {args.log}")
    print(f"Wrote {csv_path}")
    print(
        f"Improvement: min={min(improvements):.4f} (iter {best['iteration']}) "
        f"max={max(improvements):.4f} final={improvements[-1]:.4f}"
    )
    print(f"Eq payoff range: [{min(eq_payoffs):.2f}, {max(eq_payoffs):.2f}]")

    if args.markdown:
        print()
        print(markdown_table(rows))


if __name__ == "__main__":
    main()
