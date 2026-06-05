"""
Tests for ``experiments/scripts/plot_phase_diagram.py``.

Constructs synthetic aggregate ``results.json`` payloads matching the schema
written by ``compute_nash_phase_diagram.py`` and asserts the plot script:

1. Writes both output files (PNG + markdown).
2. Produces non-empty PNG bytes.
3. Sorts the markdown table by ``(c, β, κ)`` and includes the per-cell verdict,
   equilibrium payoff, and convergence rate columns required by issue #380.
4. Handles the degenerate 1×1×1 ``--smoke`` case without erroring.

The script is loaded via ``importlib`` because ``experiments/scripts`` is not
a proper Python package (same convention as ``test_compute_nash_df_precheck``).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# ``matplotlib`` is intentionally not a declared dependency (see PR #110);
# scripts/tests that need it install it on demand. Skip the whole module
# cleanly when it is unavailable in the active environment.
pytest.importorskip("matplotlib")


REPO_ROOT = Path(__file__).resolve().parent.parent
PLOT_PATH = REPO_ROOT / "experiments" / "scripts" / "plot_phase_diagram.py"


def _load_plot_module():
    """Load ``plot_phase_diagram.py`` as a standalone module for testing."""
    spec = importlib.util.spec_from_file_location("plot_phase_diagram", PLOT_PATH)
    assert spec is not None and spec.loader is not None, (
        f"Could not load spec for {PLOT_PATH}"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["plot_phase_diagram"] = module
    spec.loader.exec_module(module)
    return module


def _make_cell(beta: float, kappa: float, c: float, verdict: str, payoff: float):
    """Construct a synthetic cell record matching the driver's schema."""
    return {
        "beta": beta,
        "kappa": kappa,
        "c": c,
        "tag": f"b{beta:.2f}_k{kappa:.2f}_c{c:.2f}",
        "elapsed_seconds": 1.0,
        "total_restarts": 5,
        "converged": 3,
        "symmetric_profiles": 1,
        "asymmetric_profiles": 4,
        "best_team_payoff": payoff,
        "best_symmetric_team_payoff": payoff - 10.0,
        "best_asymmetric_team_payoff": payoff,
        "best_asymmetric_profile_label": "coordinator | liar | hero | hero",
        "verdict": verdict,
        "verdict_detail": f"Synthetic {verdict} verdict for test.",
    }


def _make_aggregate(
    beta_values: list[float],
    kappa_values: list[float],
    c_values: list[float],
    cells: list[dict],
) -> dict:
    """Construct a synthetic aggregate payload matching the driver's schema."""
    return {
        "base_scenario": "minimal_specialization",
        "algorithm": "heterogeneous_double_oracle",
        "grid": {
            "beta_values": beta_values,
            "kappa_values": kappa_values,
            "c_values": c_values,
            "total_cells": len(cells),
        },
        "parameters": {
            "num_simulations": 1000,
            "opt_simulations": 300,
            "max_iterations": 25,
            "epsilon": 50.0,
            "num_restarts": 5,
            "seed": 42,
        },
        "timing": {"elapsed_seconds": 12.3, "skipped_cached": []},
        "verdict_counts": {},
        "cells": cells,
    }


@pytest.fixture
def plot_module():
    return _load_plot_module()


def test_plot_smoke_1x1x1(plot_module, tmp_path: Path) -> None:
    """The degenerate 1×1×1 smoke case must render without error."""
    cells = [_make_cell(0.5, 0.5, 0.5, "no_convergence", -1431.04)]
    aggregate = _make_aggregate([0.5], [0.5], [0.5], cells)

    results_path = tmp_path / "results.json"
    with open(results_path, "w") as f:
        json.dump(aggregate, f)

    out_png = tmp_path / "phase_diagram.png"
    out_md = tmp_path / "phase_diagram_table.md"

    data = plot_module.load_aggregate(results_path)
    plot_module.render_png(data, out_png)
    plot_module.render_markdown(data, out_md)

    assert out_png.exists(), "PNG output was not created"
    assert out_png.stat().st_size > 0, "PNG output is empty"
    assert out_md.exists(), "Markdown output was not created"
    md_text = out_md.read_text()
    assert "no_convergence" in md_text
    assert "0.50" in md_text  # cell coordinate appears
    assert "-1431.04" in md_text  # payoff appears


def test_plot_2x2x2_runs_and_writes_outputs(plot_module, tmp_path: Path) -> None:
    """A 2×2×2 grid renders both outputs with non-zero size."""
    beta_values = [0.1, 0.9]
    kappa_values = [0.1, 0.9]
    c_values = [0.5, 2.0]
    verdicts = ["symmetric_only", "asymmetric_only", "mixed", "no_convergence"]
    cells = []
    idx = 0
    for c in c_values:
        for beta in beta_values:
            for kappa in kappa_values:
                cells.append(
                    _make_cell(beta, kappa, c, verdicts[idx % 4], -100.0 - 10.0 * idx)
                )
                idx += 1
    aggregate = _make_aggregate(beta_values, kappa_values, c_values, cells)

    results_path = tmp_path / "results.json"
    with open(results_path, "w") as f:
        json.dump(aggregate, f)

    out_png = tmp_path / "phase.png"
    out_md = tmp_path / "phase.md"
    data = plot_module.load_aggregate(results_path)
    plot_module.render_png(data, out_png)
    plot_module.render_markdown(data, out_md)

    assert out_png.exists() and out_png.stat().st_size > 1000
    assert out_md.exists() and out_md.stat().st_size > 0


def test_markdown_rows_sorted_by_c_beta_kappa(plot_module, tmp_path: Path) -> None:
    """Markdown rows must be sorted by (c, β, κ) per the acceptance criteria."""
    beta_values = [0.1, 0.9]
    kappa_values = [0.1, 0.9]
    c_values = [0.5, 2.0]
    # Build cells in scrambled order to verify the renderer sorts them.
    cells = [
        _make_cell(0.9, 0.9, 2.0, "asymmetric_only", -500.0),
        _make_cell(0.1, 0.1, 0.5, "symmetric_only", -100.0),
        _make_cell(0.1, 0.9, 0.5, "mixed", -150.0),
        _make_cell(0.9, 0.1, 0.5, "no_convergence", -200.0),
        _make_cell(0.1, 0.1, 2.0, "symmetric_only", -300.0),
        _make_cell(0.1, 0.9, 2.0, "mixed", -350.0),
        _make_cell(0.9, 0.1, 2.0, "no_convergence", -400.0),
        _make_cell(0.9, 0.9, 0.5, "asymmetric_only", -250.0),
    ]
    aggregate = _make_aggregate(beta_values, kappa_values, c_values, cells)
    results_path = tmp_path / "results.json"
    with open(results_path, "w") as f:
        json.dump(aggregate, f)
    out_md = tmp_path / "phase.md"
    out_png = tmp_path / "phase.png"
    data = plot_module.load_aggregate(results_path)
    plot_module.render_png(data, out_png)
    plot_module.render_markdown(data, out_md)

    # Parse the markdown table data rows (skip header + separator)
    lines = out_md.read_text().splitlines()
    data_rows = [
        ln
        for ln in lines
        if ln.startswith("|") and not ln.startswith("|---") and "verdict" not in ln
    ]
    parsed: list[tuple[float, float, float]] = []
    for row in data_rows:
        cols = [c.strip() for c in row.strip("|").split("|")]
        # cols: c, β, κ, verdict, payoff, conv_rate
        parsed.append((float(cols[0]), float(cols[1]), float(cols[2])))

    assert parsed == sorted(parsed), (
        f"Markdown rows are not sorted by (c, β, κ); got {parsed}"
    )
    # Sanity: header columns include the required fields.
    header_line = next(ln for ln in lines if "verdict" in ln and ln.startswith("|"))
    for required in (
        "c",
        "β",
        "κ",
        "verdict",
        "equilibrium_payoff",
        "convergence_rate",
    ):
        assert required in header_line, (
            f"Markdown header missing required column '{required}': {header_line}"
        )


def test_handles_null_payoff(plot_module, tmp_path: Path) -> None:
    """A cell with null ``best_team_payoff`` must not crash the renderers."""
    cells = [_make_cell(0.5, 0.5, 0.5, "no_convergence", 0.0)]
    cells[0]["best_team_payoff"] = None
    cells[0]["best_symmetric_team_payoff"] = None
    cells[0]["best_asymmetric_team_payoff"] = None
    aggregate = _make_aggregate([0.5], [0.5], [0.5], cells)
    results_path = tmp_path / "results.json"
    with open(results_path, "w") as f:
        json.dump(aggregate, f)
    out_png = tmp_path / "phase.png"
    out_md = tmp_path / "phase.md"
    data = plot_module.load_aggregate(results_path)
    plot_module.render_png(data, out_png)
    plot_module.render_markdown(data, out_md)
    assert out_png.exists() and out_png.stat().st_size > 0
    assert "n/a" in out_md.read_text()


def test_missing_aggregate_raises(plot_module, tmp_path: Path) -> None:
    """A non-existent ``results.json`` path must raise ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        plot_module.load_aggregate(tmp_path / "does_not_exist.json")
