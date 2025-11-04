#!/usr/bin/env python3
"""
Analyze and compare Nash equilibrium results across all scenarios.

Generates a comprehensive summary report comparing:
- Equilibrium types (pure vs mixed)
- Cooperation vs free-riding rates
- Strategy distributions
- Convergence properties
- Cross-scenario patterns

Usage:
    python experiments/scripts/analyze_nash_results.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd


def load_equilibrium_results(scenarios_dir: Path) -> Dict[str, Any]:
    """Load Nash equilibrium results for all scenarios."""
    results = {}

    for scenario_dir in sorted(scenarios_dir.iterdir()):
        if not scenario_dir.is_dir():
            continue

        nash_file = scenario_dir / "nash" / "equilibrium.json"
        if nash_file.exists():
            with open(nash_file) as f:
                results[scenario_dir.name] = json.load(f)
        else:
            print(f"⚠️  No Nash results found for {scenario_dir.name}", file=sys.stderr)

    return results


def analyze_equilibrium_type(results: Dict[str, Any]) -> pd.DataFrame:
    """Analyze equilibrium types across scenarios."""
    data = []

    for scenario, result in results.items():
        eq = result["equilibrium"]
        data.append(
            {
                "scenario": scenario,
                "type": eq["type"],
                "support_size": eq["support_size"],
                "expected_payoff": eq["expected_payoff"],
                "iterations": result["convergence"]["iterations"],
                "converged": result["convergence"]["converged"],
                "elapsed_time": result["convergence"]["elapsed_time"],
            }
        )

    return pd.DataFrame(data)


def analyze_cooperation_rates(results: Dict[str, Any]) -> pd.DataFrame:
    """Analyze cooperation vs free-riding rates."""
    data = []

    for scenario, result in results.items():
        interp = result["interpretation"]
        data.append(
            {
                "scenario": scenario,
                "cooperation_rate": interp["cooperation_rate"],
                "free_riding_rate": interp["free_riding_rate"],
                "equilibrium_type": interp["equilibrium_type"],
            }
        )

    return pd.DataFrame(data)


def analyze_strategy_distribution(results: Dict[str, Any]) -> Dict[str, List[Dict]]:
    """Analyze strategy distributions for each scenario."""
    distributions = {}

    for scenario, result in results.items():
        strategies = result["equilibrium"]["strategy_pool"]
        distributions[scenario] = [
            {
                "probability": s["probability"],
                "classification": s["classification"],
                "closest_archetype": s["closest_archetype"],
                "archetype_distance": s["archetype_distance"],
            }
            for s in strategies
        ]

    return distributions


def generate_markdown_report(
    results: Dict[str, Any],
    eq_types: pd.DataFrame,
    coop_rates: pd.DataFrame,
    distributions: Dict[str, List[Dict]],
) -> str:
    """Generate comprehensive markdown report."""

    lines = [
        "# Nash Equilibrium Analysis - Cross-Scenario Summary",
        "",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Scenarios Analyzed:** {len(results)}",
        "",
        "## Overview",
        "",
        "This report presents Nash equilibrium analysis for all Bucket Brigade scenarios,",
        "computed using the Double Oracle algorithm with Rust-accelerated payoff evaluation.",
        "",
        "## Equilibrium Types",
        "",
        eq_types.to_markdown(index=False),
        "",
        "### Key Observations",
        "",
    ]

    # Count pure vs mixed
    pure_count = (eq_types["type"] == "pure").sum()
    mixed_count = (eq_types["type"] == "mixed").sum()

    lines.extend(
        [
            f"- **Pure equilibria:** {pure_count}/{len(results)} scenarios",
            f"- **Mixed equilibria:** {mixed_count}/{len(results)} scenarios",
            f"- **Average support size:** {eq_types['support_size'].mean():.2f}",
            f"- **Average convergence time:** {eq_types['elapsed_time'].mean():.1f}s",
            f"- **Convergence rate:** {(eq_types['converged'].sum() / len(results) * 100):.1f}%",
            "",
        ]
    )

    # Scenarios by type
    if pure_count > 0:
        pure_scenarios = eq_types[eq_types["type"] == "pure"]["scenario"].tolist()
        lines.append(f"**Pure equilibrium scenarios:** {', '.join(pure_scenarios)}")
        lines.append("")

    if mixed_count > 0:
        mixed_scenarios = eq_types[eq_types["type"] == "mixed"]["scenario"].tolist()
        lines.append(f"**Mixed equilibrium scenarios:** {', '.join(mixed_scenarios)}")
        lines.append("")

    # Cooperation analysis
    lines.extend(
        [
            "## Cooperation Analysis",
            "",
            coop_rates.to_markdown(index=False),
            "",
            "### Cooperation Patterns",
            "",
            f"- **Highest cooperation:** {coop_rates.loc[coop_rates['cooperation_rate'].idxmax(), 'scenario']} "
            f"({coop_rates['cooperation_rate'].max():.1%})",
            f"- **Lowest cooperation:** {coop_rates.loc[coop_rates['cooperation_rate'].idxmin(), 'scenario']} "
            f"({coop_rates['cooperation_rate'].min():.1%})",
            f"- **Average cooperation:** {coop_rates['cooperation_rate'].mean():.1%}",
            "",
        ]
    )

    # Strategy distributions
    lines.extend(
        [
            "## Strategy Distributions",
            "",
        ]
    )

    for scenario in sorted(results.keys()):
        lines.append(f"### {scenario}")
        lines.append("")

        dist = distributions[scenario]
        eq_type = results[scenario]["equilibrium"]["type"]
        payoff = results[scenario]["equilibrium"]["expected_payoff"]

        lines.append(f"**Type:** {eq_type.capitalize()} equilibrium")
        lines.append(f"**Expected Payoff:** {payoff:.2f}")
        lines.append("")

        if len(dist) == 1:
            s = dist[0]
            lines.append(
                f"Pure strategy: **{s['classification']}** "
                f"(closest to {s['closest_archetype']}, distance={s['archetype_distance']:.3f})"
            )
        else:
            lines.append("Mixed strategy distribution:")
            lines.append("")
            for i, s in enumerate(dist, 1):
                lines.append(
                    f"{i}. **{s['classification']}** "
                    f"({s['probability']:.1%}) - "
                    f"closest to {s['closest_archetype']} "
                    f"(distance={s['archetype_distance']:.3f})"
                )

        lines.append("")

    # Cross-scenario insights
    lines.extend(
        [
            "## Cross-Scenario Insights",
            "",
            "### Parameter-Equilibrium Relationships",
            "",
        ]
    )

    # Collect scenario parameters
    param_data = []
    for scenario, result in results.items():
        params = result["parameters"]
        eq = result["equilibrium"]
        interp = result["interpretation"]
        param_data.append(
            {
                "scenario": scenario,
                "beta": params["beta"],
                "kappa": params["kappa"],
                "c": params["c"],
                "type": eq["type"],
                "cooperation": interp["cooperation_rate"],
            }
        )

    param_df = pd.DataFrame(param_data)

    # Analyze correlations
    if len(param_df) > 1:
        # High work cost scenarios
        high_cost = param_df[param_df["c"] > param_df["c"].median()]
        if len(high_cost) > 0:
            lines.append(
                f"**High work cost scenarios (c > {param_df['c'].median():.2f}):**"
            )
            lines.append(
                f"- Average cooperation: {high_cost['cooperation'].mean():.1%}"
            )
            lines.append(
                f"- Pure equilibria: {(high_cost['type'] == 'pure').sum()}/{len(high_cost)}"
            )
            lines.append("")

        # High spread scenarios
        high_spread = param_df[param_df["beta"] > param_df["beta"].median()]
        if len(high_spread) > 0:
            lines.append(
                f"**High fire spread scenarios (β > {param_df['beta'].median():.2f}):**"
            )
            lines.append(
                f"- Average cooperation: {high_spread['cooperation'].mean():.1%}"
            )
            lines.append(
                f"- Pure equilibria: {(high_spread['type'] == 'pure').sum()}/{len(high_spread)}"
            )
            lines.append("")

    # Design recommendations
    lines.extend(
        [
            "## Design Recommendations",
            "",
            "Based on Nash equilibrium analysis:",
            "",
        ]
    )

    # Scenario-specific recommendations
    if "greedy_neighbor" in results:
        gr_coop = results["greedy_neighbor"]["interpretation"]["cooperation_rate"]
        lines.append(f"1. **Greedy Neighbor Scenario** (cooperation: {gr_coop:.1%})")
        lines.append("   - High work cost creates free-riding incentive")
        if gr_coop < 0.5:
            lines.append(
                "   - Consider mechanisms to incentivize cooperation (rewards, punishment)"
            )
        lines.append("")

    avg_coop = coop_rates["cooperation_rate"].mean()
    lines.extend(
        [
            f"2. **Overall Cooperation** (average: {avg_coop:.1%})",
            f"   - {'Moderate' if 0.4 < avg_coop < 0.6 else 'High' if avg_coop >= 0.6 else 'Low'} "
            f"cooperation across scenarios",
        ]
    )

    if avg_coop < 0.5:
        lines.append("   - Game parameters may favor free-riding; consider rebalancing")

    lines.extend(
        [
            "",
            "3. **Equilibrium Diversity**",
        ]
    )

    if mixed_count > pure_count:
        lines.append(
            "   - Predominance of mixed equilibria indicates strategic tension"
        )
        lines.append("   - No dominant strategy across most scenarios")
        lines.append("   - Agents must randomize to remain unpredictable")
    else:
        lines.append(
            "   - Predominance of pure equilibria indicates clear optimal strategies"
        )
        lines.append("   - Game parameters may need tuning for strategic depth")

    lines.extend(
        [
            "",
            "## Future Work",
            "",
            "1. **Compare with evolved agents** (Issue #84)",
            "   - Do evolved strategies converge to Nash equilibria?",
            "   - Measure strategy divergence between theory and practice",
            "",
            "2. **Robustness analysis**",
            "   - Test equilibrium stability to parameter perturbations",
            "   - Identify critical parameter thresholds",
            "",
            "3. **Mechanism design**",
            "   - Design incentives to improve equilibrium outcomes",
            "   - Reduce price of anarchy where applicable",
            "",
            "---",
            "",
            "*Analysis generated using Double Oracle algorithm with 2000 simulations per evaluation.*",
        ]
    )

    return "\n".join(lines)


def main():
    """Main entry point."""
    # Find scenarios directory
    repo_root = Path(__file__).parent.parent.parent
    scenarios_dir = repo_root / "experiments" / "scenarios"

    if not scenarios_dir.exists():
        print(f"Error: Scenarios directory not found: {scenarios_dir}", file=sys.stderr)
        sys.exit(1)

    # Load results
    print("Loading Nash equilibrium results...")
    results = load_equilibrium_results(scenarios_dir)

    if not results:
        print("Error: No Nash equilibrium results found!", file=sys.stderr)
        sys.exit(1)

    print(f"Found results for {len(results)} scenarios")

    # Analyze
    print("Analyzing equilibrium types...")
    eq_types = analyze_equilibrium_type(results)

    print("Analyzing cooperation rates...")
    coop_rates = analyze_cooperation_rates(results)

    print("Analyzing strategy distributions...")
    distributions = analyze_strategy_distribution(results)

    # Generate report
    print("Generating markdown report...")
    report = generate_markdown_report(results, eq_types, coop_rates, distributions)

    # Save report
    output_file = repo_root / "experiments" / "nash_analysis_summary.md"
    with open(output_file, "w") as f:
        f.write(report)

    print(f"\n✅ Report saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Scenarios analyzed: {len(results)}")
    print(f"Pure equilibria: {(eq_types['type'] == 'pure').sum()}")
    print(f"Mixed equilibria: {(eq_types['type'] == 'mixed').sum()}")
    print(f"Average cooperation: {coop_rates['cooperation_rate'].mean():.1%}")
    print(f"Average convergence time: {eq_types['elapsed_time'].mean():.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
