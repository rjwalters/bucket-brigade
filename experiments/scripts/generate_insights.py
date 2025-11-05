#!/usr/bin/env python3
"""
Generate research insights automatically from experimental results.

This script analyzes Nash equilibrium, evolution, and comparison results to
automatically generate research insights for the web interface.

Usage:
    python experiments/scripts/generate_insights.py greedy_neighbor
    python experiments/scripts/generate_insights.py --all
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bucket_brigade.envs.scenarios import list_scenarios


class InsightGenerator:
    """Automatically generate research insights from experimental data."""

    def __init__(self, scenario_name: str):
        self.scenario_name = scenario_name
        self.scenario_dir = Path(f"experiments/scenarios/{scenario_name}")
        self.data = self._load_all_data()

    def _load_all_data(self) -> Dict[str, Any]:
        """Load all experimental results."""
        data = {}

        # Load Nash equilibrium
        nash_file = self.scenario_dir / "nash" / "equilibrium.json"
        if nash_file.exists():
            with open(nash_file) as f:
                data["nash"] = json.load(f)

        # Load evolution results
        evolution_file = self.scenario_dir / "evolved" / "best_agent.json"
        trace_file = self.scenario_dir / "evolved" / "evolution_trace.json"
        if evolution_file.exists() and trace_file.exists():
            with open(evolution_file) as f:
                data["evolved_agent"] = json.load(f)
            with open(trace_file) as f:
                data["evolution_trace"] = json.load(f)

        # Load comparison results
        comparison_file = self.scenario_dir / "comparison" / "comparison.json"
        if comparison_file.exists():
            with open(comparison_file) as f:
                data["comparison"] = json.load(f)

        # Load heuristics
        heuristics_file = self.scenario_dir / "heuristics" / "results.json"
        if heuristics_file.exists():
            with open(heuristics_file) as f:
                data["heuristics"] = json.load(f)

        # Load config
        config_file = self.scenario_dir / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                data["config"] = json.load(f)

        return data

    def _get_nash_strategy_type(self) -> str:
        """Determine the Nash equilibrium strategy type."""
        if "nash" not in self.data:
            return "unknown"

        strategy = self.data["nash"]["equilibrium"]["strategy_pool"][0]
        return strategy["classification"]

    def _get_cooperation_dynamics(self) -> Dict[str, Any]:
        """Analyze cooperation vs free-riding dynamics."""
        dynamics = {}

        if "nash" in self.data:
            nash = self.data["nash"]
            dynamics["nash_cooperation_rate"] = nash["interpretation"][
                "cooperation_rate"
            ]
            dynamics["nash_free_riding_rate"] = nash["interpretation"][
                "free_riding_rate"
            ]
            dynamics["nash_payoff"] = nash["equilibrium"]["expected_payoff"]
            dynamics["nash_type"] = nash["equilibrium"]["type"]

        if "comparison" in self.data:
            comp = self.data["comparison"]
            ranking = comp["ranking"]
            if len(ranking) >= 2:
                best = ranking[0]
                dynamics["best_strategy"] = best["name"]
                dynamics["best_payoff"] = best["mean_payoff"]
                dynamics["payoff_gap"] = best["mean_payoff"] - dynamics.get(
                    "nash_payoff", 0
                )

        return dynamics

    def _analyze_parameter_importance(self) -> List[Dict[str, Any]]:
        """Identify which parameters matter most."""
        important_params = []

        if "nash" not in self.data or "comparison" not in self.data:
            return important_params

        nash_params = self.data["nash"]["equilibrium"]["strategy_pool"][0]["parameters"]

        # Find best performing strategy
        best_strategy_name = self.data["comparison"]["ranking"][0]["name"]
        best_params = self.data["comparison"]["strategies"][best_strategy_name]

        # Parameter names
        param_names = [
            "honesty",
            "work_tendency",
            "neighbor_help",
            "own_priority",
            "risk_aversion",
            "coordination",
            "exploration",
            "fatigue_memory",
            "rest_bias",
            "altruism",
        ]

        # Compare parameters
        for i, param in enumerate(param_names):
            nash_val = nash_params[param]
            best_val = best_params[i] if i < len(best_params) else 0
            diff = abs(best_val - nash_val)

            if diff > 0.3:  # Significant difference
                important_params.append(
                    {
                        "parameter": param,
                        "nash": nash_val,
                        "best": best_val,
                        "difference": diff,
                    }
                )

        return sorted(important_params, key=lambda x: x["difference"], reverse=True)[:3]

    def generate_nash_insights(self) -> List[Dict[str, Any]]:
        """Generate Nash equilibrium-specific insights."""
        insights = []

        if "nash" not in self.data:
            return insights

        nash = self.data["nash"]
        nash_payoff = nash["equilibrium"]["expected_payoff"]
        cooperation_rate = nash["interpretation"]["cooperation_rate"]
        equilibrium_type = nash["equilibrium"]["type"]
        strategy_type = nash["equilibrium"]["strategy_pool"][0]["classification"]

        # Insight 1: Equilibrium characterization
        if cooperation_rate == 0:
            question = (
                "What does Nash equilibrium predict about cooperation in this scenario?"
            )
            finding = f"Nash equilibrium predicts complete free-riding (0% cooperation) with expected payoff of {nash_payoff:.1f}."
            implication = "Individual rationality leads to collective failure. Without external coordination mechanisms, rational agents will defect even when cooperation benefits everyone."
        elif cooperation_rate >= 0.8:
            question = "Why does Nash equilibrium favor cooperation in this scenario?"
            finding = f"Nash equilibrium selects highly cooperative strategies ({cooperation_rate * 100:.0f}% cooperation rate) achieving {nash_payoff:.1f} payoff."
            implication = "The incentive structure naturally aligns individual and collective interests, making cooperation individually rational without enforcement."
        else:
            question = "What mixed strategy does Nash equilibrium prescribe?"
            finding = f"Nash equilibrium involves partial cooperation ({cooperation_rate * 100:.0f}%) with {nash_payoff:.1f} expected payoff."
            implication = "The scenario creates tension between cooperation and free-riding, resulting in a mixed equilibrium that balances both strategies."

        evidence = [
            f"Equilibrium type: {equilibrium_type}",
            f"Strategy classification: {strategy_type}",
            f"Expected payoff: {nash_payoff:.2f}",
            f"Cooperation rate: {cooperation_rate * 100:.0f}%",
            f"Convergence: {nash['convergence']['iterations']} iterations in {nash['convergence']['elapsed_time']:.1f}s",
        ]

        insights.append(
            {
                "question": question,
                "finding": finding,
                "evidence": evidence,
                "implication": implication,
            }
        )

        # Insight 2: Strategy stability
        strategy = nash["equilibrium"]["strategy_pool"][0]
        key_params = {
            "coordination": strategy["parameters"]["coordination"],
            "honesty": strategy["parameters"]["honesty"],
            "work_tendency": strategy["parameters"]["work_tendency"],
            "altruism": strategy["parameters"]["altruism"],
        }

        high_params = [k for k, v in key_params.items() if v > 0.7]

        if high_params:
            question = "What strategic traits does Nash equilibrium emphasize?"
            finding = f"The equilibrium strongly favors {', '.join(high_params)} (all > 0.7) as critical for stability."
            evidence = [f"{k}: {v:.2f}" for k, v in key_params.items() if v > 0.7]
            evidence.append(
                f"Archetype: {strategy['closest_archetype']} (distance: {strategy['archetype_distance']:.2f})"
            )
            implication = "These traits are Nash-stable because they form mutually best responses - no agent can improve by unilaterally deviating."

            insights.append(
                {
                    "question": question,
                    "finding": finding,
                    "evidence": evidence,
                    "implication": implication,
                }
            )

        return insights

    def generate_evolution_insights(self) -> List[Dict[str, Any]]:
        """Generate evolution-specific insights."""
        insights = []

        if "evolved_agent" not in self.data or "evolution_trace" not in self.data:
            return insights

        evolved = self.data["evolved_agent"]
        trace = self.data["evolution_trace"]

        final_fitness = evolved["fitness"]
        generations = trace["generations"]
        initial_fitness = generations[0]["best_fitness"]
        improvement = final_fitness - initial_fitness

        # Insight 1: Optimization performance
        question = "How effective was evolutionary optimization?"
        finding = f"Evolution improved from {initial_fitness:.1f} to {final_fitness:.1f} payoff over {len(generations)} generations (Δ {improvement:.1f})."

        evidence = [
            f"Initial best fitness: {initial_fitness:.2f}",
            f"Final best fitness: {final_fitness:.2f}",
            f"Total improvement: {improvement:.2f}",
            f"Generations: {len(generations)}",
            f"Converged: {trace['convergence']['converged']}",
        ]

        if improvement > 30:
            implication = "Evolution discovered highly effective strategies through natural selection, showing substantial improvement over random initialization."
        elif improvement > 10:
            implication = "Evolution found meaningful improvements, though the fitness landscape may have local optima limiting further progress."
        else:
            implication = "Limited improvement suggests either strong initial random strategies or a complex fitness landscape with many local optima."

        insights.append(
            {
                "question": question,
                "finding": finding,
                "evidence": evidence,
                "implication": implication,
            }
        )

        # Insight 2: Convergence dynamics
        mid_gen = len(generations) // 2
        mid_diversity = (
            generations[mid_gen]["diversity"] if mid_gen < len(generations) else 0
        )
        final_diversity = generations[-1]["diversity"]

        question = "What does population diversity reveal about the fitness landscape?"
        if final_diversity < 0.1:
            finding = f"Population converged to low diversity ({final_diversity:.3f}), indicating a strong fitness peak."
            implication = "The population found a dominant strategy that outcompetes alternatives, suggesting a clear local or global optimum."
        elif final_diversity > 0.3:
            finding = f"Population maintained high diversity ({final_diversity:.3f}), suggesting multiple viable strategies."
            implication = "The fitness landscape has multiple peaks or plateaus where different strategies achieve similar performance."
        else:
            finding = f"Population reached moderate diversity ({final_diversity:.3f}), balancing exploration and exploitation."
            implication = "Evolution balanced convergence toward good strategies while maintaining variation for continued adaptation."

        evidence = [
            f"Final diversity: {final_diversity:.3f}",
            f"Mid-run diversity: {mid_diversity:.3f}",
            f"Best generation: {evolved['generation']}",
        ]

        insights.append(
            {
                "question": question,
                "finding": finding,
                "evidence": evidence,
                "implication": implication,
            }
        )

        return insights

    def generate_comparative_insights(self) -> List[Dict[str, Any]]:
        """Generate insights comparing methods."""
        insights = []

        if "nash" not in self.data or "comparison" not in self.data:
            return insights

        dynamics = self._get_cooperation_dynamics()

        nash_payoff = dynamics.get("nash_payoff", 0)
        best_payoff = dynamics.get("best_payoff", 0)
        payoff_gap = dynamics.get("payoff_gap", 0)

        # Insight 1: Efficiency gap between Nash and optimal
        if payoff_gap > 20:
            question = "How much performance is lost at Nash equilibrium?"
            finding = f"Nash equilibrium ({nash_payoff:.1f}) falls {payoff_gap:.1f} points short of optimal play ({best_payoff:.1f}) - a {(payoff_gap / nash_payoff * 100):.0f}% efficiency loss."
            implication = "Significant coordination failure. Individual rationality produces collectively suboptimal outcomes, indicating need for external coordination mechanisms."
        elif payoff_gap < 5:
            question = "How close is Nash equilibrium to optimal performance?"
            finding = f"Nash equilibrium ({nash_payoff:.1f}) achieves near-optimal performance, within {payoff_gap:.1f} points of best strategy ({best_payoff:.1f})."
            implication = "High efficiency - the incentive structure aligns individual and collective interests, making coordination relatively easy."
        else:
            question = "What is the price of anarchy in this scenario?"
            finding = f"Nash equilibrium ({nash_payoff:.1f}) shows moderate efficiency loss of {payoff_gap:.1f} points vs optimal ({best_payoff:.1f})."
            implication = "Partial coordination failure. Some gains possible through cooperation, but individual incentives aren't catastrophically misaligned."

        evidence = [
            f"Nash payoff: {nash_payoff:.2f}",
            f"Optimal payoff: {best_payoff:.2f}",
            f"Gap: {payoff_gap:.2f}",
            f"Efficiency: {(nash_payoff / best_payoff * 100):.1f}%",
        ]

        insights.append(
            {
                "question": question,
                "finding": finding,
                "evidence": evidence,
                "implication": implication,
            }
        )

        # Insight 2: Evolution vs Nash
        if (
            "nash" in self.data
            and "evolved_agent" in self.data
            and "comparison" in self.data
        ):
            nash_params = self.data["nash"]["equilibrium"]["strategy_pool"][0][
                "parameters"
            ]
            evolved_params = self.data["evolved_agent"]["parameters"]

            # Find evolved strategy in comparison
            evolved_payoff = None
            for entry in self.data["comparison"]["ranking"]:
                if entry["name"] == "evolved":
                    evolved_payoff = entry["mean_payoff"]
                    break

            if evolved_payoff:
                nash_payoff = dynamics.get("nash_payoff", 0)
                performance_diff = evolved_payoff - nash_payoff

                # Find key parameter differences
                key_diffs = []
                for param in ["honesty", "coordination", "work_tendency", "altruism"]:
                    nash_val = nash_params[param]
                    evolved_val = evolved_params[param]
                    if abs(evolved_val - nash_val) > 0.3:
                        key_diffs.append(
                            f"{param}: {nash_val:.2f} (Nash) vs {evolved_val:.2f} (Evolved)"
                        )

                if performance_diff > 5:
                    question = "Can evolution discover strategies superior to Nash equilibrium?"
                    finding = f"Yes - evolved strategies achieve {evolved_payoff:.1f} payoff, outperforming Nash equilibrium ({nash_payoff:.1f}) by {performance_diff:.1f} points."
                elif performance_diff < -5:
                    question = (
                        "Why does evolution fail to reach Nash equilibrium performance?"
                    )
                    finding = f"Evolution achieves only {evolved_payoff:.1f} payoff, falling {abs(performance_diff):.1f} points short of Nash equilibrium ({nash_payoff:.1f})."
                else:
                    question = (
                        "How does evolution compare to game-theoretic Nash equilibrium?"
                    )
                    finding = f"Evolution and Nash equilibrium converge to similar performance levels ({evolved_payoff:.1f} vs {nash_payoff:.1f})."

                evidence = [
                    f"Evolved payoff: {evolved_payoff:.2f}",
                    f"Nash payoff: {nash_payoff:.2f}",
                ] + key_diffs[:3]

                if performance_diff > 5:
                    implication = "Evolution can escape suboptimal equilibria by discovering innovative strategies that Nash analysis misses. Natural selection may find solutions that pure rationality cannot."
                elif performance_diff < -5:
                    implication = "Some strategy spaces are too complex for evolution to navigate effectively. The fitness landscape may have local optima that trap evolutionary search."
                else:
                    implication = "Evolution and game theory independently discover similar optimal strategies, suggesting these solutions are robust attractors in strategy space."

                insights.append(
                    {
                        "question": question,
                        "finding": finding,
                        "evidence": evidence,
                        "implication": implication,
                    }
                )

        # Insight 3: Critical Parameters
        important_params = self._analyze_parameter_importance()
        if important_params:
            param = important_params[0]  # Most important parameter

            question = (
                "Which agent parameters matter most for success in this scenario?"
            )
            finding = f"The parameter '{param['parameter']}' shows the largest difference between Nash and optimal strategies, suggesting it's critical for performance."

            evidence = [
                f"Nash strategy {param['parameter']}: {param['nash']:.2f}",
                f"Best strategy {param['parameter']}: {param['best']:.2f}",
                f"Difference: {param['difference']:.2f}",
            ]

            # Add other important parameters
            for p in important_params[1:3]:
                evidence.append(
                    f"Also important: {p['parameter']} (Δ {p['difference']:.2f})"
                )

            implication = f"Success in this scenario critically depends on optimizing {param['parameter']}. Strategies that neglect this parameter face significant performance penalties."

            insights.append(
                {
                    "question": question,
                    "finding": finding,
                    "evidence": evidence,
                    "implication": implication,
                }
            )

        return insights

    def generate_all_insights(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate all method-specific insights."""
        method_insights = {}

        nash_insights = self.generate_nash_insights()
        if nash_insights:
            method_insights["nash"] = nash_insights

        evolution_insights = self.generate_evolution_insights()
        if evolution_insights:
            method_insights["evolution"] = evolution_insights

        comparative_insights = self.generate_comparative_insights()
        if comparative_insights:
            method_insights["comparative"] = comparative_insights

        return method_insights

    def update_config_with_insights(
        self, method_insights: Dict[str, List[Dict[str, Any]]]
    ):
        """Update the scenario config file with generated insights."""
        # Try both experiment and web config locations
        config_paths = [
            self.scenario_dir / "config.json",
            Path(f"web/public/research/scenarios/{self.scenario_name}/config.json"),
        ]

        updated_count = 0
        total_insights = sum(len(insights) for insights in method_insights.values())

        for config_file in config_paths:
            if not config_file.exists():
                continue

            with open(config_file, "r") as f:
                config = json.load(f)

            # Write new structured format
            config["method_insights"] = method_insights

            # Keep legacy format for backwards compatibility
            # Flatten all insights into single list
            all_insights = []
            for method, insights in method_insights.items():
                all_insights.extend(insights)
            if all_insights:
                config["research_insights"] = all_insights

            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)

            updated_count += 1

        if updated_count > 0:
            methods_str = ", ".join(method_insights.keys())
            print(
                f"✅ Updated {updated_count} config file(s) with {total_insights} insights ({methods_str})"
            )
            return True
        else:
            print(f"❌ No config files found for {self.scenario_name}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate research insights from experimental data"
    )
    parser.add_argument(
        "scenario", nargs="?", help="Scenario name (or use --all)", default=None
    )
    parser.add_argument(
        "--all", action="store_true", help="Generate insights for all scenarios"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print insights without updating config files",
    )

    args = parser.parse_args()

    if args.all:
        scenarios = list_scenarios()
    elif args.scenario:
        scenarios = [args.scenario]
    else:
        parser.print_help()
        return 1

    for scenario in scenarios:
        print(f"\n{'=' * 80}")
        print(f"Generating insights for: {scenario}")
        print(f"{'=' * 80}\n")

        generator = InsightGenerator(scenario)
        method_insights = generator.generate_all_insights()

        if not method_insights:
            print(f"⚠️  No insights generated for {scenario}\n")
            continue

        # Print insights by method
        total_count = sum(len(insights) for insights in method_insights.values())
        print(
            f"\n📊 Generated {total_count} insights across {len(method_insights)} method(s):\n"
        )

        for method, insights in method_insights.items():
            print(f"  {method.upper()} ({len(insights)} insights):")
            for i, insight in enumerate(insights, 1):
                print(f"    {i}. {insight['question'][:80]}...")
            print()

        if not args.dry_run:
            generator.update_config_with_insights(method_insights)
        else:
            print("🔍 Dry run - config not updated")

    print(f"\n✅ Completed insight generation for {len(scenarios)} scenario(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
