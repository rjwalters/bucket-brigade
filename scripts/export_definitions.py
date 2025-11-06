#!/usr/bin/env python3
"""
Export archetypes and scenarios from Python to JSON.

This script extracts the current Python definitions and saves them as JSON.
These JSON files will become the single source of truth for code generation.
"""

import json
import sys
from pathlib import Path

# Add bucket_brigade to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bucket_brigade.agents.archetypes import ARCHETYPES
from bucket_brigade.envs.scenarios import SCENARIO_REGISTRY


def export_archetypes(output_path: Path):
    """Export archetype definitions to JSON."""
    archetypes = {}

    for name, params in ARCHETYPES.items():
        archetypes[name] = {
            "params": params.tolist(),
            "description": f"{name.replace('_', ' ').title()} archetype"
        }

    with open(output_path, 'w') as f:
        json.dump({
            "version": "1.0",
            "archetypes": archetypes
        }, f, indent=2)

    print(f"✓ Exported {len(archetypes)} archetypes to {output_path}")


def export_scenarios(output_path: Path):
    """Export scenario definitions to JSON."""
    scenarios = {}

    for name, factory in SCENARIO_REGISTRY.items():
        # Create scenario with default num_agents to get parameter structure
        scenario = factory(num_agents=4)

        scenarios[name] = {
            "beta": float(scenario.beta),
            "kappa": float(scenario.kappa),
            "A": float(scenario.A),
            "L": float(scenario.L),
            "c": float(scenario.c),
            "rho_ignite": float(scenario.rho_ignite),
            "N_min": int(scenario.N_min),
            "p_spark": float(scenario.p_spark),
            "N_spark": int(scenario.N_spark),
            "description": _get_scenario_description(name)
        }

    with open(output_path, 'w') as f:
        json.dump({
            "version": "1.0",
            "scenarios": scenarios
        }, f, indent=2)

    print(f"✓ Exported {len(scenarios)} scenarios to {output_path}")


def _get_scenario_description(name: str) -> str:
    """Get human-readable description for scenario."""
    descriptions = {
        "default": "Standard balanced scenario for general testing",
        "easy": "Low difficulty with favorable conditions - low fire spread, high extinguish rate",
        "hard": "High difficulty with challenging conditions - high fire spread, low extinguish rate",
        "trivial_cooperation": "Easy fires reward universal cooperation",
        "early_containment": "Fires start aggressive but can be stopped early",
        "greedy_neighbor": "Social dilemma between self-interest and cooperation",
        "sparse_heroics": "Few workers can make the difference",
        "rest_trap": "Fires usually extinguish themselves, but not always",
        "chain_reaction": "High spread demands distributed firefighting teams",
        "deceptive_calm": "Honest signaling rewarded during occasional flare-ups",
        "overcrowding": "Too many workers reduce efficiency",
        "mixed_motivation": "House ownership creates conflicting incentives",
    }
    return descriptions.get(name, f"{name.replace('_', ' ').title()} scenario")


def main():
    """Export all definitions."""
    root = Path(__file__).parent.parent
    definitions_dir = root / "definitions"
    definitions_dir.mkdir(exist_ok=True)

    export_archetypes(definitions_dir / "archetypes.json")
    export_scenarios(definitions_dir / "scenarios.json")

    print("\n✓ Export complete! JSON definitions are in definitions/")
    print("  These files are now the single source of truth.")


if __name__ == "__main__":
    main()
