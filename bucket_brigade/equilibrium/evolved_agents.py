"""
Utilities for loading evolved agents and integrating with Nash equilibrium computation.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Optional


def load_evolved_agent(scenario: str, version: str = "v4") -> Optional[np.ndarray]:
    """
    Load an evolved agent genome for a scenario.

    Args:
        scenario: Scenario name (e.g., "chain_reaction")
        version: Evolution version ("v3", "v4", "v5")

    Returns:
        10-parameter genome as numpy array, or None if not found
    """
    agent_path = Path(
        f"experiments/scenarios/{scenario}/evolved_{version}/best_agent.json"
    )

    if not agent_path.exists():
        return None

    with open(agent_path, "r") as f:
        data = json.load(f)

    genome = np.array(data["genome"], dtype=np.float64)

    if len(genome) != 10:
        raise ValueError(
            f"Expected 10-parameter genome, got {len(genome)} for {scenario} {version}"
        )

    return genome


def load_all_evolved_agents(
    scenario: str, versions: List[str] = ["v3", "v4", "v5"]
) -> List[np.ndarray]:
    """
    Load all available evolved agents for a scenario across multiple versions.

    Args:
        scenario: Scenario name
        versions: List of version strings to try (default: ["v3", "v4", "v5"])

    Returns:
        List of genome arrays (may be empty if none found)
    """
    agents = []

    for version in versions:
        genome = load_evolved_agent(scenario, version)
        if genome is not None:
            agents.append(genome)

    return agents


def load_evolved_agent_metadata(scenario: str, version: str = "v4") -> Optional[dict]:
    """
    Load evolved agent metadata (fitness, generation, etc.).

    Args:
        scenario: Scenario name
        version: Evolution version

    Returns:
        Metadata dictionary or None if not found
    """
    agent_path = Path(
        f"experiments/scenarios/{scenario}/evolved_{version}/best_agent.json"
    )

    if not agent_path.exists():
        return None

    with open(agent_path, "r") as f:
        data = json.load(f)

    return {
        "scenario": data.get("scenario"),
        "fitness": data.get("fitness"),
        "generation": data.get("generation"),
        "version": version,
        "parameters": data.get("parameters", {}),
    }


def get_evolved_agent_description(scenario: str, version: str = "v4") -> str:
    """
    Get human-readable description of evolved agent.

    Args:
        scenario: Scenario name
        version: Evolution version

    Returns:
        Description string
    """
    metadata = load_evolved_agent_metadata(scenario, version)

    if metadata is None:
        return f"Evolved {version} (not found)"

    fitness = metadata.get("fitness", "unknown")
    generation = metadata.get("generation", "unknown")

    return f"Evolved {version} (fitness={fitness:.2f}, gen={generation})"


def compare_genomes(genome1: np.ndarray, genome2: np.ndarray) -> float:
    """
    Compare two genomes using L2 distance.

    Args:
        genome1: First genome (10 parameters)
        genome2: Second genome (10 parameters)

    Returns:
        Euclidean distance between genomes
    """
    return float(np.linalg.norm(genome1 - genome2))


def is_genome_similar(
    genome1: np.ndarray, genome2: np.ndarray, threshold: float = 0.1
) -> bool:
    """
    Check if two genomes are similar within a threshold.

    Args:
        genome1: First genome
        genome2: Second genome
        threshold: Distance threshold for similarity

    Returns:
        True if genomes are similar
    """
    return compare_genomes(genome1, genome2) < threshold
