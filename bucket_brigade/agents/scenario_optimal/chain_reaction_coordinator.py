"""
Optimal agent for Chain Reaction scenario.

In this scenario, high spread requires distributed firefighting teams.
The optimal strategy splits agents into groups covering separate fire clusters.
"""

import numpy as np
from typing import Dict, List
from ..agent_base import AgentBase


class ChainReactionCoordinator(AgentBase):
    """
    Coordinates distributed firefighting by working on isolated fire clusters.

    Avoids overcrowding single fires and ensures all clusters are covered.
    """

    def __init__(self, agent_id: int, name: str = "ChainCoordinator"):
        super().__init__(agent_id, name)
        self.own_house = agent_id % 10

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Find fire clusters and work on the most underserved one.
        """
        houses = obs['houses']
        burning_houses = set(np.where(houses == 1)[0])

        if not burning_houses:
            return np.array([self.own_house, 0])  # Rest if no fires

        # Find fire clusters (connected components)
        clusters = self._find_fire_clusters(houses, burning_houses)

        if not clusters:
            return np.array([self.own_house, 0])

        # Choose cluster to work on (prioritize larger or more spread-risk clusters)
        target_cluster = self._select_best_cluster(clusters, houses)
        target_house = target_cluster[0]  # Work on first house in cluster

        return np.array([target_house, 1])  # Work

    def _find_fire_clusters(self, houses: np.ndarray, burning_houses: set) -> List[List[int]]:
        """Find connected components of burning houses."""
        clusters = []
        visited = set()

        for house in burning_houses:
            if house in visited:
                continue

            # Start new cluster
            cluster = []
            stack = [house]

            while stack:
                current = stack.pop()
                if current in visited:
                    continue

                visited.add(current)
                cluster.append(current)

                # Check neighbors
                neighbors = [(current - 1) % 10, (current + 1) % 10]
                for neighbor in neighbors:
                    if houses[neighbor] == 1 and neighbor not in visited:
                        stack.append(neighbor)

            if cluster:
                clusters.append(sorted(cluster))

        return clusters

    def _select_best_cluster(self, clusters: List[List[int]], houses: np.ndarray) -> List[int]:
        """Select the best cluster to work on."""
        if len(clusters) == 1:
            return clusters[0]

        # Score clusters by size and spread risk
        best_cluster = clusters[0]
        best_score = 0

        for cluster in clusters:
            # Score = cluster size + spread risk
            spread_risk = 0
            for house in cluster:
                neighbors = [(house - 1) % 10, (house + 1) % 10]
                for neighbor in neighbors:
                    if houses[neighbor] == 0:  # Safe neighbor could spread
                        spread_risk += 1

            score = len(cluster) + spread_risk

            if score > best_score:
                best_score = score
                best_cluster = cluster

        return best_cluster


# For agent submission system
def create_agent(agent_id: int, **kwargs):
    """Factory function for creating this agent type."""
    return ChainReactionCoordinator(agent_id, **kwargs)
