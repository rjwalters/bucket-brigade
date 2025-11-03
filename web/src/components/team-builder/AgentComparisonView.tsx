/**
 * Agent Comparison View Component
 *
 * Displays 2-3 agents side-by-side with radar charts and similarity analysis
 */

import { AgentRadarChart } from './AgentRadarChart';
import { calculateSimilarity, type RadarProfile } from '../../utils/agentRadarChart';
import type { AgentArchetype } from '../../types/teamBuilder';

interface AgentComparisonViewProps {
  agents: Array<{ archetype: AgentArchetype; radarProfile: RadarProfile }>;
  maxAgents?: number;
}

export function AgentComparisonView({ agents, maxAgents = 3 }: AgentComparisonViewProps) {
  const displayAgents = agents.slice(0, maxAgents);

  // Calculate pairwise similarities
  const similarities: Record<string, number> = {};
  for (let i = 0; i < displayAgents.length; i++) {
    for (let j = i + 1; j < displayAgents.length; j++) {
      const key = `${i}-${j}`;
      similarities[key] = calculateSimilarity(
        displayAgents[i].radarProfile,
        displayAgents[j].radarProfile
      );
    }
  }

  if (displayAgents.length === 0) {
    return (
      <div className="p-8 text-center text-gray-500 dark:text-gray-400">
        <p>No agents to compare. Select at least 2 agents to see comparison.</p>
      </div>
    );
  }

  if (displayAgents.length === 1) {
    return (
      <div className="p-8 text-center text-gray-500 dark:text-gray-400">
        <p>Select at least one more agent to compare.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Side-by-side radar charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {displayAgents.map((agent) => (
          <div
            key={agent.archetype.id}
            className="flex flex-col items-center p-4 bg-gray-50 dark:bg-gray-900 rounded-lg border-2 transition-all hover:shadow-lg"
            style={{ borderColor: agent.archetype.color }}
          >
            <div className="text-4xl mb-2">{agent.archetype.icon}</div>
            <h3 className="text-lg font-bold text-gray-900 dark:text-gray-100 mb-1">
              {agent.archetype.name}
            </h3>
            <p className="text-xs text-gray-600 dark:text-gray-400 italic mb-4 text-center px-2">
              "{agent.archetype.tagline}"
            </p>
            <AgentRadarChart
              archetype={agent.archetype}
              radarProfile={agent.radarProfile}
              size="small"
              showLabels={false}
            />
          </div>
        ))}
      </div>

      {/* Similarity matrix */}
      {displayAgents.length >= 2 && (
        <div className="p-6 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <h4 className="text-base font-semibold mb-4 text-gray-900 dark:text-gray-100">
            Agent Similarities
          </h4>
          <div className="space-y-3">
            {displayAgents.map((agent1, i) =>
              displayAgents.slice(i + 1).map((agent2, j) => {
                const key = `${i}-${i + j + 1}`;
                const similarity = similarities[key];

                return (
                  <div key={key} className="flex items-center justify-between gap-4">
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      <span className="text-lg">{agent1.archetype.icon}</span>
                      <span className="text-sm font-medium text-gray-700 dark:text-gray-300 truncate">
                        {agent1.archetype.name}
                      </span>
                      <span className="text-gray-400 dark:text-gray-500">↔</span>
                      <span className="text-lg">{agent2.archetype.icon}</span>
                      <span className="text-sm font-medium text-gray-700 dark:text-gray-300 truncate">
                        {agent2.archetype.name}
                      </span>
                    </div>
                    <div className="flex items-center gap-3 flex-shrink-0">
                      <div className="w-32 h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-blue-400 to-blue-600 rounded-full transition-all duration-300"
                          style={{ width: `${similarity}%` }}
                        />
                      </div>
                      <span className="font-mono font-bold text-sm w-12 text-right text-blue-600 dark:text-blue-400">
                        {similarity}%
                      </span>
                    </div>
                  </div>
                );
              })
            )}
          </div>

          {/* Similarity interpretation */}
          <div className="mt-4 pt-4 border-t border-blue-200 dark:border-blue-800">
            <p className="text-xs text-gray-600 dark:text-gray-400">
              <strong>Similarity Score:</strong> 100% = identical profiles, 0% = maximum difference
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
