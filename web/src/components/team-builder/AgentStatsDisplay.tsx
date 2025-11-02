/**
 * Agent Stats Display Component
 *
 * Visual display of agent parameters as stat bars
 */

import type { AgentArchetype } from '../../types/teamBuilder';
import { getStatDisplays } from '../../utils/agentArchetypes';

interface AgentStatsDisplayProps {
  archetype: AgentArchetype;
  compact?: boolean;
}

export function AgentStatsDisplay({ archetype, compact = false }: AgentStatsDisplayProps) {
  const stats = getStatDisplays(archetype.parameters);

  return (
    <div className="space-y-3">
      {stats.map((stat) => (
        <div key={stat.label}>
          <div className="flex justify-between items-center mb-1">
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              {stat.label}
            </span>
            {!compact && (
              <span className="text-xs text-gray-500 dark:text-gray-400">
                {Math.round(stat.value * 10)}/10
              </span>
            )}
          </div>
          <div className="relative h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              className="absolute top-0 left-0 h-full rounded-full transition-all duration-300"
              style={{
                width: `${stat.value * 100}%`,
                backgroundColor: archetype.color,
              }}
            />
          </div>
          {!compact && (
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              {stat.description}
            </p>
          )}
        </div>
      ))}
    </div>
  );
}

/**
 * Compact mini version for agent cards
 */
export function AgentStatsMini({ archetype }: { archetype: AgentArchetype }) {
  const topStats = getStatDisplays(archetype.parameters).slice(0, 3);

  return (
    <div className="space-y-1">
      {topStats.map((stat) => (
        <div key={stat.label} className="flex items-center gap-2">
          <span className="text-xs text-gray-600 dark:text-gray-400 w-20 truncate">
            {stat.label}
          </span>
          <div className="flex-1 h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all"
              style={{
                width: `${stat.value * 100}%`,
                backgroundColor: archetype.color,
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
