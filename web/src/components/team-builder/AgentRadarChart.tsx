/**
 * Agent Radar Chart Component
 *
 * Visual radar chart displaying agent behavioral parameters
 * across 6 key dimensions using Recharts.
 */

import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip } from 'recharts';
import type { AgentArchetype } from '../../types/teamBuilder';
import { RADAR_DIMENSIONS, type RadarProfile } from '../../utils/agentRadarChart';

interface AgentRadarChartProps {
  archetype: AgentArchetype;
  radarProfile: RadarProfile;
  size?: 'small' | 'medium' | 'large';
  showLabels?: boolean;
}

export function AgentRadarChart({
  archetype,
  radarProfile,
  size = 'medium',
  showLabels = true,
}: AgentRadarChartProps) {
  // Format data for Recharts
  const data = RADAR_DIMENSIONS.map((dim) => ({
    dimension: dim.label,
    value: radarProfile[dim.key],
    max: dim.max,
    description: dim.description,
  }));

  // Size configurations
  const sizeConfig = {
    small: { width: 200, height: 200, fontSize: 10 },
    medium: { width: 300, height: 300, fontSize: 12 },
    large: { width: 400, height: 400, fontSize: 14 },
  };

  const config = sizeConfig[size];

  return (
    <div className="flex flex-col items-center">
      <ResponsiveContainer width="100%" height={config.height}>
        <RadarChart data={data}>
          <PolarGrid stroke="#e5e7eb" />
          <PolarAngleAxis
            dataKey="dimension"
            tick={{ fill: '#6b7280', fontSize: config.fontSize }}
          />
          <PolarRadiusAxis
            angle={90}
            domain={[0, 10]}
            tick={{ fill: '#9ca3af', fontSize: config.fontSize - 2 }}
          />
          <Radar
            name={archetype.name}
            dataKey="value"
            stroke={archetype.color}
            fill={archetype.color}
            fillOpacity={0.5}
            strokeWidth={2}
          />
          {showLabels && (
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length > 0) {
                  const item = payload[0].payload;
                  return (
                    <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-3 shadow-lg">
                      <p className="font-semibold text-gray-900 dark:text-gray-100 mb-1">
                        {item.dimension}
                      </p>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                        {item.description}
                      </p>
                      <p className="text-lg font-bold" style={{ color: archetype.color }}>
                        {item.value}/10
                      </p>
                    </div>
                  );
                }
                return null;
              }}
            />
          )}
        </RadarChart>
      </ResponsiveContainer>

      {showLabels && (
        <div className="mt-4 text-center">
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Hover over each dimension for details
          </p>
        </div>
      )}
    </div>
  );
}

/**
 * Compact version for displaying in agent cards
 */
export function AgentRadarChartCompact({ archetype, radarProfile }: { archetype: AgentArchetype; radarProfile: RadarProfile }) {
  const data = RADAR_DIMENSIONS.map((dim) => ({
    dimension: dim.label.substring(0, 4), // Abbreviated labels
    value: radarProfile[dim.key],
  }));

  return (
    <ResponsiveContainer width="100%" height={150}>
      <RadarChart data={data}>
        <PolarGrid stroke="#e5e7eb" strokeDasharray="3 3" />
        <PolarAngleAxis
          dataKey="dimension"
          tick={{ fill: '#9ca3af', fontSize: 10 }}
        />
        <Radar
          dataKey="value"
          stroke={archetype.color}
          fill={archetype.color}
          fillOpacity={0.6}
          strokeWidth={2}
        />
      </RadarChart>
    </ResponsiveContainer>
  );
}

/**
 * Comparison radar chart showing multiple agents
 */
export function AgentRadarChartComparison({
  agents,
  size = 'large',
}: {
  agents: Array<{ archetype: AgentArchetype; radarProfile: RadarProfile }>;
  size?: 'small' | 'medium' | 'large';
}) {
  const sizeConfig = {
    small: { height: 250, fontSize: 10 },
    medium: { height: 350, fontSize: 12 },
    large: { height: 450, fontSize: 14 },
  };

  const config = sizeConfig[size];

  // Combine all radar profiles into one dataset
  const data = RADAR_DIMENSIONS.map((dim) => {
    const point: any = {
      dimension: dim.label,
    };

    agents.forEach((agent, idx) => {
      point[`agent${idx}`] = agent.radarProfile[dim.key];
    });

    return point;
  });

  return (
    <div className="flex flex-col items-center">
      <ResponsiveContainer width="100%" height={config.height}>
        <RadarChart data={data}>
          <PolarGrid stroke="#e5e7eb" />
          <PolarAngleAxis
            dataKey="dimension"
            tick={{ fill: '#6b7280', fontSize: config.fontSize }}
          />
          <PolarRadiusAxis
            angle={90}
            domain={[0, 10]}
            tick={{ fill: '#9ca3af', fontSize: config.fontSize - 2 }}
          />
          {agents.map((agent, idx) => (
            <Radar
              key={agent.archetype.id}
              name={agent.archetype.name}
              dataKey={`agent${idx}`}
              stroke={agent.archetype.color}
              fill={agent.archetype.color}
              fillOpacity={0.3}
              strokeWidth={2}
            />
          ))}
          <Tooltip />
        </RadarChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex flex-wrap justify-center gap-4 mt-4">
        {agents.map((agent) => (
          <div key={agent.archetype.id} className="flex items-center gap-2">
            <div
              className="w-4 h-4 rounded-full"
              style={{ backgroundColor: agent.archetype.color }}
            />
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              {agent.archetype.name}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
