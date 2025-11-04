import React from 'react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts';
import { calculateRadarProfile, formatForChart, RADAR_DIMENSIONS } from '../utils/agentRadarChart';
import type { ArchetypeParams } from '../data/archetypes';

interface AgentRadarChartProps {
  params: ArchetypeParams;
  className?: string;
}

export const AgentRadarChart: React.FC<AgentRadarChartProps> = ({ params, className = '' }) => {
  const profile = calculateRadarProfile(params);
  const chartData = formatForChart(profile);

  return (
    <div className={className}>
      <ResponsiveContainer width="100%" height={300}>
        <RadarChart data={chartData}>
          <PolarGrid stroke="rgb(var(--color-border-primary))" />
          <PolarAngleAxis
            dataKey="dimension"
            tick={{ fill: 'rgb(var(--color-text-primary))', fontSize: 12 }}
          />
          <PolarRadiusAxis
            angle={90}
            domain={[0, 10]}
            tick={{ fill: 'rgb(var(--color-text-secondary))', fontSize: 10 }}
          />
          <Radar
            name="Agent Profile"
            dataKey="value"
            stroke="#3b82f6"
            fill="#3b82f6"
            fillOpacity={0.3}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
};
