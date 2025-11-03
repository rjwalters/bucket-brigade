/**
 * Team Radar Overlay Component
 *
 * Shows combined team radar with balance score and coverage analysis
 */

import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer } from 'recharts';
import {
  formatForChart,
  calculateTeamBalance,
  calculateTeamCoverage,
  RADAR_DIMENSIONS,
  type RadarProfile,
} from '../../utils/agentRadarChart';
import type { AgentArchetype } from '../../types/teamBuilder';

interface TeamRadarOverlayProps {
  team: Array<{ archetype: AgentArchetype; radarProfile: RadarProfile }>;
}

export function TeamRadarOverlay({ team }: TeamRadarOverlayProps) {
  if (team.length === 0) {
    return (
      <div className="p-8 text-center text-gray-500 dark:text-gray-400">
        <p>No agents in team. Add agents to see team analysis.</p>
      </div>
    );
  }

  // Extract radar profiles from team
  const profiles: RadarProfile[] = team.map((agent) => agent.radarProfile);

  // Calculate team metrics
  const teamBalance = calculateTeamBalance(profiles);
  const teamCoverage = calculateTeamCoverage(profiles);

  // Calculate average team profile
  const avgProfile: RadarProfile = {
    cooperation: profiles.reduce((sum, p) => sum + p.cooperation, 0) / profiles.length,
    reliability: profiles.reduce((sum, p) => sum + p.reliability, 0) / profiles.length,
    workEthic: profiles.reduce((sum, p) => sum + p.workEthic, 0) / profiles.length,
    selfPreservation: profiles.reduce((sum, p) => sum + p.selfPreservation, 0) / profiles.length,
    riskManagement: profiles.reduce((sum, p) => sum + p.riskManagement, 0) / profiles.length,
    initiative: profiles.reduce((sum, p) => sum + p.initiative, 0) / profiles.length,
  };

  const chartData = formatForChart(avgProfile);

  // Determine balance rating
  const getBalanceRating = (score: number): { text: string; color: string } => {
    if (score >= 80) return { text: 'Excellent', color: 'text-green-600 dark:text-green-400' };
    if (score >= 60) return { text: 'Good', color: 'text-blue-600 dark:text-blue-400' };
    if (score >= 40) return { text: 'Fair', color: 'text-yellow-600 dark:text-yellow-400' };
    return { text: 'Poor', color: 'text-red-600 dark:text-red-400' };
  };

  const balanceRating = getBalanceRating(teamBalance);

  return (
    <div className="space-y-6">
      {/* Team balance score */}
      <div className="p-6 bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <h3 className="text-lg font-bold text-gray-900 dark:text-gray-100 mb-1">
              Team Balance
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              How well-rounded is this team across all dimensions?
            </p>
            <p className={`text-sm font-semibold mt-2 ${balanceRating.color}`}>
              {balanceRating.text} Balance
            </p>
          </div>
          <div className="text-center">
            <div className="text-4xl font-bold text-purple-600 dark:text-purple-400">
              {teamBalance}%
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">Balance Score</div>
          </div>
        </div>
      </div>

      {/* Average team radar */}
      <div className="p-6 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700">
        <h4 className="text-base font-semibold mb-4 text-center text-gray-900 dark:text-gray-100">
          Average Team Profile
        </h4>
        <ResponsiveContainer width="100%" height={350}>
          <RadarChart data={chartData}>
            <PolarGrid stroke="#e5e7eb" strokeDasharray="3 3" />
            <PolarAngleAxis dataKey="dimension" tick={{ fill: '#6b7280', fontSize: 12 }} />
            <PolarRadiusAxis angle={90} domain={[0, 10]} tick={{ fill: '#9ca3af', fontSize: 10 }} />
            <Radar
              name="Team Average"
              dataKey="value"
              stroke="#8b5cf6"
              fill="#8b5cf6"
              fillOpacity={0.4}
              strokeWidth={3}
            />
          </RadarChart>
        </ResponsiveContainer>
        <p className="text-xs text-center text-gray-500 dark:text-gray-400 mt-2">
          Team of {team.length} agent{team.length !== 1 ? 's' : ''}
        </p>
      </div>

      {/* Team coverage breakdown */}
      <div className="p-6 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700">
        <h4 className="text-base font-semibold mb-4 text-gray-900 dark:text-gray-100">
          Team Coverage by Dimension
        </h4>
        <div className="space-y-3">
          {RADAR_DIMENSIONS.map((dim) => {
            const percentage = teamCoverage[dim.key] / 100;
            const isLow = percentage < 0.3;
            const isHigh = percentage > 0.7;

            return (
              <div key={dim.key} className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span className="font-medium text-gray-700 dark:text-gray-300">{dim.label}</span>
                  <span
                    className={`font-mono font-semibold ${
                      isLow
                        ? 'text-red-600 dark:text-red-400'
                        : isHigh
                          ? 'text-green-600 dark:text-green-400'
                          : 'text-gray-600 dark:text-gray-400'
                    }`}
                  >
                    {Math.round(percentage * 100)}%
                  </span>
                </div>
                <div className="w-full h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-300 ${
                      isLow ? 'bg-red-500' : isHigh ? 'bg-green-500' : 'bg-blue-500'
                    }`}
                    style={{ width: `${percentage * 100}%` }}
                  />
                </div>
                <p className="text-xs text-gray-500 dark:text-gray-400">{dim.description}</p>
              </div>
            );
          })}
        </div>

        {/* Missing dimensions warning */}
        {Object.entries(teamCoverage).some(([_, pct]) => pct < 30) && (
          <div className="mt-6 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
            <p className="text-sm text-yellow-900 dark:text-yellow-200 font-medium mb-1">
              ⚠️ Low Coverage Warning
            </p>
            <p className="text-xs text-yellow-800 dark:text-yellow-300">
              This team lacks strength in{' '}
              <strong>
                {RADAR_DIMENSIONS.filter((dim) => teamCoverage[dim.key] < 30)
                  .map((dim) => dim.label)
                  .join(', ')}
              </strong>
              . Consider adding agents with higher values in these dimensions.
            </p>
          </div>
        )}

        {/* Excellent coverage celebration */}
        {Object.entries(teamCoverage).every(([_, pct]) => pct >= 60) && (
          <div className="mt-6 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
            <p className="text-sm text-green-900 dark:text-green-200 font-medium">
              ✨ Well-Balanced Team!
            </p>
            <p className="text-xs text-green-800 dark:text-green-300">
              This team has good coverage across all dimensions.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
