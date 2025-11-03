/**
 * Tournament Results Component
 *
 * Displays comprehensive tournament analysis
 */

import { Link } from 'react-router-dom';
import type { TournamentResult } from '../../types/teamBuilder';
import { getScenarioTemplate } from '../../utils/scenarioGenerator';

interface TournamentResultsProps {
  result: TournamentResult;
  onClose: () => void;
  onRunAgain?: () => void;
  onSave?: () => void;
}

export function TournamentResults({
  result,
  onClose,
  onRunAgain,
  onSave,
}: TournamentResultsProps) {
  const { statistics, agentContributions, scenarioTypePerformance } = result;

  // Score distribution bins
  const createHistogram = (scores: number[], bins = 10) => {
    const min = Math.min(...scores);
    const max = Math.max(...scores);
    const range = max - min;
    const binSize = range / bins;

    const histogram = Array(bins).fill(0);
    scores.forEach((score) => {
      const binIndex = Math.min(Math.floor((score - min) / binSize), bins - 1);
      histogram[binIndex]++;
    });

    return histogram.map((count, i) => ({
      bin: min + i * binSize,
      count,
    }));
  };

  const scores = result.scenarios.map((s) => s.teamScore);
  const histogram = createHistogram(scores);
  const maxCount = Math.max(...histogram.map((h) => h.count));

  const formatDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleString();
  };

  const formatDuration = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes}m ${secs}s`;
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto bg-gray-50 dark:bg-gray-900">
      <div className="max-w-6xl mx-auto p-8">
        {/* Header */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8 mb-8">
          <div className="text-center mb-6">
            <h1 className="text-4xl font-bold text-gray-900 dark:text-gray-100 mb-2">
              🎉 TOURNAMENT COMPLETE! 🎉
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              {result.teamName} • {formatDate(result.timestamp)}
            </p>
          </div>

          {/* Overall Performance */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 text-center">
              <div className="text-sm text-blue-600 dark:text-blue-400 mb-1">
                Average Score
              </div>
              <div className="text-3xl font-bold text-blue-700 dark:text-blue-300">
                {statistics.mean.toFixed(1)}
              </div>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4 text-center">
              <div className="text-sm text-green-600 dark:text-green-400 mb-1">
                Success Rate
              </div>
              <div className="text-3xl font-bold text-green-700 dark:text-green-300">
                {(statistics.successRate * 100).toFixed(0)}%
              </div>
            </div>

            <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4 text-center">
              <div className="text-sm text-purple-600 dark:text-purple-400 mb-1">
                Houses Saved
              </div>
              <div className="text-3xl font-bold text-purple-700 dark:text-purple-300">
                {statistics.housesSavedAvg.toFixed(1)}/10
              </div>
            </div>

            <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-4 text-center">
              <div className="text-sm text-orange-600 dark:text-orange-400 mb-1">
                Duration
              </div>
              <div className="text-3xl font-bold text-orange-700 dark:text-orange-300">
                {formatDuration(result.duration)}
              </div>
            </div>
          </div>
        </div>

        {/* Score Distribution */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-4">
            Score Distribution
          </h2>
          <div className="flex items-end gap-2 h-48">
            {histogram.map((bar, idx) => (
              <div key={idx} className="flex-1 flex flex-col items-center">
                <div
                  className="w-full bg-blue-500 dark:bg-blue-600 rounded-t transition-all"
                  style={{
                    height: `${(bar.count / maxCount) * 100}%`,
                  }}
                  title={`${bar.bin.toFixed(0)}-${(bar.bin + (statistics.max - statistics.min) / 10).toFixed(0)}: ${bar.count} games`}
                />
                {idx % 2 === 0 && (
                  <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                    {bar.bin.toFixed(0)}
                  </div>
                )}
              </div>
            ))}
          </div>
          <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mt-2">
            <div>Min: {statistics.min.toFixed(1)}</div>
            <div>Median: {statistics.median.toFixed(1)}</div>
            <div>Max: {statistics.max.toFixed(1)}</div>
          </div>
        </div>

        {/* MVP Agents */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-4">
            🏆 Agent Performance Rankings
          </h2>

          {/* Top 3 */}
          <div className="grid grid-cols-3 gap-4 mb-6">
            {agentContributions.slice(0, 3).map((agent, idx) => {
              const medals = ['🥇', '🥈', '🥉'];
              const colors = [
                'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800',
                'bg-gray-50 dark:bg-gray-700 border-gray-200 dark:border-gray-600',
                'bg-orange-50 dark:bg-orange-900/20 border-orange-200 dark:border-orange-800',
              ];

              return (
                <div
                  key={agent.position}
                  className={`border-2 rounded-lg p-4 ${colors[idx]}`}
                >
                  <div className="text-3xl text-center mb-2">{medals[idx]}</div>
                  <div className="text-center">
                    <div className="font-bold text-gray-900 dark:text-gray-100 mb-1">
                      {agent.archetype}
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      House {agent.position}
                    </div>
                    <div className="text-2xl font-bold text-gray-900 dark:text-gray-100 mt-2">
                      +{agent.avgContribution.toFixed(1)}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      {agent.mvpCount} MVP{agent.mvpCount !== 1 ? 's' : ''}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Full Rankings */}
          <div className="space-y-2">
            {agentContributions.map((agent) => (
              <div
                key={agent.position}
                className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg"
              >
                <div className="flex items-center gap-4">
                  <div className="text-lg font-bold text-gray-600 dark:text-gray-400 w-8">
                    #{agent.rank}
                  </div>
                  <div>
                    <div className="font-medium text-gray-900 dark:text-gray-100">
                      {agent.archetype}
                    </div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                      House {agent.position}
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-lg font-bold text-gray-900 dark:text-gray-100">
                    {agent.avgContribution >= 0 ? '+' : ''}
                    {agent.avgContribution.toFixed(1)}
                  </div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">
                    {(agent.consistency * 100).toFixed(0)}% consistent
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Scenario Type Performance */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-4">
            Performance by Scenario Type
          </h2>
          <div className="space-y-3">
            {Object.entries(scenarioTypePerformance).map(([type, stats]) => {
              const template = getScenarioTemplate(type as any);
              return (
                <div key={type} className="flex items-center gap-4">
                  <div className="flex-1">
                    <div className="font-medium text-gray-900 dark:text-gray-100">
                      {template?.name || type}
                    </div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                      {stats.count} scenarios
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="text-right">
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        Avg Score
                      </div>
                      <div className="font-bold text-gray-900 dark:text-gray-100">
                        {stats.avgScore.toFixed(1)}
                      </div>
                    </div>
                    <div className="w-24">
                      <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-green-500 dark:bg-green-600 rounded-full"
                          style={{ width: `${stats.successRate * 100}%` }}
                        />
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                        {(stats.successRate * 100).toFixed(0)}% success
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Game Replay Info */}
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 mb-6">
          <div className="flex items-start gap-3">
            <div className="text-2xl">🎬</div>
            <div className="flex-1">
              <h3 className="font-semibold text-blue-900 dark:text-blue-100 mb-1">
                Sample Games Saved!
              </h3>
              <p className="text-sm text-blue-700 dark:text-blue-300 mb-2">
                We've automatically saved some interesting games from this tournament (best, worst, median, and a few random samples).
                View them in the Game Replay tab to see exactly how your agents performed.
              </p>
              <Link
                to="/replay"
                className="inline-flex items-center px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors"
              >
                🎮 View Game Replays →
              </Link>
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-4">
          <button
            type="button"
            onClick={onClose}
            className="flex-1 px-6 py-3 bg-gray-600 hover:bg-gray-700 text-white font-semibold rounded-lg transition-colors"
          >
            🏠 Back to Team Builder
          </button>
          {onSave && (
            <button
              type="button"
              onClick={onSave}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg transition-colors"
            >
              💾 Save Results
            </button>
          )}
          {onRunAgain && (
            <button
              type="button"
              onClick={onRunAgain}
              className="px-6 py-3 bg-green-600 hover:bg-green-700 text-white font-semibold rounded-lg transition-colors"
            >
              🔄 Run Again
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
