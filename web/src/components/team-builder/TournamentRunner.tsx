/**
 * Tournament Runner Component
 *
 * Shows progress while tournament is running
 */

import { useEffect, useState } from 'react';
import type { TournamentProgress } from '../../types/teamBuilder';

interface TournamentRunnerProps {
  progress: TournamentProgress;
  onCancel?: () => void;
}

export function TournamentRunner({ progress, onCancel }: TournamentRunnerProps) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setElapsed(Date.now() - progress.startTime);
    }, 100);

    return () => clearInterval(interval);
  }, [progress.startTime]);

  const percentComplete = (progress.current / progress.total) * 100;
  const timeRemaining = Math.max(0, progress.estimatedCompletion - Date.now());

  const formatTime = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes}m ${secs}s`;
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black bg-opacity-50">
      <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-2xl max-w-2xl w-full p-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-gray-100 mb-2">
            🏆 TOURNAMENT IN PROGRESS
          </h2>
          <p className="text-gray-600 dark:text-gray-400">
            Testing your team across diverse scenarios...
          </p>
        </div>

        {/* Progress Bar */}
        <div className="mb-6">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Scenario: {progress.current} / {progress.total}
            </span>
            <span className="text-sm font-bold text-blue-600 dark:text-blue-400">
              {percentComplete.toFixed(0)}%
            </span>
          </div>
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-300 ease-out"
              style={{ width: `${percentComplete}%` }}
            />
          </div>
        </div>

        {/* Statistics */}
        {progress.results.length > 0 && (
          <div className="grid grid-cols-2 gap-4 mb-6">
            <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                Current Score
              </div>
              <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {progress.results[progress.results.length - 1].teamScore.toFixed(1)}
              </div>
            </div>

            <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                Average Score
              </div>
              <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {progress.statistics.mean.toFixed(1)}
              </div>
            </div>

            <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Best Score</div>
              <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                {progress.statistics.max.toFixed(1)}
              </div>
            </div>

            <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Worst Score</div>
              <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                {progress.statistics.min.toFixed(1)}
              </div>
            </div>
          </div>
        )}

        {/* Time Information */}
        <div className="flex justify-between items-center text-sm text-gray-600 dark:text-gray-400 mb-6">
          <div>Elapsed: {formatTime(elapsed)}</div>
          <div>Remaining: ~{formatTime(timeRemaining)}</div>
        </div>

        {/* Mini Visualization */}
        {progress.results.length > 5 && (
          <div className="mb-6">
            <div className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Score Trend
            </div>
            <div className="h-16 flex items-end gap-1">
              {progress.results.slice(-20).map((result, idx) => {
                const height =
                  (result.teamScore / (progress.statistics.max || 1)) * 100;
                return (
                  <div
                    key={idx}
                    className="flex-1 bg-blue-500 dark:bg-blue-600 rounded-t transition-all"
                    style={{ height: `${Math.max(5, height)}%` }}
                    title={`Score: ${result.teamScore.toFixed(1)}`}
                  />
                );
              })}
            </div>
          </div>
        )}

        {/* Cancel Button */}
        {onCancel && (
          <div className="text-center">
            <button
              type="button"
              onClick={onCancel}
              className="px-6 py-2 bg-gray-600 hover:bg-gray-700 text-white font-medium rounded-lg transition-colors"
            >
              Cancel Tournament
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
