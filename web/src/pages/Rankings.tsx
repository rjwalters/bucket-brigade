import React, { useState, useEffect } from 'react';
import { Trophy, TrendingUp, Users, Target } from 'lucide-react';
import { AgentRanking, BatchResult, STORAGE_KEYS } from '../types';
import { loadFromStorage } from '../utils/storage';

const Rankings: React.FC = () => {
  const [rankings, setRankings] = useState<AgentRanking[]>([]);
  const [batchResults, setBatchResults] = useState<BatchResult[]>([]);

  useEffect(() => {
    loadRankingsData();
  }, []);

  const loadRankingsData = () => {
    const results = loadFromStorage<BatchResult[]>(STORAGE_KEYS.BATCH_RESULTS, []);
    setBatchResults(results);

    if (results.length > 0) {
      // Calculate rankings from batch results
      const agentStats = new Map<number, { totalReward: number; gamesPlayed: number; rewards: number[] }>();

      results.forEach(result => {
        result.agent_rewards.forEach((reward, agentId) => {
          if (!agentStats.has(agentId)) {
            agentStats.set(agentId, { totalReward: 0, gamesPlayed: 0, rewards: [] });
          }
          const stats = agentStats.get(agentId)!;
          stats.totalReward += reward;
          stats.gamesPlayed += 1;
          stats.rewards.push(reward);
        });
      });

      const calculatedRankings: AgentRanking[] = Array.from(agentStats.entries())
        .map(([agentId, stats]) => ({
          agent_id: agentId,
          score: stats.totalReward / stats.gamesPlayed,
          games_played: stats.gamesPlayed,
          avg_reward: stats.totalReward / stats.gamesPlayed,
          uncertainty: calculateUncertainty(stats.rewards)
        }))
        .sort((a, b) => b.score - a.score);

      setRankings(calculatedRankings);
    }
  };

  const calculateUncertainty = (rewards: number[]): number => {
    if (rewards.length < 2) return 0;
    const mean = rewards.reduce((a, b) => a + b, 0) / rewards.length;
    const variance = rewards.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (rewards.length - 1);
    return Math.sqrt(variance / rewards.length); // Standard error
  };

  const getRankIcon = (rank: number) => {
    switch (rank) {
      case 1: return '🥇';
      case 2: return '🥈';
      case 3: return '🥉';
      default: return `#${rank}`;
    }
  };

  const getRankColor = (rank: number) => {
    switch (rank) {
      case 1: return 'text-yellow-600';
      case 2: return 'text-gray-600';
      case 3: return 'text-orange-600';
      default: return 'text-gray-600';
    }
  };

  if (rankings.length === 0) {
    return (
      <div className="text-center py-12">
        <Trophy className="w-16 h-16 mx-auto mb-4 text-gray-300" />
        <h2 className="text-2xl font-bold text-gray-900 mb-4">No Rankings Available</h2>
        <p className="text-gray-600 max-w-md mx-auto">
          Rankings will be calculated from batch experiment results. Run some games and upload the data to see agent performance.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Agent Rankings</h1>
        <p className="text-lg text-gray-600">
          Performance rankings based on {batchResults.length} games played
        </p>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="card text-center">
          <Trophy className="w-8 h-8 mx-auto mb-2 text-yellow-600" />
          <div className="text-2xl font-bold text-gray-900">{rankings[0]?.score.toFixed(1) || 'N/A'}</div>
          <div className="text-sm text-gray-600">Top Score</div>
        </div>
        <div className="card text-center">
          <Users className="w-8 h-8 mx-auto mb-2 text-blue-600" />
          <div className="text-2xl font-bold text-gray-900">{rankings.length}</div>
          <div className="text-sm text-gray-600">Active Agents</div>
        </div>
        <div className="card text-center">
          <Target className="w-8 h-8 mx-auto mb-2 text-green-600" />
          <div className="text-2xl font-bold text-gray-900">{batchResults.length}</div>
          <div className="text-sm text-gray-600">Games Played</div>
        </div>
        <div className="card text-center">
          <TrendingUp className="w-8 h-8 mx-auto mb-2 text-purple-600" />
          <div className="text-2xl font-bold text-gray-900">
            {(rankings.reduce((sum, r) => sum + r.avg_reward, 0) / rankings.length).toFixed(1)}
          </div>
          <div className="text-sm text-gray-600">Avg Performance</div>
        </div>
      </div>

      {/* Rankings Table */}
      <div className="card">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Performance Rankings</h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-3 px-4 font-semibold text-gray-900">Rank</th>
                <th className="text-left py-3 px-4 font-semibold text-gray-900">Agent</th>
                <th className="text-left py-3 px-4 font-semibold text-gray-900">Score</th>
                <th className="text-left py-3 px-4 font-semibold text-gray-900">Games</th>
                <th className="text-left py-3 px-4 font-semibold text-gray-900">Avg Reward</th>
                <th className="text-left py-3 px-4 font-semibold text-gray-900">Uncertainty</th>
              </tr>
            </thead>
            <tbody>
              {rankings.map((ranking, index) => (
                <tr key={ranking.agent_id} className="border-b border-gray-100 hover:bg-gray-50">
                  <td className="py-3 px-4">
                    <div className="flex items-center space-x-2">
                      <span className={`text-lg font-bold ${getRankColor(index + 1)}`}>
                        {getRankIcon(index + 1)}
                      </span>
                      <span className="font-medium text-gray-900">#{index + 1}</span>
                    </div>
                  </td>
                  <td className="py-3 px-4 font-medium text-gray-900">
                    Agent {ranking.agent_id}
                  </td>
                  <td className="py-3 px-4 font-mono font-semibold text-blue-600">
                    {ranking.score.toFixed(2)}
                  </td>
                  <td className="py-3 px-4 text-gray-600">
                    {ranking.games_played}
                  </td>
                  <td className="py-3 px-4 font-mono text-gray-900">
                    {ranking.avg_reward.toFixed(2)}
                  </td>
                  <td className="py-3 px-4 font-mono text-gray-600">
                    ±{ranking.uncertainty?.toFixed(2) || 'N/A'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Rankings;
