import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Play, TrendingUp, Users, Flame, FileText, Trophy, Code, Zap } from 'lucide-react';
import { GameReplay } from '../types';
import { loadGameReplays, loadBatchResults } from '../utils/storage';

const Dashboard: React.FC = () => {
  const [gameCount, setGameCount] = useState(0);
  const [totalGames, setTotalGames] = useState(0);
  const [avgReward, setAvgReward] = useState(0);
  const [recentGames, setRecentGames] = useState<GameReplay[]>([]);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = () => {
    const replays = loadGameReplays();
    const results = loadBatchResults();

    setGameCount(replays.length);
    setTotalGames(results.length);

    if (results.length > 0) {
      const avg = results.reduce((sum, r) => sum + r.team_reward, 0) / results.length;
      setAvgReward(avg);
    }

    // Get recent games (last 5)
    setRecentGames(replays.slice(-5).reverse());
  };

  const stats = [
    {
      label: 'Games Played',
      value: totalGames.toString(),
      icon: Play,
      color: 'text-blue-600',
      bgColor: 'bg-blue-100'
    },
    {
      label: 'Stored Replays',
      value: gameCount.toString(),
      icon: FileText,
      color: 'text-green-600',
      bgColor: 'bg-green-100'
    },
    {
      label: 'Avg Team Reward',
      value: avgReward.toFixed(1),
      icon: TrendingUp,
      color: 'text-purple-600',
      bgColor: 'bg-purple-100'
    },
    {
      label: 'Active Agents',
      value: '6', // This could be calculated from data
      icon: Users,
      color: 'text-orange-600',
      bgColor: 'bg-orange-100'
    }
  ];

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="text-center mb-8 bg-gradient-to-r from-orange-50 to-red-50 rounded-lg p-8 border border-orange-200">
      <div className="flex items-center justify-center mb-4">
      <Flame className="w-10 h-10 text-orange-600 mr-3" />
        <h1 className="text-4xl font-bold text-gray-900">Bucket Brigade</h1>
      </div>
      <p className="text-xl text-gray-700 mb-4 max-w-3xl mx-auto">
      A frontier town where <strong>10 houses stand in a circle</strong>, connected by paths that carry both people and relentless fire.
        When flames erupt, they leap from home to home with terrifying speed.
        </p>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto mb-6">
          The townsfolk have formed a <strong>Bucket Brigade</strong> — but not everyone wants to be a hero.
          Some are exhausted, others selfish, a few might even deceive. In this microcosm of human nature,
          <strong>cooperation isn't guaranteed — it's earned</strong>.
        </p>
        <div className="flex flex-wrap justify-center gap-4 text-sm">
          <div className="flex items-center bg-white px-3 py-1 rounded-full border border-orange-200">
            <Code className="w-4 h-4 mr-2 text-blue-600" />
            Write JavaScript agents
          </div>
          <div className="flex items-center bg-white px-3 py-1 rounded-full border border-orange-200">
            <Trophy className="w-4 h-4 mr-2 text-yellow-600" />
            Compete in tournaments
          </div>
          <div className="flex items-center bg-white px-3 py-1 rounded-full border border-orange-200">
            <Zap className="w-4 h-4 mr-2 text-green-600" />
            Real-time rankings
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <div key={index} className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">{stat.label}</p>
                  <p className="text-2xl font-bold text-gray-900">{stat.value}</p>
                </div>
                <div className={`p-3 rounded-full ${stat.bgColor}`}>
                  <Icon className={`w-6 h-6 ${stat.color}`} />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Games */}
        <div className="card">
          <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
            <Play className="w-5 h-5 mr-2 text-blue-600" />
            Recent Games
          </h2>
          {recentGames.length > 0 ? (
            <div className="space-y-3">
              {recentGames.map((game, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div>
                    <p className="font-medium text-gray-900">
                      Game #{game.nights[game.nights.length - 1]?.night || 0}
                    </p>
                    <p className="text-sm text-gray-600">
                      {game.nights.length} nights • {game.scenario.num_agents} agents
                    </p>
                  </div>
                  <Link
                    to={`/replay/${index}`}
                    className="btn-primary text-sm"
                  >
                    Watch Replay
                  </Link>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <FileText className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>No games loaded yet</p>
              <p className="text-sm">Upload game data to get started</p>
            </div>
          )}
        </div>

        {/* Getting Started */}
        <div className="card">
          <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
            <Flame className="w-5 h-5 mr-2 text-orange-600" />
            Getting Started
          </h2>
          <div className="space-y-4">
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0 w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                <span className="text-sm font-medium text-blue-600">1</span>
              </div>
              <div>
                <h3 className="font-medium text-gray-900">Run Batch Experiments</h3>
                <p className="text-sm text-gray-600">
                  Use <code className="bg-gray-100 px-1 rounded">python scripts/run_batch.py</code> to generate game data
                </p>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0 w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                <span className="text-sm font-medium text-blue-600">2</span>
              </div>
              <div>
                <h3 className="font-medium text-gray-900">Upload Game Data</h3>
                <p className="text-sm text-gray-600">
                  Import JSON replays and CSV results using the settings panel
                </p>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0 w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                <span className="text-sm font-medium text-blue-600">3</span>
              </div>
              <div>
                <h3 className="font-medium text-gray-900">Analyze & Visualize</h3>
                <p className="text-sm text-gray-600">
                  Explore game replays and track agent performance in rankings
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
