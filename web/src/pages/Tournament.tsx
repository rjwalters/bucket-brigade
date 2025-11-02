import React, { useState, useRef, useCallback } from 'react';
import { Play, Square, Code, Trophy } from 'lucide-react';
import {
  TournamentRunner,
  Agent,
  SCENARIOS,
  GameResult
} from '../utils/browserEngine';
import {
  create_tournament_agents
} from '../utils/browserAgents';

const Tournament: React.FC = () => {
  const [selectedScenario, setSelectedScenario] = useState<keyof typeof SCENARIOS>('trivial_cooperation');
  const [userAgentCode, setUserAgentCode] = useState(`// Your agent function
// Input: obs (observation object)
// Output: [house_index, mode] where mode is 0=REST, 1=WORK

return function(obs) {
  // obs contains:
  // - obs.houses: array of 10 house states (0=SAFE, 1=BURNING, 2=RUINED)
  // - obs.signals: array of agent signals (0=REST, 1=WORK)
  // - obs.locations: array of agent positions (0-9)
  // - obs.last_actions: array of [house, mode] from last night
  // - obs.scenario_info: scenario parameters
  // - obs.agent_id: your agent ID
  // - obs.night: current night number

  // Example: Always work on first burning house
  const burning_house = obs.houses.findIndex(house => house === 1);
  if (burning_house !== -1) {
    return [burning_house, 1]; // Work on burning house
  }

  // No fires - rest
  return [obs.agent_id % 10, 0]; // Rest at own house
}`);
  const [userAgentName, setUserAgentName] = useState('MyAgent');
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [rankings, setRankings] = useState<Array<{agent: Agent; score: number; rank: number}>>([]);
  const [error, setError] = useState<string | null>(null);

  const tournament_runner_ref = useRef<TournamentRunner | null>(null);

  const scenario_options = Object.keys(SCENARIOS) as Array<keyof typeof SCENARIOS>;

  const run_tournament = useCallback(async () => {
    setError(null);
    setProgress(0);
    setRankings([]);

    // Create user agent
    const user_agent_func = new Function('obs', `
      try {
        ${userAgentCode}
      } catch (error) {
        console.error('Agent error:', error);
        return [Math.floor(Math.random() * 10), Math.floor(Math.random() * 2)];
      }
    `) as (obs: any) => number[];

    const user_agent = {
      name: userAgentName,
      act_function: user_agent_func
    };

    // Create all agents for tournament
    const agents = create_tournament_agents(user_agent);

    // Create tournament runner
    const scenario = SCENARIOS[selectedScenario];
    const runner = new TournamentRunner(agents, scenario, 20); // 20 games
    tournament_runner_ref.current = runner;

    setIsRunning(true);

    try {
      const game_results = await runner.run_tournament((completed, total) => {
        setProgress((completed / total) * 100);
      });

      calculate_rankings(agents, game_results);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Tournament failed');
    } finally {
      setIsRunning(false);
      tournament_runner_ref.current = null;
    }
  }, [selectedScenario, userAgentCode, userAgentName]);

  const stop_tournament = useCallback(() => {
    if (tournament_runner_ref.current) {
      tournament_runner_ref.current.stop();
      setIsRunning(false);
    }
  }, []);

  const calculate_rankings = (agents: Agent[], game_results: GameResult[]) => {
    // Calculate total scores for each agent across all games
    const agent_scores = new Map<number, number>();

    game_results.forEach(result => {
      result.agent_scores.forEach((score, agent_idx) => {
        const current = agent_scores.get(agent_idx) || 0;
        agent_scores.set(agent_idx, current + score);
      });
    });

    // Create rankings array with scores
    const ranking_data = agents.map(agent => ({
      agent,
      score: agent_scores.get(agent.id) || 0,
      rank: 0  // Placeholder, will be set after sorting
    }));

    // Sort by score (descending)
    ranking_data.sort((a, b) => b.score - a.score);

    // Add ranks
    const rankings_with_ranks = ranking_data.map((item, index) => ({
      ...item,
      rank: index + 1
    }));

    setRankings(rankings_with_ranks);
  };

  const get_rank_icon = (rank: number) => {
    switch (rank) {
      case 1: return '🥇';
      case 2: return '🥈';
      case 3: return '🥉';
      default: return `#${rank}`;
    }
  };

  const get_rank_color = (rank: number) => {
    switch (rank) {
      case 1: return 'text-yellow-600 bg-yellow-50';
      case 2: return 'text-gray-600 bg-gray-50';
      case 3: return 'text-orange-600 bg-orange-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-2 flex items-center justify-center">
          <Trophy className="w-8 h-8 mr-3 text-yellow-600" />
          Agent Tournament
        </h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto">
          Upload your agent and compete against expert strategies in real-time tournaments.
          See how you rank against optimal agents for each scenario!
        </p>
      </div>

      {/* Scenario Selection */}
      <div className="card">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Choose Scenario</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {scenario_options.map(scenario => (
            <label key={scenario} className="flex items-center space-x-3 p-4 border rounded-lg cursor-pointer hover:bg-gray-50">
              <input
                type="radio"
                name="scenario"
                value={scenario}
                checked={selectedScenario === scenario}
                onChange={(e) => setSelectedScenario(e.target.value as keyof typeof SCENARIOS)}
                className="text-blue-600 focus:ring-blue-500"
              />
              <div>
                <div className="font-medium text-gray-900 capitalize">
                  {scenario.replace('_', ' ')}
                </div>
                <div className="text-sm text-gray-600">
                  β={SCENARIOS[scenario].beta}, κ={SCENARIOS[scenario].kappa}, cost={SCENARIOS[scenario].c}
                </div>
              </div>
            </label>
          ))}
        </div>
      </div>

      {/* Agent Editor */}
      <div className="card">
        <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
          <Code className="w-5 h-5 mr-2 text-blue-600" />
          Your Agent
        </h2>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Agent Name
            </label>
            <input
              type="text"
              value={userAgentName}
              onChange={(e) => setUserAgentName(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Enter agent name"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Agent Code (JavaScript)
            </label>
            <textarea
              value={userAgentCode}
              onChange={(e) => setUserAgentCode(e.target.value)}
              className="w-full h-64 px-3 py-2 border border-gray-300 rounded-md font-mono text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Write your agent function..."
            />
          </div>

          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h4 className="font-medium text-blue-900 mb-2">Available in obs:</h4>
            <ul className="text-sm text-blue-800 space-y-1">
              <li>• <code>obs.houses</code> - Array of 10 house states (0=SAFE, 1=BURNING, 2=RUINED)</li>
              <li>• <code>obs.signals</code> - Array of agent signals (0=REST, 1=WORK)</li>
              <li>• <code>obs.locations</code> - Array of agent positions (0-9)</li>
              <li>• <code>obs.last_actions</code> - Previous night actions</li>
              <li>• <code>obs.scenario_info</code> - Scenario parameters</li>
              <li>• <code>obs.agent_id</code> - Your agent ID</li>
              <li>• <code>obs.night</code> - Current night number</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900">Tournament</h2>
          <div className="flex items-center space-x-4">
            <div className="text-sm text-gray-600">
              Progress: {Math.round(progress)}%
            </div>
            <div className="w-32 bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          {!isRunning ? (
            <button
              onClick={run_tournament}
              className="btn-primary flex items-center space-x-2"
            >
              <Play className="w-5 h-5" />
              <span>Start Tournament</span>
            </button>
          ) : (
            <button
              onClick={stop_tournament}
              className="btn-primary bg-red-600 hover:bg-red-700 flex items-center space-x-2"
            >
              <Square className="w-5 h-5" />
              <span>Stop Tournament</span>
            </button>
          )}

          <div className="text-sm text-gray-600">
            20 games • 10 agents • {scenario_options.find(s => s === selectedScenario)?.replace('_', ' ')}
          </div>
        </div>

        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="text-red-800 font-medium">Error:</div>
            <div className="text-red-700 mt-1">{error}</div>
          </div>
        )}
      </div>

      {/* Rankings */}
      {rankings.length > 0 && (
        <div className="card">
          <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
            <Trophy className="w-5 h-5 mr-2 text-yellow-600" />
            Final Rankings
          </h2>

          <div className="space-y-3">
            {rankings.map((ranking) => (
              <div
                key={ranking.agent.id}
                className={`flex items-center justify-between p-4 rounded-lg border-2 ${
                  ranking.agent.name === userAgentName
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200'
                }`}
              >
                <div className="flex items-center space-x-4">
                  <div className={`text-2xl ${get_rank_color(ranking.rank)} px-3 py-1 rounded-full font-bold`}>
                    {get_rank_icon(ranking.rank)}
                  </div>
                  <div>
                    <div className="font-semibold text-gray-900">{ranking.agent.name}</div>
                    <div className="text-sm text-gray-600">
                      Agent {ranking.agent.id} • Rank #{ranking.rank}
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-xl font-bold text-gray-900">
                    {ranking.score.toFixed(1)}
                  </div>
                  <div className="text-sm text-gray-600">Total Score</div>
                </div>
              </div>
            ))}
          </div>

          {rankings.find(r => r.agent.name === userAgentName) && (
            <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <div className="text-blue-800">
                🎉 Your agent <strong>{userAgentName}</strong> ranked #
                {rankings.find(r => r.agent.name === userAgentName)?.rank} out of {rankings.length}!
              </div>
            </div>
          )}
        </div>
      )}

      {/* How it Works */}
      <div className="card bg-gradient-to-r from-blue-50 to-purple-50 border-blue-200">
        <h2 className="text-xl font-semibold text-blue-900 mb-4">How It Works</h2>
        <div className="space-y-4 text-blue-800">
          <div className="flex items-start space-x-3">
            <span className="font-bold bg-blue-200 text-blue-900 px-2 py-1 rounded">1</span>
            <div>
              <div className="font-medium">Choose a scenario</div>
              <div className="text-sm">Each scenario has different optimal strategies and challenges.</div>
            </div>
          </div>
          <div className="flex items-start space-x-3">
            <span className="font-bold bg-blue-200 text-blue-900 px-2 py-1 rounded">2</span>
            <div>
              <div className="font-medium">Write your agent</div>
              <div className="text-sm">Create a JavaScript function that returns [house_index, mode] based on observations.</div>
            </div>
          </div>
          <div className="flex items-start space-x-3">
            <span className="font-bold bg-blue-200 text-blue-900 px-2 py-1 rounded">3</span>
            <div>
              <div className="font-medium">Run tournament</div>
              <div className="text-sm">Your agent competes against expert strategies in 20 simultaneous games.</div>
            </div>
          </div>
          <div className="flex items-start space-x-3">
            <span className="font-bold bg-blue-200 text-blue-900 px-2 py-1 rounded">4</span>
            <div>
              <div className="font-medium">See your ranking</div>
              <div className="text-sm">Get real-time results and compare against optimal agents.</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Tournament;
