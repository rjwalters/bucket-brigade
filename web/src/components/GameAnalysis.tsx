import React from 'react';
import { Trophy, TrendingUp, Flame, Users, Zap, AlertTriangle, CheckCircle } from 'lucide-react';
import type { GameReplay } from '../types';

interface GameAnalysisProps {
  game: GameReplay;
}

interface AgentStats {
  agentId: number;
  totalReward: number;
  workNights: number;
  restNights: number;
  contribution: number; // percentage of total work done
  efficiency: number; // reward per work night
}

const GameAnalysis: React.FC<GameAnalysisProps> = ({ game }) => {
  // Calculate final stats
  const finalNight = game.nights[game.nights.length - 1];
  const savedHouses = finalNight.houses.filter(h => h === 0).length;
  const ruinedHouses = finalNight.houses.filter(h => h === 2).length;
  const burningHouses = finalNight.houses.filter(h => h === 1).length;
  const totalNights = game.nights.length;

  // Calculate team performance
  const totalTeamReward = finalNight.rewards.reduce((sum, r) => sum + r, 0);
  const avgRewardPerAgent = totalTeamReward / game.scenario.num_agents;

  // Calculate agent statistics
  const agentStats: AgentStats[] = [];
  for (let agentId = 0; agentId < game.scenario.num_agents; agentId++) {
    let workNights = 0;
    let totalReward = 0;

    game.nights.forEach(night => {
      if (night.actions[agentId][1] === 1) { // WORK mode
        workNights++;
      }
      totalReward = night.rewards[agentId]; // Cumulative reward
    });

    const restNights = totalNights - workNights;
    const totalWorkNights = game.nights.reduce((sum, night) => {
      return sum + night.actions.filter(a => a[1] === 1).length;
    }, 0);
    const contribution = totalWorkNights > 0 ? (workNights / totalWorkNights) * 100 : 0;
    const efficiency = workNights > 0 ? totalReward / workNights : 0;

    agentStats.push({
      agentId,
      totalReward,
      workNights,
      restNights,
      contribution,
      efficiency
    });
  }

  // Sort agents by total reward
  const sortedAgents = [...agentStats].sort((a, b) => b.totalReward - a.totalReward);

  // Determine game outcome
  const gameSuccess = savedHouses >= 7; // Success if 70%+ houses saved
  const teamCooperation = agentStats.every(a => a.workNights >= totalNights * 0.3); // Everyone worked at least 30%

  // Key insights
  const insights: string[] = [];

  if (gameSuccess) {
    insights.push(`Successfully saved ${savedHouses} houses! Strong cooperation prevented disaster.`);
  } else if (savedHouses >= 5) {
    insights.push(`Saved ${savedHouses} houses - partial success. More coordination could have helped.`);
  } else {
    insights.push(`Only ${savedHouses} houses saved. Fire spread faster than the team could contain it.`);
  }

  if (teamCooperation) {
    insights.push(`All agents contributed meaningfully to firefighting efforts.`);
  } else {
    const freeriders = agentStats.filter(a => a.workNights < totalNights * 0.3);
    insights.push(`Agent${freeriders.length > 1 ? 's' : ''} ${freeriders.map(a => a.agentId).join(', ')} had low work participation.`);
  }

  const maxContributor = sortedAgents[0];
  const minContributor = sortedAgents[sortedAgents.length - 1];

  if (maxContributor.workNights > minContributor.workNights * 2) {
    insights.push(`Agent ${maxContributor.agentId} worked ${maxContributor.workNights} nights while Agent ${minContributor.agentId} only worked ${minContributor.workNights}.`);
  }

  if (game.scenario.p_spark > 0) {
    insights.push(`Persistent sparks (${(game.scenario.p_spark * 100).toFixed(0)}% chance) kept the team under pressure.`);
  }

  return (
    <div className="space-y-6 mt-8 pt-8 border-t-2 border-outline-primary">
      <div className="text-center">
        <h2 className="text-3xl font-bold text-content-primary mb-2">Game Analysis</h2>
        <p className="text-content-secondary">Performance breakdown and insights</p>
      </div>

      {/* Outcome Summary */}
      <div className={`card text-center ${gameSuccess ? 'bg-green-50 dark:bg-green-950 border-green-200 dark:border-green-800' : 'bg-red-50 dark:bg-red-950 border-red-200 dark:border-red-800'}`}>
        <div className="flex items-center justify-center mb-4">
          {gameSuccess ? (
            <CheckCircle className="w-12 h-12 text-green-600 dark:text-green-400" />
          ) : (
            <AlertTriangle className="w-12 h-12 text-red-600 dark:text-red-400" />
          )}
        </div>
        <h3 className={`text-2xl font-bold mb-2 ${gameSuccess ? 'text-green-900 dark:text-green-100' : 'text-red-900 dark:text-red-100'}`}>
          {gameSuccess ? 'Mission Success!' : 'Mission Failed'}
        </h3>
        <p className={`text-lg ${gameSuccess ? 'text-green-700 dark:text-green-300' : 'text-red-700 dark:text-red-300'}`}>
          {savedHouses} houses saved, {ruinedHouses} ruined, {burningHouses} still burning
        </p>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card bg-blue-50 dark:bg-blue-950 border-blue-200 dark:border-blue-800">
          <div className="flex items-center justify-between mb-2">
            <Trophy className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            <span className="text-2xl font-bold text-blue-900 dark:text-blue-100">{totalTeamReward.toFixed(1)}</span>
          </div>
          <p className="text-sm text-blue-700 dark:text-blue-300">Total Team Reward</p>
        </div>

        <div className="card bg-purple-50 dark:bg-purple-950 border-purple-200 dark:border-purple-800">
          <div className="flex items-center justify-between mb-2">
            <TrendingUp className="w-6 h-6 text-purple-600 dark:text-purple-400" />
            <span className="text-2xl font-bold text-purple-900 dark:text-purple-100">{avgRewardPerAgent.toFixed(1)}</span>
          </div>
          <p className="text-sm text-purple-700 dark:text-purple-300">Avg Per Agent</p>
        </div>

        <div className="card bg-orange-50 dark:bg-orange-950 border-orange-200 dark:border-orange-800">
          <div className="flex items-center justify-between mb-2">
            <Flame className="w-6 h-6 text-orange-600 dark:text-orange-400" />
            <span className="text-2xl font-bold text-orange-900 dark:text-orange-100">{totalNights}</span>
          </div>
          <p className="text-sm text-orange-700 dark:text-orange-300">Nights Survived</p>
        </div>

        <div className="card bg-green-50 dark:bg-green-950 border-green-200 dark:border-green-800">
          <div className="flex items-center justify-between mb-2">
            <Users className="w-6 h-6 text-green-600 dark:text-green-400" />
            <span className="text-2xl font-bold text-green-900 dark:text-green-100">{game.scenario.num_agents}</span>
          </div>
          <p className="text-sm text-green-700 dark:text-green-300">Team Size</p>
        </div>
      </div>

      {/* Agent Performance */}
      <div className="card">
        <h3 className="text-xl font-semibold text-content-primary mb-4 flex items-center">
          <Users className="w-5 h-5 mr-2 text-blue-600 dark:text-blue-400" />
          Individual Contributions
        </h3>

        <div className="space-y-4">
          {sortedAgents.map((agent, idx) => (
            <div key={agent.agentId} className="border border-outline-primary rounded-lg p-4 bg-surface-secondary">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center text-white font-bold ${
                    idx === 0 ? 'bg-yellow-500' :
                    idx === 1 ? 'bg-gray-400' :
                    idx === 2 ? 'bg-orange-600' :
                    'bg-gray-600'
                  }`}>
                    {agent.agentId}
                  </div>
                  <div>
                    <div className="font-semibold text-content-primary">Agent {agent.agentId}</div>
                    <div className="text-sm text-content-secondary">Rank #{idx + 1}</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-2xl font-bold text-content-primary">{agent.totalReward.toFixed(1)}</div>
                  <div className="text-sm text-content-secondary">Total Reward</div>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>
                  <div className="text-content-secondary mb-1">Work Nights</div>
                  <div className="font-semibold text-content-primary flex items-center">
                    <Zap className="w-4 h-4 mr-1 text-yellow-600 dark:text-yellow-400" />
                    {agent.workNights} / {totalNights}
                  </div>
                </div>
                <div>
                  <div className="text-content-secondary mb-1">Contribution</div>
                  <div className="font-semibold text-content-primary">
                    {agent.contribution.toFixed(1)}%
                  </div>
                </div>
                <div>
                  <div className="text-content-secondary mb-1">Efficiency</div>
                  <div className="font-semibold text-content-primary">
                    {agent.efficiency.toFixed(1)} per work
                  </div>
                </div>
              </div>

              {/* Work participation bar */}
              <div className="mt-3">
                <div className="h-2 bg-surface-tertiary rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-500 transition-all"
                    style={{ width: `${(agent.workNights / totalNights) * 100}%` }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Key Insights */}
      <div className="card bg-blue-50 dark:bg-blue-950 border-blue-200 dark:border-blue-800">
        <h3 className="text-xl font-semibold text-content-primary mb-4 flex items-center">
          <Zap className="w-5 h-5 mr-2 text-blue-600 dark:text-blue-400" />
          Key Insights
        </h3>
        <ul className="space-y-2">
          {insights.map((insight, idx) => (
            <li key={idx} className="flex items-start text-blue-700 dark:text-blue-300">
              <span className="mr-2 mt-1">•</span>
              <span>{insight}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Scenario Parameters */}
      <div className="card bg-surface-tertiary">
        <h3 className="text-xl font-semibold text-content-primary mb-4">Scenario Configuration</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
          <div>
            <div className="text-content-secondary">Fire Spread</div>
            <div className="font-semibold text-content-primary">{(game.scenario.beta * 100).toFixed(0)}%</div>
          </div>
          <div>
            <div className="text-content-secondary">Extinguish Efficiency</div>
            <div className="font-semibold text-content-primary">{(game.scenario.kappa * 100).toFixed(0)}%</div>
          </div>
          <div>
            <div className="text-content-secondary">Initial Fires</div>
          </div>
          <div>
            <div className="text-content-secondary">Work Cost</div>
            <div className="font-semibold text-content-primary">{game.scenario.c.toFixed(1)} per night</div>
          </div>
          <div>
            <div className="text-content-secondary">Saved House Value</div>
            <div className="font-semibold text-content-primary">{game.scenario.A.toFixed(0)}</div>
          </div>
          <div>
            <div className="text-content-secondary">Ruined House Penalty</div>
            <div className="font-semibold text-content-primary">-{game.scenario.L.toFixed(0)}</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default GameAnalysis;
