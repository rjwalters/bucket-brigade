import React from 'react';
import { Scenario, GameNight } from '../types';

interface GameSidebarProps {
  scenario: Scenario;
  currentNightData: GameNight | null;
  allNights: GameNight[];
  className?: string;
}

const GameSidebar: React.FC<GameSidebarProps> = ({
  scenario,
  currentNightData,
  allNights,
  className = ''
}) => {
  // Calculate summary statistics
  const totalNights = allNights.length;
  const finalNight = allNights[allNights.length - 1];

  if (finalNight) {
    const savedHouses = finalNight.houses.filter(h => h === 0).length;
    const ruinedHouses = finalNight.houses.filter(h => h === 2).length;
    const teamReward = (scenario.A * (savedHouses / 10)) - (scenario.L * (ruinedHouses / 10));

    // Calculate total individual rewards
    const totalRewards = allNights.reduce((acc, night) => {
      night.rewards.forEach((reward, agentId) => {
        acc[agentId] = (acc[agentId] || 0) + reward;
      });
      return acc;
    }, {} as Record<number, number>);

    return (
      <div className={`game-sidebar space-y-6 ${className}`}>
        {/* Scenario Parameters */}
        <div className="sidebar-section">
          <h3 className="sidebar-title">Scenario Parameters</h3>
          <div className="sidebar-content">
            <div className="parameter-grid">
              <div className="parameter-item">
                <span className="parameter-label">Fire Spread (β):</span>
                <span className="parameter-value font-mono">{scenario.beta.toFixed(3)}</span>
              </div>
              <div className="parameter-item">
                <span className="parameter-label">Extinguish (κ):</span>
                <span className="parameter-value font-mono">{scenario.kappa.toFixed(3)}</span>
              </div>
              <div className="parameter-item">
                <span className="parameter-label">Reward/A (saved):</span>
                <span className="parameter-value font-mono">{scenario.A}</span>
              </div>
              <div className="parameter-item">
                <span className="parameter-label">Penalty/L (ruined):</span>
                <span className="parameter-value font-mono">{scenario.L}</span>
              </div>
              <div className="parameter-item">
                <span className="parameter-label">Work Cost (c):</span>
                <span className="parameter-value font-mono">{scenario.c.toFixed(2)}</span>
              </div>
              <div className="parameter-item">
                <span className="parameter-label">Initial Burn %:</span>
                <span className="parameter-value font-mono">{(scenario.rho_ignite * 100).toFixed(0)}%</span>
              </div>
              <div className="parameter-item">
                <span className="parameter-label">Min Nights:</span>
                <span className="parameter-value font-mono">{scenario.N_min}</span>
              </div>
              <div className="parameter-item">
                <span className="parameter-label">Spark Prob:</span>
                <span className="parameter-value font-mono">{scenario.p_spark.toFixed(3)}</span>
              </div>
              <div className="parameter-item">
                <span className="parameter-label">Agents:</span>
                <span className="parameter-value font-mono">{scenario.num_agents}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Current Night Stats */}
        {currentNightData && (
          <div className="sidebar-section">
            <h3 className="sidebar-title">Night {currentNightData.night} Stats</h3>
            <div className="sidebar-content">
              <div className="stats-grid">
                <div className="stat-item">
                  <div className="stat-number text-green-600">
                    {currentNightData.houses.filter(h => h === 0).length}
                  </div>
                  <div className="stat-label">Safe</div>
                </div>
                <div className="stat-item">
                  <div className="stat-number text-red-600">
                    {currentNightData.houses.filter(h => h === 1).length}
                  </div>
                  <div className="stat-label">Burning</div>
                </div>
                <div className="stat-item">
                  <div className="stat-number text-gray-600">
                    {currentNightData.houses.filter(h => h === 2).length}
                  </div>
                  <div className="stat-label">Ruined</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Final Results */}
        {finalNight && (
          <div className="sidebar-section">
            <h3 className="sidebar-title">Final Results</h3>
            <div className="sidebar-content">
              <div className="final-stats">
                <div className="final-stat-item">
                  <span className="final-stat-label">Total Nights:</span>
                  <span className="final-stat-value">{totalNights}</span>
                </div>
                <div className="final-stat-item">
                  <span className="final-stat-label">Houses Saved:</span>
                  <span className="final-stat-value text-green-600">{savedHouses}</span>
                </div>
                <div className="final-stat-item">
                  <span className="final-stat-label">Houses Ruined:</span>
                  <span className="final-stat-value text-red-600">{ruinedHouses}</span>
                </div>
                <div className="final-stat-item border-t pt-2">
                  <span className="final-stat-label font-semibold">Team Reward:</span>
                  <span className={`final-stat-value font-bold ${
                    teamReward >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {teamReward >= 0 ? '+' : ''}{teamReward.toFixed(1)}
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Individual Agent Rewards */}
        {Object.keys(totalRewards).length > 0 && (
          <div className="sidebar-section">
            <h3 className="sidebar-title">Agent Performance</h3>
            <div className="sidebar-content">
              <div className="agent-rewards space-y-2">
                {Object.entries(totalRewards)
                  .sort(([,a], [,b]) => b - a)
                  .map(([agentId, reward]) => (
                    <div key={agentId} className="agent-reward-item flex justify-between items-center">
                      <span className="agent-id font-medium">Agent {agentId}:</span>
                      <span className={`agent-reward font-mono ${
                        reward >= 0 ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {reward >= 0 ? '+' : ''}{reward.toFixed(1)}
                      </span>
                    </div>
                  ))}
              </div>
            </div>
          </div>
        )}
      </div>
    );
  }

  return null;
};

export default GameSidebar;
