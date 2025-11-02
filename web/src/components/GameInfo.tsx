import React from 'react';
import { Scenario, GameNight } from '../types';

interface GameInfoProps {
  scenario: Scenario;
  nightData: GameNight;
  currentNight: number;
}

const GameInfo: React.FC<GameInfoProps> = ({ scenario, nightData, currentNight }) => {
  const savedHouses = nightData.houses.filter(h => h === 0).length;
  const burningHouses = nightData.houses.filter(h => h === 1).length;
  const ruinedHouses = nightData.houses.filter(h => h === 2).length;

  return (
    <div className="space-y-4">
      {/* Scenario Parameters */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-3">Scenario Parameters</h3>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-600">Fire Spread (β):</span>
            <span className="font-mono">{scenario.beta.toFixed(3)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Extinguish (κ):</span>
            <span className="font-mono">{scenario.kappa.toFixed(3)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Reward per Saved (A):</span>
            <span className="font-mono">{scenario.A}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Penalty per Ruined (L):</span>
            <span className="font-mono">{scenario.L}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Work Cost (c):</span>
            <span className="font-mono">{scenario.c.toFixed(2)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Min Nights:</span>
            <span className="font-mono">{scenario.N_min}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Agents:</span>
            <span className="font-mono">{scenario.num_agents}</span>
          </div>
        </div>
      </div>

      {/* Current Night Stats */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-3">Night {currentNight} Stats</h3>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between items-center">
            <span className="text-gray-600">Safe Houses:</span>
            <div className="flex items-center space-x-2">
              <span className="font-mono">{savedHouses}</span>
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            </div>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-600">Burning Houses:</span>
            <div className="flex items-center space-x-2">
              <span className="font-mono">{burningHouses}</span>
              <div className="w-3 h-3 bg-red-500 rounded-full"></div>
            </div>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-600">Ruined Houses:</span>
            <div className="flex items-center space-x-2">
              <span className="font-mono">{ruinedHouses}</span>
              <div className="w-3 h-3 bg-gray-500 rounded-full"></div>
            </div>
          </div>
        </div>
      </div>

      {/* Agent Actions & Rewards */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-3">Agent Activity</h3>
        <div className="space-y-2 max-h-48 overflow-y-auto">
          {nightData.signals.map((signal, agentId) => {
            const action = nightData.actions[agentId];
            const reward = nightData.rewards[agentId];
            const location = nightData.locations[agentId];

            return (
              <div key={agentId} className="flex items-center justify-between text-sm p-2 bg-gray-50 rounded">
                <div className="flex items-center space-x-2">
                  <span className="font-medium text-gray-900">Agent {agentId}</span>
                  <span className="text-xs text-gray-500">at house {location}</span>
                </div>
                <div className="flex items-center space-x-3">
                  <span className={`px-2 py-1 rounded text-xs ${
                    signal === 1 ? 'bg-blue-100 text-blue-800' : 'bg-gray-100 text-gray-800'
                  }`}>
                    {signal === 1 ? 'WORK' : 'REST'}
                  </span>
                  <span className={`font-mono text-xs ${
                    reward >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {reward >= 0 ? '+' : ''}{reward.toFixed(1)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Team Reward */}
      <div className="card bg-gradient-to-r from-blue-50 to-purple-50 border-blue-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-2">Team Reward</h3>
        <div className="text-2xl font-bold text-blue-600">
          {(
            scenario.A * (savedHouses / 10) -
            scenario.L * (ruinedHouses / 10)
          ).toFixed(1)}
        </div>
        <p className="text-sm text-gray-600 mt-1">
          {(scenario.A * (savedHouses / 10)).toFixed(1)} saved - {(scenario.L * (ruinedHouses / 10)).toFixed(1)} ruined
        </p>
      </div>
    </div>
  );
};

export default GameInfo;
