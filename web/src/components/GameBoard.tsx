import React from 'react';
import { HouseState } from '../types';

interface GameBoardProps {
  houses: HouseState[];
  agents: number[];
  night: number;
}

const GameBoard: React.FC<GameBoardProps> = ({ houses, agents, night }) => {
  // Create a 10-element ring layout
  const positions = Array.from({ length: 10 }, (_, i) => {
    const angle = (i / 10) * 2 * Math.PI - Math.PI / 2; // Start from top
    const radius = 120;
    const centerX = 200;
    const centerY = 200;

    return {
      x: centerX + radius * Math.cos(angle),
      y: centerY + radius * Math.sin(angle),
      index: i
    };
  });

  const getHouseClass = (state: HouseState) => {
    switch (state) {
      case 0: return 'house-safe'; // SAFE
      case 1: return 'house-burning'; // BURNING
      case 2: return 'house-ruined'; // RUINED
      default: return 'house-ruined';
    }
  };

  const getHouseSymbol = (state: HouseState) => {
    switch (state) {
      case 0: return '🏠'; // SAFE
      case 1: return '🔥'; // BURNING
      case 2: return '💀'; // RUINED
      default: return '❓';
    }
  };

  const getAgentAtHouse = (houseIndex: number) => {
    return agents.filter(agentId => agents[agentId] === houseIndex).length;
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-gray-900">Game Board</h2>
        <div className="text-sm text-gray-600">
          Night {night}
        </div>
      </div>

      <div className="relative">
        {/* SVG Game Board */}
        <svg width="400" height="400" className="border border-gray-200 rounded-lg bg-gray-50">
          {/* Connection lines between houses */}
          {positions.map((pos, i) => {
            const nextPos = positions[(i + 1) % 10];
            return (
              <line
                key={`line-${i}`}
                x1={pos.x}
                y1={pos.y}
                x2={nextPos.x}
                y2={nextPos.y}
                stroke="#e5e7eb"
                strokeWidth="2"
              />
            );
          })}

          {/* Houses */}
          {positions.map((pos, i) => {
            const agentCount = getAgentAtHouse(i);
            const houseState = houses[i];

            return (
              <g key={`house-${i}`}>
                {/* House circle */}
                <circle
                  cx={pos.x}
                  cy={pos.y}
                  r="25"
                  className={`${getHouseClass(houseState)} ${agentCount > 0 ? 'agent-present' : ''}`}
                  strokeWidth="3"
                />

                {/* House symbol */}
                <text
                  x={pos.x}
                  y={pos.y}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  className="text-lg pointer-events-none select-none"
                >
                  {getHouseSymbol(houseState)}
                </text>

                {/* Agent count */}
                {agentCount > 0 && (
                  <text
                    x={pos.x + 20}
                    y={pos.y - 20}
                    className="text-xs font-bold text-blue-600 bg-white rounded-full px-1 border border-blue-300"
                  >
                    {agentCount}
                  </text>
                )}

                {/* House index */}
                <text
                  x={pos.x}
                  y={pos.y + 35}
                  textAnchor="middle"
                  className="text-xs text-gray-500 fill-current"
                >
                  {i}
                </text>
              </g>
            );
          })}
        </svg>

        {/* Legend */}
        <div className="mt-4 flex flex-wrap gap-4 text-sm">
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 rounded-full house-safe"></div>
            <span>Safe (🏠)</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 rounded-full house-burning"></div>
            <span>Burning (🔥)</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 rounded-full house-ruined"></div>
            <span>Ruined (💀)</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 rounded-full bg-blue-200 border-2 border-blue-400"></div>
            <span>Agent Present</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default GameBoard;
