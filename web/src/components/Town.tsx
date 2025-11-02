import React from 'react';
import { HouseState } from '../types';

interface TownProps {
  houses: HouseState[];
  className?: string;
}

const Town: React.FC<TownProps> = ({ houses, className = '' }) => {
  // Create positions for 10 houses in a circle
  const housePositions = Array.from({ length: 10 }, (_, i) => {
    const angle = (i / 10) * 2 * Math.PI - Math.PI / 2; // Start from top
    const radius = 120; // Distance from center
    const centerX = 150; // Center of the circle
    const centerY = 150;

    return {
      x: centerX + radius * Math.cos(angle),
      y: centerY + radius * Math.sin(angle),
      index: i,
      angle: angle
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

  return (
    <div className={`town-visualization ${className}`}>
      <svg width="300" height="300" className="town-svg">
        {/* Connection lines between houses */}
        {housePositions.map((pos, i) => {
          const nextPos = housePositions[(i + 1) % 10];
          return (
            <line
              key={`connection-${i}`}
              x1={pos.x}
              y1={pos.y}
              x2={nextPos.x}
              y2={nextPos.y}
              stroke="#e5e7eb"
              strokeWidth="2"
              className="town-connection"
            />
          );
        })}

        {/* Houses */}
        {housePositions.map((pos, i) => {
          const houseState = houses[i];
          const symbol = getHouseSymbol(houseState);

          return (
            <g key={`house-${i}`} className="town-house-group">
              {/* House circle */}
              <circle
                cx={pos.x}
                cy={pos.y}
                r="25"
                className={`town-house ${getHouseClass(houseState)}`}
                data-house-index={i}
                data-house-state={houseState}
              />

              {/* House symbol */}
              <text
                x={pos.x}
                y={pos.y}
                textAnchor="middle"
                dominantBaseline="middle"
                className="town-house-symbol select-none"
              >
                {symbol}
              </text>

              {/* House index label */}
              <text
                x={pos.x}
                y={pos.y + 35}
                textAnchor="middle"
                className="town-house-index text-xs text-gray-500 fill-current"
              >
                {i}
              </text>
            </g>
          );
        })}
      </svg>

      {/* Legend */}
      <div className="town-legend mt-4 flex flex-wrap gap-3 justify-center text-sm">
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 rounded-full house-safe border-2 border-green-300"></div>
          <span>Safe (🏠)</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 rounded-full house-burning border-2 border-red-300 animate-pulse"></div>
          <span>Burning (🔥)</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 rounded-full house-ruined border-2 border-gray-300"></div>
          <span>Ruined (💀)</span>
        </div>
      </div>
    </div>
  );
};

export default Town;
