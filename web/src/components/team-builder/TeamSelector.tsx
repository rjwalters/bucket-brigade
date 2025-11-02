/**
 * Team Selector Component
 *
 * Circular layout displaying all team positions
 */

import type { TeamComposition } from '../../types/teamBuilder';
import { AgentCard } from './AgentCard';

interface TeamSelectorProps {
  team: TeamComposition;
  onPositionClick: (position: number) => void;
  selectedPosition: number | null;
}

export function TeamSelector({ team, onPositionClick, selectedPosition }: TeamSelectorProps) {
  const teamSize = team.positions.filter((p) => p != null).length;

  // Calculate positions for circular layout
  const getPositionStyle = (index: number, total: number) => {
    const angle = (index * 360) / total - 90; // Start from top
    const radius = 180; // Distance from center
    const x = Math.cos((angle * Math.PI) / 180) * radius;
    const y = Math.sin((angle * Math.PI) / 180) * radius;

    return {
      left: `calc(50% + ${x}px)`,
      top: `calc(50% + ${y}px)`,
      transform: 'translate(-50%, -50%)',
    };
  };

  return (
    <div className="relative w-full aspect-square max-w-2xl mx-auto">
      {/* Center circle with town visual */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="w-32 h-32 rounded-full bg-gradient-to-br from-orange-400 to-red-500 flex items-center justify-center shadow-lg">
          <div className="text-center">
            <div className="text-4xl mb-1">🔥</div>
            <div className="text-xs font-bold text-white">TOWN</div>
            <div className="text-xs text-white opacity-90">{teamSize}/10</div>
          </div>
        </div>
      </div>

      {/* Agent positions in circle */}
      {team.positions.map((archetype, index) => (
        <div
          key={index}
          className="absolute"
          style={getPositionStyle(index, team.positions.length)}
        >
          <AgentCard
            position={index}
            archetype={archetype}
            onClick={() => onPositionClick(index)}
            isSelected={selectedPosition === index}
          />
        </div>
      ))}

      {/* Connection lines (optional, for visual effect) */}
      <svg
        className="absolute inset-0 pointer-events-none"
        style={{ width: '100%', height: '100%' }}
      >
        <defs>
          <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#f97316" stopOpacity="0.2" />
            <stop offset="100%" stopColor="#ef4444" stopOpacity="0.2" />
          </linearGradient>
        </defs>
        {team.positions.map((_, index) => {
          const nextIndex = (index + 1) % team.positions.length;
          const angle1 = (index * 360) / team.positions.length - 90;
          const angle2 = (nextIndex * 360) / team.positions.length - 90;
          const radius = 180;

          const x1 = 50 + (Math.cos((angle1 * Math.PI) / 180) * radius * 100) / 400;
          const y1 = 50 + (Math.sin((angle1 * Math.PI) / 180) * radius * 100) / 400;
          const x2 = 50 + (Math.cos((angle2 * Math.PI) / 180) * radius * 100) / 400;
          const y2 = 50 + (Math.sin((angle2 * Math.PI) / 180) * radius * 100) / 400;

          return (
            <line
              key={`line-${index}`}
              x1={`${x1}%`}
              y1={`${y1}%`}
              x2={`${x2}%`}
              y2={`${y2}%`}
              stroke="url(#lineGradient)"
              strokeWidth="2"
              strokeDasharray="5,5"
            />
          );
        })}
      </svg>
    </div>
  );
}

/**
 * Compact grid layout as alternative
 */
export function TeamGridView({ team, onPositionClick, selectedPosition }: TeamSelectorProps) {
  return (
    <div className="grid grid-cols-5 gap-4 max-w-3xl mx-auto">
      {team.positions.map((archetype, index) => (
        <AgentCard
          key={index}
          position={index}
          archetype={archetype}
          onClick={() => onPositionClick(index)}
          isSelected={selectedPosition === index}
        />
      ))}
    </div>
  );
}
