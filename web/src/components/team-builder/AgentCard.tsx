/**
 * Agent Card Component
 *
 * Displays an agent in a team position with click to select
 */

import type { AgentArchetype } from '../../types/teamBuilder';

interface AgentCardProps {
  position: number;
  archetype: AgentArchetype | null;
  onClick: () => void;
  isSelected?: boolean;
}

export function AgentCard({ position, archetype, onClick, isSelected = false }: AgentCardProps) {
  if (!archetype) {
    return (
      <button
        type="button"
        onClick={onClick}
        className={`
          relative w-24 h-24 rounded-lg border-2 border-dashed
          border-gray-300 dark:border-gray-600
          hover:border-gray-400 dark:hover:border-gray-500
          hover:bg-gray-50 dark:hover:bg-gray-800
          transition-all duration-200
          flex flex-col items-center justify-center
          cursor-pointer
          ${isSelected ? 'ring-2 ring-blue-500' : ''}
        `}
        aria-label={`Select agent for House ${position}`}
      >
        <span className="text-3xl text-gray-400 dark:text-gray-600">❓</span>
        <span className="text-xs text-gray-500 dark:text-gray-400 mt-1">Empty</span>
        <span className="text-xs text-gray-400 dark:text-gray-500">House {position}</span>
      </button>
    );
  }

  return (
    <button
      type="button"
      onClick={onClick}
      className={`
        relative w-24 h-24 rounded-lg border-2
        hover:scale-105 hover:shadow-lg
        transition-all duration-200
        flex flex-col items-center justify-center
        cursor-pointer
        ${isSelected ? 'ring-2 ring-blue-500 scale-105' : ''}
      `}
      style={{
        borderColor: archetype.color,
        backgroundColor: `${archetype.color}10`,
      }}
      aria-label={`Change agent for House ${position}: ${archetype.name}`}
    >
      <span className="text-3xl mb-1">{archetype.icon}</span>
      <span className="text-xs font-medium text-gray-700 dark:text-gray-300 text-center px-1">
        {archetype.name}
      </span>
      <span className="text-xs text-gray-500 dark:text-gray-400">House {position}</span>

      {/* Hover tooltip */}
      <div className="absolute -top-2 -right-2 opacity-0 group-hover:opacity-100 transition-opacity">
        <div className="w-5 h-5 rounded-full bg-blue-500 flex items-center justify-center text-white text-xs">
          ✏️
        </div>
      </div>
    </button>
  );
}

/**
 * Large version for modal display
 */
export function AgentCardLarge({ archetype }: { archetype: AgentArchetype }) {
  return (
    <div
      className="relative p-6 rounded-xl border-4 shadow-lg"
      style={{
        borderColor: archetype.color,
        backgroundColor: `${archetype.color}05`,
      }}
    >
      <div className="text-center">
        <div className="text-6xl mb-4">{archetype.icon}</div>
        <h3 className="text-2xl font-bold text-gray-800 dark:text-gray-200 mb-2">
          {archetype.name}
        </h3>
        <p
          className="text-sm italic font-medium mb-4"
          style={{ color: archetype.color }}
        >
          "{archetype.tagline}"
        </p>
      </div>
    </div>
  );
}
