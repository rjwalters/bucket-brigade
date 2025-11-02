/**
 * Agent Selector Modal Component
 *
 * Carousel modal for selecting an agent archetype
 */

import React, { useState } from 'react';
import type { AgentArchetype } from '../../types/teamBuilder';
import { getAllArchetypes } from '../../utils/agentArchetypes';
import { AgentCardLarge } from './AgentCard';
import { AgentStatsDisplay } from './AgentStatsDisplay';
import { AgentRadarChart } from './AgentRadarChart';
import { calculateRadarProfile } from '../../utils/agentRadarChart';

interface AgentSelectorModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelect: (archetype: AgentArchetype) => void;
  position: number;
  currentArchetype: AgentArchetype | null;
}

export function AgentSelectorModal({
  isOpen,
  onClose,
  onSelect,
  position,
  currentArchetype,
}: AgentSelectorModalProps) {
  const archetypes = getAllArchetypes();
  const initialIndex = currentArchetype
    ? archetypes.findIndex((a) => a.id === currentArchetype.id)
    : 0;

  const [selectedIndex, setSelectedIndex] = useState(initialIndex);
  const [viewMode, setViewMode] = useState<'radar' | 'bars'>('radar');
  const selectedArchetype = archetypes[selectedIndex];
  const radarProfile = calculateRadarProfile(selectedArchetype.parameters);

  const handlePrevious = () => {
    setSelectedIndex((prev) => (prev - 1 + archetypes.length) % archetypes.length);
  };

  const handleNext = () => {
    setSelectedIndex((prev) => (prev + 1) % archetypes.length);
  };

  const handleSelect = () => {
    onSelect(selectedArchetype);
    onClose();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowLeft') {
      handlePrevious();
    } else if (e.key === 'ArrowRight') {
      handleNext();
    } else if (e.key === 'Enter') {
      handleSelect();
    } else if (e.key === 'Escape') {
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black bg-opacity-50"
      onClick={onClose}
      onKeyDown={handleKeyDown}
    >
      <div
        className="bg-white dark:bg-gray-900 rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="sticky top-0 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700 p-6 z-10">
          <div className="flex justify-between items-center">
            <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-200">
              Select Agent for House {position}
            </h2>
            <button
              type="button"
              onClick={onClose}
              className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 text-2xl"
              aria-label="Close"
            >
              ×
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-6">
          {/* Carousel Navigation */}
          <div className="flex items-center justify-between mb-6">
            <button
              type="button"
              onClick={handlePrevious}
              className="px-4 py-2 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 rounded-lg font-medium transition-colors"
            >
              ← Previous
            </button>

            <div className="text-sm text-gray-600 dark:text-gray-400">
              {selectedIndex + 1} / {archetypes.length}
            </div>

            <button
              type="button"
              onClick={handleNext}
              className="px-4 py-2 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 rounded-lg font-medium transition-colors"
            >
              Next →
            </button>
          </div>

          {/* Agent Display */}
          <AgentCardLarge archetype={selectedArchetype} />

          {/* Description */}
          <div className="mt-6 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
              {selectedArchetype.description}
            </p>
          </div>

          {/* Stats - Toggle between Radar and Bars */}
          <div className="mt-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200">
                Behavioral Profile
              </h3>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setViewMode('radar')}
                  className={`px-3 py-1 text-sm font-medium rounded-lg transition-colors ${
                    viewMode === 'radar'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
                  }`}
                >
                  📊 Radar
                </button>
                <button
                  type="button"
                  onClick={() => setViewMode('bars')}
                  className={`px-3 py-1 text-sm font-medium rounded-lg transition-colors ${
                    viewMode === 'bars'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
                  }`}
                >
                  📈 Bars
                </button>
              </div>
            </div>

            {viewMode === 'radar' ? (
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <AgentRadarChart
                  archetype={selectedArchetype}
                  radarProfile={radarProfile}
                  size="medium"
                  showLabels={true}
                />
              </div>
            ) : (
              <AgentStatsDisplay archetype={selectedArchetype} />
            )}
          </div>

          {/* Strategy Notes */}
          <div className="mt-6">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-3">
              Strategy Profile
            </h3>
            <ul className="space-y-2">
              {selectedArchetype.strategyNotes.map((note, idx) => (
                <li
                  key={idx}
                  className="flex items-start gap-2 text-sm text-gray-700 dark:text-gray-300"
                >
                  <span className="text-lg" style={{ color: selectedArchetype.color }}>
                    •
                  </span>
                  <span>{note}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Actions */}
          <div className="mt-8 flex gap-3">
            <button
              type="button"
              onClick={handleSelect}
              className="flex-1 px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg transition-colors"
            >
              Select This Agent
            </button>
            <button
              type="button"
              onClick={() => {
                const randomIndex = Math.floor(Math.random() * archetypes.length);
                setSelectedIndex(randomIndex);
              }}
              className="px-6 py-3 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-800 dark:text-gray-200 font-semibold rounded-lg transition-colors"
            >
              🎲 Random
            </button>
          </div>

          {/* Quick Select Buttons */}
          <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">Quick Select:</p>
            <div className="flex flex-wrap gap-2">
              {archetypes.map((archetype, idx) => (
                <button
                  key={archetype.id}
                  type="button"
                  onClick={() => setSelectedIndex(idx)}
                  className={`
                    px-3 py-2 rounded-lg border-2 text-sm font-medium transition-all
                    ${
                      selectedIndex === idx
                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                        : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                    }
                  `}
                  style={{
                    borderColor: selectedIndex === idx ? archetype.color : undefined,
                  }}
                  title={archetype.name}
                >
                  {archetype.icon}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
