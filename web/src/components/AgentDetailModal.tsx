import React from 'react';
import { Modal } from './Modal';
import { AgentRadarChart } from './AgentRadarChart';
import { getArchetype, PARAMETER_DESCRIPTIONS } from '../data/archetypes';
import type { ArchetypeParams } from '../data/archetypes';

interface AgentDetailModalProps {
  isOpen: boolean;
  onClose: () => void;
  archetypeId: string;
}

export const AgentDetailModal: React.FC<AgentDetailModalProps> = ({ isOpen, onClose, archetypeId }) => {
  const archetype = getArchetype(archetypeId);

  if (!archetype) {
    return null;
  }

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={archetype.name}>
      <div className="space-y-6">
        {/* Description */}
        <div className="p-4 bg-blue-50 dark:bg-blue-950 rounded-lg border border-blue-200 dark:border-blue-800">
          <p className="text-gray-900 dark:text-gray-100">{archetype.description}</p>
        </div>

        {/* Radar Chart */}
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-3">
            Behavioral Profile
          </h3>
          <AgentRadarChart params={archetype.params} />
        </div>

        {/* Full Parameter List */}
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-3">
            Parameter Details
          </h3>
          <div className="space-y-3">
            {(Object.keys(archetype.params) as Array<keyof ArchetypeParams>).map((paramKey) => {
              const paramInfo = PARAMETER_DESCRIPTIONS[paramKey];
              const value = archetype.params[paramKey];

              return (
                <div
                  key={paramKey}
                  className="p-3 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700"
                >
                  <div className="flex justify-between items-start mb-1">
                    <div className="font-semibold text-gray-900 dark:text-gray-100">
                      {paramInfo.label}
                    </div>
                    <div className="text-lg font-bold text-blue-600 dark:text-blue-400">
                      {value.toFixed(1)}
                    </div>
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">
                    {paramInfo.description}
                  </div>
                  {/* Visual bar */}
                  <div className="mt-2 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-blue-500 dark:bg-blue-600 rounded-full"
                      style={{ width: `${value * 100}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </Modal>
  );
};
