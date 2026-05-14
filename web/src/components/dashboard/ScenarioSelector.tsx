import React from 'react';
import { Shuffle } from 'lucide-react';
import type { ScenarioConfig } from '../../data/dashboardPresets';

interface ScenarioSelectorProps {
  scenarios: ScenarioConfig[];
  selectedId: string;
  onSelect: (id: string) => void;
  onRandom: () => void;
}

/**
 * Card listing scenario presets with a radio-style picker.
 *
 * Mirrors TeamSelector — accepts the scenario list, selected id, and
 * callbacks for selection and "pick a random preset".
 */
export const ScenarioSelector: React.FC<ScenarioSelectorProps> = React.memo(
  ({ scenarios, selectedId, onSelect, onRandom }) => {
    return (
      <div className="card flex flex-col h-[500px]">
        <div className="flex items-center justify-between mb-4 flex-shrink-0">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">Choose Scenario</h2>
          <button
            onClick={onRandom}
            className="btn-secondary text-sm flex items-center gap-2"
          >
            <Shuffle className="w-4 h-4" />
            Random
          </button>
        </div>

        <div className="space-y-3 flex-1 overflow-y-auto min-h-0 pr-2">
          {scenarios.map((scenario) => (
            <label
              key={scenario.id}
              className={`block p-4 rounded-lg border-2 cursor-pointer transition-all ${
                selectedId === scenario.id
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-950'
                  : 'border-gray-200 dark:border-gray-700 hover:border-blue-300 dark:hover:border-blue-600'
              }`}
            >
              <input
                type="radio"
                name="scenario"
                value={scenario.id}
                checked={selectedId === scenario.id}
                onChange={(e) => onSelect(e.target.value)}
                className="sr-only"
              />
              <div>
                <div className="font-medium text-gray-900 dark:text-gray-100">{scenario.name}</div>
                <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">{scenario.description}</div>
              </div>
            </label>
          ))}
        </div>
      </div>
    );
  }
);

ScenarioSelector.displayName = 'ScenarioSelector';
