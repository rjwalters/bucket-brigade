import { memo } from 'react';
import { SCENARIOS, type ScenarioName } from '../../types/research';

export interface ScenarioSelectorProps {
  selected: ScenarioName;
  onChange: (scenario: ScenarioName) => void;
}

function ScenarioSelectorImpl({ selected, onChange }: ScenarioSelectorProps) {
  return (
    <div className="mb-8">
      <label className="block text-sm font-medium mb-2 text-content-primary">
        Select Scenario
      </label>
      <select
        value={selected}
        onChange={(e) => onChange(e.target.value as ScenarioName)}
        className="w-full md:w-auto px-4 py-2 border border-outline-primary rounded-lg bg-surface-secondary text-content-primary"
      >
        {SCENARIOS.map((scenario) => (
          <option key={scenario} value={scenario}>
            {scenario.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
          </option>
        ))}
      </select>
    </div>
  );
}

export const ScenarioSelector = memo(ScenarioSelectorImpl);
