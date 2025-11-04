import React from 'react';
import { Modal } from './Modal';

interface ScenarioParams {
  beta: number;
  kappa: number;
  team_reward_house_survives: number;
  team_penalty_house_burns: number;
  reward_own_house_survives: number;
  reward_other_house_survives: number;
  penalty_own_house_burns: number;
  penalty_other_house_burns: number;
  c: number;
  N_min: number;
  p_spark: number;
  num_agents: number;
}

interface ScenarioDetailModalProps {
  isOpen: boolean;
  onClose: () => void;
  scenarioName: string;
  scenarioDescription: string;
  params: ScenarioParams;
}

interface ParamInfo {
  label: string;
  description: string;
  format: (value: number) => string;
  unit?: string;
}

const PARAM_INFO: Record<keyof ScenarioParams, ParamInfo> = {
  beta: {
    label: 'Fire Spread Rate (β)',
    description: 'Probability that fire spreads from a burning house to each adjacent burning house per night. Higher values mean fires spread faster and more aggressively.',
    format: (v) => `${(v * 100).toFixed(0)}%`,
  },
  kappa: {
    label: 'Extinguish Efficiency (κ)',
    description: 'Probability that a working agent successfully extinguishes a fire. Higher values mean individual efforts are more effective.',
    format: (v) => `${(v * 100).toFixed(0)}%`,
  },
  p_spark: {
    label: 'Fire Ignition Probability',
    description: 'Probability that each safe house catches fire each night (both at game start and during play). Higher values create more fires and chaos.',
    format: (v) => `${(v * 100).toFixed(1)}%`,
  },
  team_reward_house_survives: {
    label: 'Team Reward (survives)',
    description: 'Team reward for each house that survives until the end of the game (collective outcome).',
    format: (v) => v.toString(),
    unit: 'points',
  },
  team_penalty_house_burns: {
    label: 'Team Penalty (burns)',
    description: 'Team penalty for each house that burns down completely (collective outcome).',
    format: (v) => v.toString(),
    unit: 'points',
  },
  reward_own_house_survives: {
    label: 'Own House Reward',
    description: 'Individual reward when your own house survives until the end. Creates self-interest incentive.',
    format: (v) => v.toString(),
    unit: 'points',
  },
  reward_other_house_survives: {
    label: 'Other House Reward',
    description: 'Individual reward when another agent\'s house survives. Encourages helping neighbors.',
    format: (v) => v.toString(),
    unit: 'points',
  },
  penalty_own_house_burns: {
    label: 'Own House Penalty',
    description: 'Individual penalty when your own house burns down. Punishes failure to protect property.',
    format: (v) => v.toString(),
    unit: 'points',
  },
  penalty_other_house_burns: {
    label: 'Other House Penalty',
    description: 'Individual penalty when another agent\'s house burns down. Discourages neglecting neighbors.',
    format: (v) => v.toString(),
    unit: 'points',
  },
  c: {
    label: 'Work Cost',
    description: 'Energy cost paid each night an agent chooses to work. Higher values make work more expensive relative to rest.',
    format: (v) => v.toString(),
    unit: 'points',
  },
  N_min: {
    label: 'Minimum Nights',
    description: 'Guaranteed minimum number of nights the game will last. Provides a baseline time horizon for planning.',
    format: (v) => v.toString(),
    unit: 'nights',
  },
  num_agents: {
    label: 'Number of Agents',
    description: 'Total number of agents in the town. There are always 10 houses arranged in a ring. Each house is owned by exactly one agent, with ownership assigned round-robin (agent i owns houses i, i+N, i+2N, etc.).',
    format: (v) => v.toString(),
    unit: 'agents',
  },
};

export const ScenarioDetailModal: React.FC<ScenarioDetailModalProps> = ({
  isOpen,
  onClose,
  scenarioName,
  scenarioDescription,
  params,
}) => {
  return (
    <Modal isOpen={isOpen} onClose={onClose} title={`Scenario: ${scenarioName}`}>
      <div className="space-y-6">
        {/* Description */}
        <div className="p-4 bg-orange-50 dark:bg-orange-950 rounded-lg border border-orange-200 dark:border-orange-800">
          <p className="text-gray-900 dark:text-gray-100">{scenarioDescription}</p>
        </div>

        {/* Parameters */}
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-3">
            Scenario Parameters
          </h3>
          <div className="space-y-3">
            {(Object.keys(params) as Array<keyof ScenarioParams>).map((paramKey) => {
              const paramInfo = PARAM_INFO[paramKey];
              const value = params[paramKey];

              return (
                <div
                  key={paramKey}
                  className="p-3 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700"
                >
                  <div className="flex justify-between items-start mb-1">
                    <div className="font-semibold text-gray-900 dark:text-gray-100">
                      {paramInfo.label}
                    </div>
                    <div className="text-lg font-bold text-orange-600 dark:text-orange-400">
                      {paramInfo.format(value)}
                      {paramInfo.unit && (
                        <span className="text-sm font-normal text-gray-600 dark:text-gray-400 ml-1">
                          {paramInfo.unit}
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">
                    {paramInfo.description}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Footer note */}
        <div className="p-3 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 text-sm text-gray-600 dark:text-gray-400">
          <strong>Note:</strong> These parameters define the game mechanics and incentive structure.
          They interact in complex ways to create different strategic challenges for agents.
        </div>
      </div>
    </Modal>
  );
};
