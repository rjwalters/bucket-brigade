import React, { useState } from 'react';
import { Dices } from 'lucide-react';
import { AgentRadarChart } from './AgentRadarChart';
import { getArchetype, PARAMETER_DESCRIPTIONS } from '../data/archetypes';
import type { ArchetypeParams } from '../data/archetypes';

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

interface GameParametersProps {
  teamArchetypes: string[];
  scenarioParams: ScenarioParams;
  scenarioName: string;
  isRandomTeam: boolean;
  isRandomScenario: boolean;
  onRegenerateTeam: () => void;
  onRegenerateScenario: () => void;
}

interface ParamInfo {
  label: string;
  description: string;
  format: (value: number) => string;
  unit?: string;
}

const SCENARIO_PARAM_INFO: Record<keyof ScenarioParams, ParamInfo> = {
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
    description: 'Reward each agent receives for each house that survives. Creates collective incentive since all agents receive this independently (public goods).',
    format: (v) => v.toString(),
    unit: 'points',
  },
  team_penalty_house_burns: {
    label: 'Team Penalty (burns)',
    description: 'Penalty each agent receives for each house that burns down. All agents receive this penalty independently (public cost).',
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
    description: 'Minimum number of nights before the game can end. After reaching this threshold, the game ends when all houses are either safe (no fires) or completely burned down. Prevents games from ending too quickly and ensures meaningful gameplay.',
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

type DetailType =
  | { type: 'agent'; archetype: string }
  | { type: 'scenario'; param: keyof ScenarioParams }
  | null;

export const GameParameters: React.FC<GameParametersProps> = ({
  teamArchetypes,
  scenarioParams,
  scenarioName,
  isRandomTeam,
  isRandomScenario,
  onRegenerateTeam,
  onRegenerateScenario,
}) => {
  const [selectedDetail, setSelectedDetail] = useState<DetailType>(null);

  const handleAgentClick = (archetype: string) => {
    setSelectedDetail({ type: 'agent', archetype });
  };

  const handleParamClick = (param: keyof ScenarioParams) => {
    setSelectedDetail({ type: 'scenario', param });
  };

  const archetype = selectedDetail?.type === 'agent'
    ? getArchetype(selectedDetail.archetype)
    : null;

  return (
    <div className="card col-span-2">
      <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-4">
        Game Parameters
      </h2>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column - Team & Scenario */}
        <div className="space-y-6">
          {/* Team Composition */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Team Composition
                <span className="ml-2 text-xs text-gray-500 dark:text-gray-400">(click for details)</span>
              </h3>
              {isRandomTeam && (
                <button
                  onClick={onRegenerateTeam}
                  className="flex items-center gap-1 px-2 py-1 text-xs text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-950 rounded transition-colors"
                  title="Generate new random agents"
                >
                  <Dices className="w-3 h-3" />
                  Re-roll
                </button>
              )}
            </div>
            <div className="flex flex-wrap gap-2">
              {teamArchetypes.map((archetype, idx) => (
                <button
                  key={idx}
                  onClick={() => handleAgentClick(archetype)}
                  className={`px-3 py-2 border-2 rounded text-sm font-medium transition-all ${
                    selectedDetail?.type === 'agent' && selectedDetail.archetype === archetype
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-950 text-blue-700 dark:text-blue-300'
                      : 'border-gray-200 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:border-blue-300 dark:hover:border-blue-600'
                  }`}
                  title={`Agent ${idx + 1}: ${archetype}`}
                >
                  {archetype}
                </button>
              ))}
            </div>
          </div>

          {/* Scenario Parameters */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Scenario: {scenarioName}
                <span className="ml-2 text-xs text-gray-500 dark:text-gray-400">(click for details)</span>
              </h3>
              {isRandomScenario && (
                <button
                  onClick={onRegenerateScenario}
                  className="flex items-center gap-1 px-2 py-1 text-xs text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-950 rounded transition-colors"
                  title="Generate new random parameters"
                >
                  <Dices className="w-3 h-3" />
                  Re-roll
                </button>
              )}
            </div>

            <div className="space-y-3">
              {/* Fire Dynamics */}
              <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-2">
                <div className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-1.5 px-1">Fire Dynamics</div>
                <div className="space-y-1">
                  {(['beta', 'kappa', 'p_spark'] as Array<keyof ScenarioParams>).map((paramKey) => {
                    const paramInfo = SCENARIO_PARAM_INFO[paramKey];
                    const value = scenarioParams[paramKey];
                    const isSelected = selectedDetail?.type === 'scenario' && selectedDetail.param === paramKey;

                    return (
                      <button
                        key={paramKey}
                        onClick={() => handleParamClick(paramKey)}
                        className={`w-full p-2 rounded text-xs text-left transition-all border-2 flex items-center justify-between ${
                          isSelected
                            ? 'border-orange-500 bg-orange-50 dark:bg-orange-950'
                            : 'border-transparent hover:border-orange-300 dark:hover:border-orange-600 hover:bg-orange-50 dark:hover:bg-orange-950'
                        }`}
                      >
                        <div className="text-gray-700 dark:text-gray-300 font-medium">{paramInfo.label}</div>
                        <div className="font-semibold text-gray-900 dark:text-gray-100">
                          {paramInfo.format(value)}
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Team Rewards */}
              <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-2">
                <div className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-1.5 px-1">Team Rewards (Collective)</div>
                <div className="space-y-1">
                  {(['team_reward_house_survives', 'team_penalty_house_burns'] as Array<keyof ScenarioParams>).map((paramKey) => {
                    const paramInfo = SCENARIO_PARAM_INFO[paramKey];
                    const value = scenarioParams[paramKey];
                    const isSelected = selectedDetail?.type === 'scenario' && selectedDetail.param === paramKey;

                    return (
                      <button
                        key={paramKey}
                        onClick={() => handleParamClick(paramKey)}
                        className={`w-full p-2 rounded text-xs text-left transition-all border-2 flex items-center justify-between ${
                          isSelected
                            ? 'border-orange-500 bg-orange-50 dark:bg-orange-950'
                            : 'border-transparent hover:border-orange-300 dark:hover:border-orange-600 hover:bg-orange-50 dark:hover:bg-orange-950'
                        }`}
                      >
                        <div className="text-gray-700 dark:text-gray-300 font-medium">{paramInfo.label}</div>
                        <div className="font-semibold text-gray-900 dark:text-gray-100">
                          {paramInfo.format(value)}
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Individual Incentives */}
              <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-2">
                <div className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-1.5 px-1">Individual Incentives (Per-Agent)</div>
                <div className="space-y-1">
                  {(['reward_own_house_survives', 'reward_other_house_survives', 'penalty_own_house_burns', 'penalty_other_house_burns', 'c'] as Array<keyof ScenarioParams>).map((paramKey) => {
                    const paramInfo = SCENARIO_PARAM_INFO[paramKey];
                    const value = scenarioParams[paramKey];
                    const isSelected = selectedDetail?.type === 'scenario' && selectedDetail.param === paramKey;

                    return (
                      <button
                        key={paramKey}
                        onClick={() => handleParamClick(paramKey)}
                        className={`w-full p-2 rounded text-xs text-left transition-all border-2 flex items-center justify-between ${
                          isSelected
                            ? 'border-orange-500 bg-orange-50 dark:bg-orange-950'
                            : 'border-transparent hover:border-orange-300 dark:hover:border-orange-600 hover:bg-orange-50 dark:hover:bg-orange-950'
                        }`}
                      >
                        <div className="text-gray-700 dark:text-gray-300 font-medium">{paramInfo.label}</div>
                        <div className="font-semibold text-gray-900 dark:text-gray-100">
                          {paramInfo.format(value)}
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Game Structure */}
              <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-2">
                <div className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-1.5 px-1">Game Structure</div>
                <div className="space-y-1">
                  {(['N_min'] as Array<keyof ScenarioParams>).map((paramKey) => {
                    const paramInfo = SCENARIO_PARAM_INFO[paramKey];
                    const value = scenarioParams[paramKey];
                    const isSelected = selectedDetail?.type === 'scenario' && selectedDetail.param === paramKey;

                    return (
                      <button
                        key={paramKey}
                        onClick={() => handleParamClick(paramKey)}
                        className={`w-full p-2 rounded text-xs text-left transition-all border-2 flex items-center justify-between ${
                          isSelected
                            ? 'border-orange-500 bg-orange-50 dark:bg-orange-950'
                            : 'border-transparent hover:border-orange-300 dark:hover:border-orange-600 hover:bg-orange-50 dark:hover:bg-orange-950'
                        }`}
                      >
                        <div className="text-gray-700 dark:text-gray-300 font-medium">{paramInfo.label}</div>
                        <div className="font-semibold text-gray-900 dark:text-gray-100">
                          {paramInfo.format(value)}
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column - Details */}
        <div className="min-h-[400px]">
          {selectedDetail ? (
            <div className="sticky top-4 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700">
              {selectedDetail.type === 'agent' && archetype ? (
                <div className="space-y-4">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
                      {archetype.name}
                    </h3>
                    <p className="text-sm text-gray-700 dark:text-gray-300">
                      {archetype.description}
                    </p>
                  </div>

                  {/* Radar Chart */}
                  <div>
                    <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-2">
                      Behavioral Profile
                    </h4>
                    <AgentRadarChart params={archetype.params} className="mt-2" />
                  </div>

                  {/* Parameter List */}
                  <div>
                    <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-2">
                      Parameters
                    </h4>
                    <div className="space-y-2">
                      {(Object.keys(archetype.params) as Array<keyof ArchetypeParams>).map((paramKey) => {
                        const paramInfo = PARAMETER_DESCRIPTIONS[paramKey];
                        const value = archetype.params[paramKey];

                        return (
                          <div key={paramKey} className="text-xs">
                            <div className="flex justify-between items-center mb-1">
                              <span className="text-gray-700 dark:text-gray-300 font-medium">
                                {paramInfo.label}
                              </span>
                              <span className="text-blue-600 dark:text-blue-400 font-bold">
                                {value.toFixed(1)}
                              </span>
                            </div>
                            <div className="h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
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
              ) : selectedDetail.type === 'scenario' ? (
                <div>
                  <div className="mb-3">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
                      {SCENARIO_PARAM_INFO[selectedDetail.param].label}
                    </h3>
                    <div className="flex items-baseline gap-2">
                      <span className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                        {SCENARIO_PARAM_INFO[selectedDetail.param].format(scenarioParams[selectedDetail.param])}
                      </span>
                      {SCENARIO_PARAM_INFO[selectedDetail.param].unit && (
                        <span className="text-sm text-gray-600 dark:text-gray-400">
                          {SCENARIO_PARAM_INFO[selectedDetail.param].unit}
                        </span>
                      )}
                    </div>
                  </div>
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    {SCENARIO_PARAM_INFO[selectedDetail.param].description}
                  </p>
                </div>
              ) : null}
            </div>
          ) : (
            <div className="sticky top-4 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 flex items-center justify-center h-full min-h-[400px]">
              <p className="text-center text-gray-500 dark:text-gray-400">
                Select an agent or parameter to view details
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
