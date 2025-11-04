import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Play, Flame, Shuffle, Info } from 'lucide-react';
import { GameParameters } from '../components/GameParameters';

// Team presets with their archetype compositions
interface TeamPreset {
  id: string;
  name: string;
  description: string;
  archetypes: string[];
}

// Scenario configurations for testing different cooperation dynamics
interface ScenarioConfig {
  id: string;
  name: string;
  description: string;
  params: {
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
  };
}

// Note: These are parameterized heuristic agents for demonstration.
// Research uses evolved strategies (genetic algorithms) and Nash equilibria.
// Each archetype has 10 behavioral parameters (click agent name to see details).
const TEAM_PRESETS: TeamPreset[] = [
  {
    id: 'solo_hero',
    name: 'Solo: Hero',
    description: 'One agent against the flames - ultimate test of individual skill (1 agent)',
    archetypes: ['hero']
  },
  {
    id: 'duo_firefighters',
    name: 'Duo: Firefighters',
    description: 'Two honest firefighters - minimal team, maximum cooperation (2 agents)',
    archetypes: ['firefighter', 'firefighter']
  },
  {
    id: 'trio_mixed',
    name: 'Trio: Mixed',
    description: 'Small diverse team - tests cooperation with limited resources (3 agents)',
    archetypes: ['firefighter', 'coordinator', 'hero']
  },
  {
    id: 'all_firefighters',
    name: 'All Firefighters',
    description: 'Honest, hard-working, cooperative agents - classic teamwork (4 agents)',
    archetypes: ['firefighter', 'firefighter', 'firefighter', 'firefighter']
  },
  {
    id: 'mixed_balanced',
    name: 'Mixed: Balanced',
    description: 'Diverse team with complementary strengths (4 agents)',
    archetypes: ['firefighter', 'coordinator', 'hero', 'free_rider']
  },
  {
    id: 'large_coordinators',
    name: 'Large: Coordinators',
    description: 'Big team with high coordination - tests scalability (5 agents)',
    archetypes: ['coordinator', 'coordinator', 'coordinator', 'coordinator', 'coordinator']
  },
  {
    id: 'large_mixed',
    name: 'Large: Mixed',
    description: 'Diverse large team - complex social dynamics (6 agents)',
    archetypes: ['firefighter', 'firefighter', 'coordinator', 'hero', 'free_rider', 'liar']
  },
  {
    id: 'full_town',
    name: 'Full Town',
    description: 'Maximum team size - every house has a dedicated owner (10 agents)',
    archetypes: ['firefighter', 'firefighter', 'coordinator', 'coordinator', 'hero', 'hero', 'free_rider', 'free_rider', 'liar', 'liar']
  },
  {
    id: 'random',
    name: 'Random Team',
    description: 'Randomly generated team (1-10 agents, varied archetypes)',
    archetypes: ['random', 'random', 'random', 'random']
  }
];

// Research scenarios aligned with bucket_brigade/envs/scenarios.py
const TEST_SCENARIOS: ScenarioConfig[] = [
  {
    id: 'random',
    name: 'Random',
    description: 'Random parameters each game - tests generalization across diverse conditions',
    params: {
      beta: 0.25,
      kappa: 0.5,
      team_reward_house_survives: 100,
      team_penalty_house_burns: 100,
      reward_own_house_survives: 100,
      reward_other_house_survives: 50,
      penalty_own_house_burns: 0,
      penalty_other_house_burns: 0,
      c: 0.5,
      N_min: 15,
      p_spark: 0.02,
      num_agents: 4
    }
  },
  {
    id: 'default',
    name: 'Default',
    description: 'Balanced scenario with moderate fire spread and good extinguish efficiency',
    params: {
      beta: 0.25,
      kappa: 0.5,
      team_reward_house_survives: 100,
      team_penalty_house_burns: 100,
      reward_own_house_survives: 100,
      reward_other_house_survives: 50,
      penalty_own_house_burns: 0,
      penalty_other_house_burns: 0,
      c: 0.5,
      N_min: 12,
      p_spark: 0.02,
      num_agents: 4
    }
  },
  {
    id: 'easy',
    name: 'Easy',
    description: 'Low fire spread and high extinguish efficiency - cooperation should succeed',
    params: {
      beta: 0.1,
      kappa: 0.8,
      team_reward_house_survives: 100,
      team_penalty_house_burns: 100,
      reward_own_house_survives: 100,
      reward_other_house_survives: 50,
      penalty_own_house_burns: 0,
      penalty_other_house_burns: 0,
      c: 0.5,
      N_min: 10,
      p_spark: 0.01,
      num_agents: 4
    }
  },
  {
    id: 'hard',
    name: 'Hard',
    description: 'High fire spread and low extinguish efficiency - requires strong coordination',
    params: {
      beta: 0.4,
      kappa: 0.3,
      team_reward_house_survives: 100,
      team_penalty_house_burns: 100,
      reward_own_house_survives: 100,
      reward_other_house_survives: 50,
      penalty_own_house_burns: 0,
      penalty_other_house_burns: 0,
      c: 0.5,
      N_min: 15,
      p_spark: 0.05,
      num_agents: 4
    }
  },
  {
    id: 'trivial_cooperation',
    name: 'Trivial Cooperation',
    description: 'Fires are rare and extinguish easily - minimal cooperation needed',
    params: {
      beta: 0.15,
      kappa: 0.9,
      team_reward_house_survives: 100,
      team_penalty_house_burns: 100,
      reward_own_house_survives: 100,
      reward_other_house_survives: 50,
      penalty_own_house_burns: 0,
      penalty_other_house_burns: 0,
      c: 0.5,
      N_min: 12,
      p_spark: 0.0,
      num_agents: 4
    }
  },
  {
    id: 'early_containment',
    name: 'Early Containment',
    description: 'Fires start aggressive but can be stopped early with coordination',
    params: {
      beta: 0.35,
      kappa: 0.6,
      team_reward_house_survives: 100,
      team_penalty_house_burns: 100,
      reward_own_house_survives: 100,
      reward_other_house_survives: 50,
      penalty_own_house_burns: 0,
      penalty_other_house_burns: 0,
      c: 0.5,
      N_min: 12,
      p_spark: 0.02,
      num_agents: 4
    }
  },
  {
    id: 'greedy_neighbor',
    name: 'Greedy Neighbor',
    description: 'Social dilemma between self-interest and cooperation - high work cost',
    params: {
      beta: 0.15,
      kappa: 0.4,
      team_reward_house_survives: 100,
      team_penalty_house_burns: 100,
      reward_own_house_survives: 150,
      reward_other_house_survives: 25,
      penalty_own_house_burns: 0,
      penalty_other_house_burns: 0,
      c: 1.0,
      N_min: 12,
      p_spark: 0.02,
      num_agents: 4
    }
  },
  {
    id: 'sparse_heroics',
    name: 'Sparse Heroics',
    description: 'Few workers can make the difference - tests heroic action under cost',
    params: {
      beta: 0.1,
      kappa: 0.5,
      team_reward_house_survives: 100,
      team_penalty_house_burns: 100,
      reward_own_house_survives: 100,
      reward_other_house_survives: 50,
      penalty_own_house_burns: 0,
      penalty_other_house_burns: 0,
      c: 0.8,
      N_min: 20,
      p_spark: 0.02,
      num_agents: 4
    }
  },
  {
    id: 'rest_trap',
    name: 'Rest Trap',
    description: 'Fires usually extinguish themselves, but not always - tempts free-riding',
    params: {
      beta: 0.05,
      kappa: 0.95,
      team_reward_house_survives: 100,
      team_penalty_house_burns: 100,
      reward_own_house_survives: 100,
      reward_other_house_survives: 50,
      penalty_own_house_burns: 0,
      penalty_other_house_burns: 0,
      c: 0.2,
      N_min: 12,
      p_spark: 0.02,
      num_agents: 4
    }
  },
  {
    id: 'chain_reaction',
    name: 'Chain Reaction',
    description: 'High spread requires distributed teams - tests spatial coordination',
    params: {
      beta: 0.45,
      kappa: 0.6,
      team_reward_house_survives: 100,
      team_penalty_house_burns: 100,
      reward_own_house_survives: 100,
      reward_other_house_survives: 50,
      penalty_own_house_burns: 0,
      penalty_other_house_burns: 0,
      c: 0.7,
      N_min: 15,
      p_spark: 0.03,
      num_agents: 4
    }
  },
  {
    id: 'deceptive_calm',
    name: 'Deceptive Calm',
    description: 'Occasional flare-ups reward honest signaling - tests communication',
    params: {
      beta: 0.25,
      kappa: 0.6,
      team_reward_house_survives: 100,
      team_penalty_house_burns: 100,
      reward_own_house_survives: 100,
      reward_other_house_survives: 50,
      penalty_own_house_burns: 0,
      penalty_other_house_burns: 0,
      c: 0.4,
      N_min: 20,
      p_spark: 0.05,
      num_agents: 4
    }
  },
  {
    id: 'overcrowding',
    name: 'Overcrowding',
    description: 'Too many workers reduce efficiency - tests resource allocation',
    params: {
      beta: 0.2,
      kappa: 0.3,
      team_reward_house_survives: 50,
      team_penalty_house_burns: 100,
      reward_own_house_survives: 100,
      reward_other_house_survives: 50,
      penalty_own_house_burns: 0,
      penalty_other_house_burns: 0,
      c: 0.6,
      N_min: 12,
      p_spark: 0.02,
      num_agents: 4
    }
  },
  {
    id: 'mixed_motivation',
    name: 'Mixed Motivation',
    description: 'Ownership creates self-interest conflicts - tests fairness',
    params: {
      beta: 0.3,
      kappa: 0.5,
      team_reward_house_survives: 100,
      team_penalty_house_burns: 100,
      reward_own_house_survives: 100,
      reward_other_house_survives: 50,
      penalty_own_house_burns: 0,
      penalty_other_house_burns: 0,
      c: 0.6,
      N_min: 15,
      p_spark: 0.03,
      num_agents: 4
    }
  }
];

// Helper functions to generate random values
const generateRandomScenario = (): ScenarioConfig['params'] => {
  const hasSparks = Math.random() < 0.5; // 50% chance of sparks
  const N_min = Math.floor(Math.random() * (20 - 10 + 1)) + 10; // 10-20
  const num_agents = Math.floor(Math.random() * 10) + 1; // 1-10 agents

  return {
    beta: Math.random() * (0.35 - 0.15) + 0.15, // 0.15-0.35
    kappa: Math.random() * (0.6 - 0.4) + 0.4, // 0.4-0.6
      team_reward_house_survives: 100,
      team_penalty_house_burns: 100,
      reward_own_house_survives: 100,
      reward_other_house_survives: 50,
      penalty_own_house_burns: 0,
      penalty_other_house_burns: 0,
    c: 0.5,
    N_min: N_min,
    p_spark: hasSparks ? (Math.random() * (0.05 - 0.01) + 0.01) : 0, // 0 or 0.01-0.05
    num_agents: num_agents
  };
};

const generateRandomAgents = (): string[] => {
  const agentTypes = ['firefighter', 'free_rider', 'hero', 'coordinator', 'liar'];
  const teamSize = Math.floor(Math.random() * 10) + 1; // 1-10 agents
  return Array.from({ length: teamSize }, () =>
    agentTypes[Math.floor(Math.random() * agentTypes.length)]
  );
};

const SimpleDashboard: React.FC = () => {
  const navigate = useNavigate();
  const [selectedTeam, setSelectedTeam] = useState<string>('mixed_balanced');
  const [selectedScenario, setSelectedScenario] = useState<string>('default');
  const [showInfo, setShowInfo] = useState<boolean>(false);

  // State for randomly generated values
  const [randomScenarioParams, setRandomScenarioParams] = useState<ScenarioConfig['params']>(generateRandomScenario());
  const [randomAgents, setRandomAgents] = useState<string[]>(generateRandomAgents());

  // Regenerate random scenario params when 'random' scenario is selected
  useEffect(() => {
    if (selectedScenario === 'random') {
      setRandomScenarioParams(generateRandomScenario());
    }
  }, [selectedScenario]);

  // Regenerate random agents when 'random' team is selected
  useEffect(() => {
    if (selectedTeam === 'random') {
      setRandomAgents(generateRandomAgents());
    }
  }, [selectedTeam]);

  const handleRandomTeam = () => {
    const randomIndex = Math.floor(Math.random() * TEAM_PRESETS.length);
    setSelectedTeam(TEAM_PRESETS[randomIndex].id);
  };

  const handleRandomScenario = () => {
    const randomIndex = Math.floor(Math.random() * TEST_SCENARIOS.length);
    setSelectedScenario(TEST_SCENARIOS[randomIndex].id);
  };

  const handleRegenerateScenario = () => {
    setRandomScenarioParams(generateRandomScenario());
  };

  const handleRegenerateAgents = () => {
    setRandomAgents(generateRandomAgents());
  };

  const handleRunGame = () => {
    // Store selections in sessionStorage for the game engine to use
    let team = TEAM_PRESETS.find(t => t.id === selectedTeam);
    let scenario = TEST_SCENARIOS.find(s => s.id === selectedScenario);

    // Use generated random values if applicable
    if (selectedTeam === 'random' && team) {
      team = { ...team, archetypes: randomAgents };
    }
    if (selectedScenario === 'random' && scenario) {
      scenario = { ...scenario, params: randomScenarioParams };
    }

    if (team && scenario) {
      // Update scenario's num_agents to match the actual team size
      scenario = {
        ...scenario,
        params: {
          ...scenario.params,
          num_agents: team.archetypes.length
        }
      };

      sessionStorage.setItem('selected_team', JSON.stringify(team));
      sessionStorage.setItem('selected_scenario', JSON.stringify(scenario));
      navigate('/replay/new');
    }
  };

  // Get team data, using random agents if 'random' is selected
  const selectedTeamData = selectedTeam === 'random'
    ? { ...TEAM_PRESETS.find(t => t.id === 'random')!, archetypes: randomAgents }
    : TEAM_PRESETS.find(t => t.id === selectedTeam);

  // Get scenario data, using random params if 'random' is selected and update num_agents to match team size
  const baseScenarioData = selectedScenario === 'random'
    ? { ...TEST_SCENARIOS.find(s => s.id === 'random')!, params: randomScenarioParams }
    : TEST_SCENARIOS.find(s => s.id === selectedScenario);

  const selectedScenarioData = baseScenarioData && selectedTeamData
    ? {
        ...baseScenarioData,
        params: {
          ...baseScenarioData.params,
          num_agents: selectedTeamData.archetypes.length
        }
      }
    : baseScenarioData;

  return (
    <div className="max-w-5xl mx-auto space-y-8">
      {/* Hero Section */}
      <div className="text-center bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-950 dark:to-red-950 rounded-lg p-8 border border-orange-200 dark:border-orange-800">
        <div className="flex items-center justify-center mb-4">
          <Flame className="w-12 h-12 text-orange-600 mr-3" />
          <h1 className="text-5xl font-bold text-gray-900 dark:text-gray-100">Bucket Brigade</h1>
        </div>
        <p className="text-xl text-gray-700 dark:text-gray-300 mb-2 max-w-3xl mx-auto">
          Watch cooperation emerge (or fail) in a frontier town facing fire
        </p>
        <p className="text-base text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
          A circle of houses and fire that spreads relentlessly. Will the Agents work together or let the town burn?
        </p>
      </div>

      {/* Main Configuration */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Team Selection */}
        <div className="card flex flex-col h-[500px]">
          <div className="flex items-center justify-between mb-4 flex-shrink-0">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">Choose Team</h2>
            <button
              onClick={handleRandomTeam}
              className="btn-secondary text-sm flex items-center gap-2"
            >
              <Shuffle className="w-4 h-4" />
              Random
            </button>
          </div>

          <div className="space-y-3 flex-1 overflow-y-auto min-h-0 pr-2">
            {TEAM_PRESETS.map((team) => (
              <label
                key={team.id}
                className={`block p-4 rounded-lg border-2 cursor-pointer transition-all ${
                  selectedTeam === team.id
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-950'
                    : 'border-gray-200 dark:border-gray-700 hover:border-blue-300 dark:hover:border-blue-600'
                }`}
              >
                <input
                  type="radio"
                  name="team"
                  value={team.id}
                  checked={selectedTeam === team.id}
                  onChange={(e) => setSelectedTeam(e.target.value)}
                  className="sr-only"
                />
                <div className="flex items-start">
                  <div className="flex-1">
                    <div className="font-medium text-gray-900 dark:text-gray-100">{team.name}</div>
                    <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">{team.description}</div>
                  </div>
                </div>
              </label>
            ))}
          </div>
        </div>

        {/* Scenario Selection */}
        <div className="card flex flex-col h-[500px]">
          <div className="flex items-center justify-between mb-4 flex-shrink-0">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">Choose Scenario</h2>
            <button
              onClick={handleRandomScenario}
              className="btn-secondary text-sm flex items-center gap-2"
            >
              <Shuffle className="w-4 h-4" />
              Random
            </button>
          </div>

          <div className="space-y-3 flex-1 overflow-y-auto min-h-0 pr-2">
            {TEST_SCENARIOS.map((scenario) => (
              <label
                key={scenario.id}
                className={`block p-4 rounded-lg border-2 cursor-pointer transition-all ${
                  selectedScenario === scenario.id
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-950'
                    : 'border-gray-200 dark:border-gray-700 hover:border-blue-300 dark:hover:border-blue-600'
                }`}
              >
                <input
                  type="radio"
                  name="scenario"
                  value={scenario.id}
                  checked={selectedScenario === scenario.id}
                  onChange={(e) => setSelectedScenario(e.target.value)}
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
      </div>

      {/* Game Parameters */}
      {selectedTeamData && selectedScenarioData && (
        <div className="grid grid-cols-1 gap-6">
          <GameParameters
            teamArchetypes={selectedTeamData.archetypes}
            scenarioParams={selectedScenarioData.params}
            scenarioName={selectedScenarioData.name}
            isRandomTeam={selectedTeam === 'random'}
            isRandomScenario={selectedScenario === 'random'}
            onRegenerateTeam={handleRegenerateAgents}
            onRegenerateScenario={handleRegenerateScenario}
          />
        </div>
      )}

      {/* Run Button */}
      <div className="text-center">
        <button
          onClick={handleRunGame}
          className="btn-primary text-lg px-12 py-4 flex items-center gap-3 mx-auto"
        >
          <Play className="w-6 h-6" />
          Run Game
        </button>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-3">
          Watch the game unfold with real-time visualization and post-game analysis
        </p>
      </div>

      {/* Learn More Section */}
      <div className="card bg-gray-50 dark:bg-gray-800 border-gray-200 dark:border-gray-700">
        <button
          onClick={() => setShowInfo(!showInfo)}
          className="w-full flex items-center justify-between text-left"
        >
          <div className="flex items-center gap-2">
            <Info className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">About Bucket Brigade</h3>
          </div>
          <span className="text-gray-500 dark:text-gray-400">{showInfo ? '▼' : '▶'}</span>
        </button>

        {showInfo && (
          <div className="mt-4 space-y-3 text-sm text-gray-700 dark:text-gray-300">
            <p>
              Bucket Brigade is a multi-agent cooperation game where agents must work together
              to save a ring of 10 houses from fire. Each house is owned by one agent (with ownership
              assigned round-robin). Each night, agents choose which house to help and whether
              to work or rest.
            </p>
            <p>
              <strong>The Challenge:</strong> Fire spreads between neighboring houses, but agents
              can extinguish fires by working. However, work has a cost, creating a tension between
              protecting one's own houses and helping the collective.
            </p>
            <p>
              <strong>Agent Types:</strong> Different archetypes employ different strategies:
              Firefighters prioritize fire suppression, Coordinators plan strategically, Heroes
              take risks, Free Riders minimize effort, and Liars may signal incorrectly.
            </p>
            <p>
              <strong>Research Goal:</strong> Understanding which strategies succeed in different
              scenarios helps us learn about cooperation, trust, and coordination in multi-agent
              systems.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default SimpleDashboard;
