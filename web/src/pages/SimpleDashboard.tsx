import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Play, Flame, Shuffle, Info } from 'lucide-react';

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
    A: number;
    L: number;
    c: number;
    rho_ignite: number;
    N_min: number;
    p_spark: number;
    N_spark: number;
    num_agents: number;
  };
}

const TEAM_PRESETS: TeamPreset[] = [
  {
    id: 'firefighters',
    name: 'All Firefighters',
    description: 'Cooperative team focused on extinguishing fires',
    archetypes: ['firefighter', 'firefighter', 'firefighter', 'firefighter']
  },
  {
    id: 'coordinators',
    name: 'All Coordinators',
    description: 'Strategic team that coordinates responses',
    archetypes: ['coordinator', 'coordinator', 'coordinator', 'coordinator']
  },
  {
    id: 'heroes',
    name: 'All Heroes',
    description: 'Brave agents willing to take risks',
    archetypes: ['hero', 'hero', 'hero', 'hero']
  },
  {
    id: 'mixed',
    name: 'Mixed Team',
    description: 'Balanced team with diverse strategies',
    archetypes: ['firefighter', 'coordinator', 'hero', 'liar']
  },
  {
    id: 'free_riders',
    name: 'Free Riders',
    description: 'Selfish agents that minimize effort',
    archetypes: ['free_rider', 'free_rider', 'free_rider', 'free_rider']
  }
];

const TEST_SCENARIOS: ScenarioConfig[] = [
  {
    id: 'default',
    name: 'Default',
    description: 'Balanced scenario with moderate fire spread and good extinguish efficiency',
    params: {
      beta: 0.25,
      kappa: 0.5,
      A: 100,
      L: 100,
      c: 0.5,
      rho_ignite: 0.2,
      N_min: 12,
      p_spark: 0.02,
      N_spark: 12,
      num_agents: 4
    }
  },
  {
    id: 'early_containment',
    name: 'Early Containment',
    description: 'Single fire, low spread - tests if cooperation can contain early',
    params: {
      beta: 0.15,
      kappa: 0.6,
      A: 100,
      L: 100,
      c: 0.5,
      rho_ignite: 0.1,
      N_min: 10,
      p_spark: 0.0,
      N_spark: 0,
      num_agents: 4
    }
  },
  {
    id: 'rapid_spread',
    name: 'Rapid Spread',
    description: 'High spread rate - requires immediate coordinated response',
    params: {
      beta: 0.4,
      kappa: 0.5,
      A: 100,
      L: 100,
      c: 0.5,
      rho_ignite: 0.2,
      N_min: 15,
      p_spark: 0.03,
      N_spark: 15,
      num_agents: 4
    }
  },
  {
    id: 'low_efficiency',
    name: 'Low Efficiency',
    description: 'Poor extinguishing - tests sustained cooperation',
    params: {
      beta: 0.25,
      kappa: 0.3,
      A: 100,
      L: 100,
      c: 0.5,
      rho_ignite: 0.2,
      N_min: 20,
      p_spark: 0.02,
      N_spark: 20,
      num_agents: 4
    }
  },
  {
    id: 'high_cost',
    name: 'High Cost',
    description: 'Expensive work - tests willingness to pay cooperation costs',
    params: {
      beta: 0.25,
      kappa: 0.5,
      A: 100,
      L: 100,
      c: 2.0,
      rho_ignite: 0.2,
      N_min: 12,
      p_spark: 0.02,
      N_spark: 12,
      num_agents: 4
    }
  },
  {
    id: 'many_fires',
    name: 'Many Fires',
    description: 'Multiple starting fires - tests resource allocation',
    params: {
      beta: 0.25,
      kappa: 0.5,
      A: 100,
      L: 100,
      c: 0.5,
      rho_ignite: 0.5,
      N_min: 15,
      p_spark: 0.0,
      N_spark: 0,
      num_agents: 4
    }
  },
  {
    id: 'persistent_sparks',
    name: 'Persistent Sparks',
    description: 'Continuous new fires - tests sustained vigilance',
    params: {
      beta: 0.25,
      kappa: 0.5,
      A: 100,
      L: 100,
      c: 0.5,
      rho_ignite: 0.1,
      N_min: 20,
      p_spark: 0.08,
      N_spark: 20,
      num_agents: 4
    }
  },
  {
    id: 'asymmetric_incentives',
    name: 'Asymmetric Incentives',
    description: 'High penalty, low reward - tests fairness concerns',
    params: {
      beta: 0.25,
      kappa: 0.5,
      A: 50,
      L: 150,
      c: 0.5,
      rho_ignite: 0.2,
      N_min: 12,
      p_spark: 0.02,
      N_spark: 12,
      num_agents: 4
    }
  },
  {
    id: 'quick_game',
    name: 'Quick Game',
    description: 'Short duration - tests immediate response',
    params: {
      beta: 0.25,
      kappa: 0.5,
      A: 100,
      L: 100,
      c: 0.5,
      rho_ignite: 0.2,
      N_min: 5,
      p_spark: 0.0,
      N_spark: 0,
      num_agents: 4
    }
  },
  {
    id: 'marathon',
    name: 'Marathon',
    description: 'Long duration - tests fatigue and sustained cooperation',
    params: {
      beta: 0.2,
      kappa: 0.5,
      A: 100,
      L: 100,
      c: 0.5,
      rho_ignite: 0.15,
      N_min: 30,
      p_spark: 0.015,
      N_spark: 30,
      num_agents: 4
    }
  }
];

const SimpleDashboard: React.FC = () => {
  const navigate = useNavigate();
  const [selectedTeam, setSelectedTeam] = useState<string>('mixed');
  const [selectedScenario, setSelectedScenario] = useState<string>('default');
  const [showInfo, setShowInfo] = useState<boolean>(false);

  const handleRandomTeam = () => {
    const randomIndex = Math.floor(Math.random() * TEAM_PRESETS.length);
    setSelectedTeam(TEAM_PRESETS[randomIndex].id);
  };

  const handleRandomScenario = () => {
    const randomIndex = Math.floor(Math.random() * TEST_SCENARIOS.length);
    setSelectedScenario(TEST_SCENARIOS[randomIndex].id);
  };

  const handleRunGame = () => {
    // Store selections in sessionStorage for the game engine to use
    const team = TEAM_PRESETS.find(t => t.id === selectedTeam);
    const scenario = TEST_SCENARIOS.find(s => s.id === selectedScenario);

    if (team && scenario) {
      sessionStorage.setItem('selected_team', JSON.stringify(team));
      sessionStorage.setItem('selected_scenario', JSON.stringify(scenario));
      navigate('/replay/new');
    }
  };

  const selectedTeamData = TEAM_PRESETS.find(t => t.id === selectedTeam);
  const selectedScenarioData = TEST_SCENARIOS.find(s => s.id === selectedScenario);

  return (
    <div className="max-w-5xl mx-auto space-y-8">
      {/* Hero Section */}
      <div className="text-center bg-gradient-to-r from-orange-50 to-red-50 rounded-lg p-8 border border-orange-200">
        <div className="flex items-center justify-center mb-4">
          <Flame className="w-12 h-12 text-orange-600 mr-3" />
          <h1 className="text-5xl font-bold text-gray-900">Bucket Brigade</h1>
        </div>
        <p className="text-xl text-gray-700 mb-2 max-w-3xl mx-auto">
          Watch cooperation emerge (or fail) in a frontier town facing fire
        </p>
        <p className="text-base text-gray-600 max-w-2xl mx-auto">
          10 houses in a circle, 4 agents with buckets, and fire that spreads relentlessly.
          Will they work together or let the town burn?
        </p>
      </div>

      {/* Main Configuration */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Team Selection */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900">Choose Team</h2>
            <button
              onClick={handleRandomTeam}
              className="btn-secondary text-sm flex items-center gap-2"
            >
              <Shuffle className="w-4 h-4" />
              Random
            </button>
          </div>

          <div className="space-y-3">
            {TEAM_PRESETS.map((team) => (
              <label
                key={team.id}
                className={`block p-4 rounded-lg border-2 cursor-pointer transition-all ${
                  selectedTeam === team.id
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-blue-300'
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
                    <div className="font-medium text-gray-900">{team.name}</div>
                    <div className="text-sm text-gray-600 mt-1">{team.description}</div>
                  </div>
                </div>
              </label>
            ))}
          </div>

          {selectedTeamData && (
            <div className="mt-4 p-3 bg-gray-50 rounded-lg">
              <div className="text-sm font-medium text-gray-700 mb-2">Composition:</div>
              <div className="flex flex-wrap gap-2">
                {selectedTeamData.archetypes.map((archetype, idx) => (
                  <span
                    key={idx}
                    className="px-2 py-1 bg-white border border-gray-200 rounded text-xs font-medium text-gray-700"
                  >
                    {archetype}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Scenario Selection */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900">Choose Scenario</h2>
            <button
              onClick={handleRandomScenario}
              className="btn-secondary text-sm flex items-center gap-2"
            >
              <Shuffle className="w-4 h-4" />
              Random
            </button>
          </div>

          <div className="space-y-3 max-h-[500px] overflow-y-auto pr-2">
            {TEST_SCENARIOS.map((scenario) => (
              <label
                key={scenario.id}
                className={`block p-4 rounded-lg border-2 cursor-pointer transition-all ${
                  selectedScenario === scenario.id
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-blue-300'
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
                  <div className="font-medium text-gray-900">{scenario.name}</div>
                  <div className="text-sm text-gray-600 mt-1">{scenario.description}</div>
                </div>
              </label>
            ))}
          </div>

          {selectedScenarioData && (
            <div className="mt-4 p-3 bg-gray-50 rounded-lg">
              <div className="text-sm font-medium text-gray-700 mb-2">Parameters:</div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div><span className="text-gray-600">Fire spread:</span> {(selectedScenarioData.params.beta * 100).toFixed(0)}%</div>
                <div><span className="text-gray-600">Extinguish:</span> {(selectedScenarioData.params.kappa * 100).toFixed(0)}%</div>
                <div><span className="text-gray-600">Initial fires:</span> {(selectedScenarioData.params.rho_ignite * 100).toFixed(0)}%</div>
                <div><span className="text-gray-600">Min nights:</span> {selectedScenarioData.params.N_min}</div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Run Button */}
      <div className="text-center">
        <button
          onClick={handleRunGame}
          className="btn-primary text-lg px-12 py-4 flex items-center gap-3 mx-auto"
        >
          <Play className="w-6 h-6" />
          Run Game
        </button>
        <p className="text-sm text-gray-500 mt-3">
          Watch the game unfold with real-time visualization and post-game analysis
        </p>
      </div>

      {/* Learn More Section */}
      <div className="card bg-gray-50 border-gray-200">
        <button
          onClick={() => setShowInfo(!showInfo)}
          className="w-full flex items-center justify-between text-left"
        >
          <div className="flex items-center gap-2">
            <Info className="w-5 h-5 text-blue-600" />
            <h3 className="text-lg font-semibold text-gray-900">About Bucket Brigade</h3>
          </div>
          <span className="text-gray-500">{showInfo ? '▼' : '▶'}</span>
        </button>

        {showInfo && (
          <div className="mt-4 space-y-3 text-sm text-gray-700">
            <p>
              Bucket Brigade is a multi-agent cooperation game where agents must work together
              to save houses from fire. Each night, agents choose which house to help and whether
              to work or rest.
            </p>
            <p>
              <strong>The Challenge:</strong> Fire spreads between neighboring houses, but agents
              can extinguish fires by working. However, work has a cost, creating a tension between
              individual incentives and collective benefit.
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
