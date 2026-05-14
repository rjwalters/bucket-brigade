import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Play } from 'lucide-react';
import { GameParameters } from '../components/GameParameters';
import { HeroSection } from '../components/dashboard/HeroSection';
import { TeamSelector } from '../components/dashboard/TeamSelector';
import { ScenarioSelector } from '../components/dashboard/ScenarioSelector';
import { AboutSection } from '../components/dashboard/AboutSection';
import {
  TEAM_PRESETS,
  TEST_SCENARIOS,
  type ScenarioConfig
} from '../data/dashboardPresets';
import {
  generateRandomScenario,
  generateRandomAgents
} from '../utils/dashboardRandom';

/**
 * Landing page that lets the user pick a team preset and scenario, preview
 * the resulting parameters, and launch a game.
 *
 * The bulk of the rendering is delegated to focused subcomponents in
 * `components/dashboard/`. This page is responsible for orchestrating
 * selection state, handling "random" regeneration, and stashing the chosen
 * configuration in `sessionStorage` before navigating to `/replay/new`.
 */
const SimpleDashboard: React.FC = () => {
  const navigate = useNavigate();
  const [selectedTeam, setSelectedTeam] = useState<string>('mixed_balanced');
  const [selectedScenario, setSelectedScenario] = useState<string>('default');

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

  const handleRandomTeam = useCallback(() => {
    const randomIndex = Math.floor(Math.random() * TEAM_PRESETS.length);
    setSelectedTeam(TEAM_PRESETS[randomIndex].id);
  }, []);

  const handleRandomScenario = useCallback(() => {
    const randomIndex = Math.floor(Math.random() * TEST_SCENARIOS.length);
    setSelectedScenario(TEST_SCENARIOS[randomIndex].id);
  }, []);

  const handleRegenerateScenario = useCallback(() => {
    setRandomScenarioParams(generateRandomScenario());
  }, []);

  const handleRegenerateAgents = useCallback(() => {
    setRandomAgents(generateRandomAgents());
  }, []);

  // Get team data, using random agents if 'random' is selected
  const selectedTeamData = useMemo(() => {
    if (selectedTeam === 'random') {
      const base = TEAM_PRESETS.find(t => t.id === 'random');
      return base ? { ...base, archetypes: randomAgents } : undefined;
    }
    return TEAM_PRESETS.find(t => t.id === selectedTeam);
  }, [selectedTeam, randomAgents]);

  // Get scenario data, using random params if 'random' is selected and update
  // num_agents to match team size.
  const selectedScenarioData = useMemo(() => {
    const baseScenarioData = selectedScenario === 'random'
      ? (() => {
          const base = TEST_SCENARIOS.find(s => s.id === 'random');
          return base ? { ...base, params: randomScenarioParams } : undefined;
        })()
      : TEST_SCENARIOS.find(s => s.id === selectedScenario);

    if (!baseScenarioData || !selectedTeamData) {
      return baseScenarioData;
    }

    return {
      ...baseScenarioData,
      params: {
        ...baseScenarioData.params,
        num_agents: selectedTeamData.archetypes.length
      }
    };
  }, [selectedScenario, randomScenarioParams, selectedTeamData]);

  const handleRunGame = useCallback(() => {
    // Store selections in sessionStorage for the game engine to use.
    let team = TEAM_PRESETS.find(t => t.id === selectedTeam);
    let scenario = TEST_SCENARIOS.find(s => s.id === selectedScenario);

    // Use generated random values if applicable.
    if (selectedTeam === 'random' && team) {
      team = { ...team, archetypes: randomAgents };
    }
    if (selectedScenario === 'random' && scenario) {
      scenario = { ...scenario, params: randomScenarioParams };
    }

    if (team && scenario) {
      // Update scenario's num_agents to match the actual team size.
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
  }, [selectedTeam, selectedScenario, randomAgents, randomScenarioParams, navigate]);

  return (
    <div className="max-w-5xl mx-auto space-y-8">
      <HeroSection />

      {/* Main Configuration */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <TeamSelector
          teams={TEAM_PRESETS}
          selectedId={selectedTeam}
          onSelect={setSelectedTeam}
          onRandom={handleRandomTeam}
        />
        <ScenarioSelector
          scenarios={TEST_SCENARIOS}
          selectedId={selectedScenario}
          onSelect={setSelectedScenario}
          onRandom={handleRandomScenario}
        />
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

      <AboutSection />
    </div>
  );
};

export default SimpleDashboard;
