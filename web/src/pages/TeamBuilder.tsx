/**
 * Team Builder Page
 *
 * Main page for creating and testing agent teams
 */

import { useState, useEffect } from 'react';
import type {
  TeamComposition,
  AgentArchetype,
  TournamentProgress,
  TournamentResult,
} from '../types/teamBuilder';
import {
  createTeam,
  createTeamFromTemplate,
  loadTeam,
  saveTeam,
  getActiveTeamId,
  setActiveTeamId,
  updateTeamPosition,
  updateTeamName,
  getTeamSize,
  isTeamValid,
  saveTournamentResult,
} from '../utils/teamBuilderStorage';
import { getAllTemplates } from '../utils/teamTemplates';
import { TeamSelector } from '../components/team-builder/TeamSelector';
import { AgentSelectorModal } from '../components/team-builder/AgentSelectorModal';
import { TournamentRunner } from '../components/team-builder/TournamentRunner';
import { TournamentResults } from '../components/team-builder/TournamentResults';
import { createTournamentEngine } from '../utils/tournamentEngine';

export default function TeamBuilder() {
  const [team, setTeam] = useState<TeamComposition>(() => {
    const activeId = getActiveTeamId();
    if (activeId) {
      const loaded = loadTeam(activeId);
      if (loaded) return loaded;
    }
    return createTeam('My Brigade');
  });

  const [selectedPosition, setSelectedPosition] = useState<number | null>(null);
  const [showAgentModal, setShowAgentModal] = useState(false);
  const [editingName, setEditingName] = useState(false);
  const [tempName, setTempName] = useState(team.name);

  // Tournament state
  const [tournamentRunning, setTournamentRunning] = useState(false);
  const [tournamentProgress, setTournamentProgress] = useState<TournamentProgress | null>(
    null,
  );
  const [tournamentResult, setTournamentResult] = useState<TournamentResult | null>(null);

  // Save team when it changes
  useEffect(() => {
    saveTeam(team);
    setActiveTeamId(team.id);
  }, [team]);

  const handlePositionClick = (position: number) => {
    setSelectedPosition(position);
    setShowAgentModal(true);
  };

  const handleAgentSelect = (archetype: AgentArchetype) => {
    if (selectedPosition === null) return;

    const updated = updateTeamPosition(team.id, selectedPosition, archetype);
    if (updated) {
      setTeam(updated);
    }
    setSelectedPosition(null);
  };

  const handleNameSave = () => {
    const updated = updateTeamName(team.id, tempName.trim() || 'My Brigade');
    if (updated) {
      setTeam(updated);
    }
    setEditingName(false);
  };

  const handleLoadTemplate = (templateId: string) => {
    const template = getAllTemplates().find((t) => t.id === templateId);
    if (template) {
      const newTeam = createTeamFromTemplate(template);
      setTeam(newTeam);
    }
  };

  const handleAddPosition = () => {
    if (team.positions.length < 10) {
      setTeam({
        ...team,
        positions: [...team.positions, null],
      });
    }
  };

  const handleRemovePosition = () => {
    if (team.positions.length > 2) {
      setTeam({
        ...team,
        positions: team.positions.slice(0, -1),
      });
    }
  };

  const handleClearTeam = () => {
    setTeam({
      ...team,
      positions: team.positions.map(() => null),
    });
  };

  const handleStartTournament = async () => {
    setTournamentRunning(true);
    setTournamentProgress(null);
    setTournamentResult(null);

    const engine = createTournamentEngine();

    try {
      const result = await engine.runTournament(
        team,
        {
          teamId: team.id,
          numScenarios: 100,
        },
        (progress) => {
          setTournamentProgress(progress);
        },
      );

      setTournamentResult(result);
      saveTournamentResult(result);
    } catch (error) {
      console.error('Tournament failed:', error);
    } finally {
      setTournamentRunning(false);
      setTournamentProgress(null);
    }
  };

  const handleCancelTournament = () => {
    setTournamentRunning(false);
    setTournamentProgress(null);
  };

  const handleCloseResults = () => {
    setTournamentResult(null);
  };

  const handleRunAgain = () => {
    setTournamentResult(null);
    handleStartTournament();
  };

  const teamSize = getTeamSize(team);
  const valid = isTeamValid(team);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-gray-100 mb-2">
            🔥 Build Your Brigade 🔥
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Compose a team of agents and test them against 100 scenarios
          </p>
        </div>

        {/* Team Name */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mb-8">
          <div className="flex items-center justify-center gap-4">
            <label className="text-sm font-medium text-gray-600 dark:text-gray-400">
              Team Name:
            </label>
            {editingName ? (
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={tempName}
                  onChange={(e) => setTempName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') handleNameSave();
                    if (e.key === 'Escape') {
                      setEditingName(false);
                      setTempName(team.name);
                    }
                  }}
                  className="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  autoFocus
                />
                <button
                  type="button"
                  onClick={handleNameSave}
                  className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded"
                >
                  Save
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setEditingName(false);
                    setTempName(team.name);
                  }}
                  className="px-3 py-1 bg-gray-300 dark:bg-gray-600 hover:bg-gray-400 dark:hover:bg-gray-500 text-gray-800 dark:text-gray-200 rounded"
                >
                  Cancel
                </button>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => setEditingName(true)}
                className="text-xl font-bold text-gray-900 dark:text-gray-100 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
              >
                {team.name} ✏️
              </button>
            )}
          </div>
        </div>

        {/* Team Selector */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8 mb-8">
          <TeamSelector
            team={team}
            onPositionClick={handlePositionClick}
            selectedPosition={selectedPosition}
          />
        </div>

        {/* Controls */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          {/* Team Size Controls */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
              Team Size: {teamSize}/{team.positions.length}
            </h3>
            <div className="flex gap-3">
              <button
                type="button"
                onClick={handleAddPosition}
                disabled={team.positions.length >= 10}
                className="flex-1 px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors"
              >
                + Add Position
              </button>
              <button
                type="button"
                onClick={handleRemovePosition}
                disabled={team.positions.length <= 2}
                className="flex-1 px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors"
              >
                - Remove Last
              </button>
            </div>
            <button
              type="button"
              onClick={handleClearTeam}
              className="w-full mt-3 px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white font-medium rounded-lg transition-colors"
            >
              Clear All Agents
            </button>
          </div>

          {/* Template Selector */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
              Load Template
            </h3>
            <select
              onChange={(e) => {
                if (e.target.value) {
                  handleLoadTemplate(e.target.value);
                  e.target.value = '';
                }
              }}
              className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              defaultValue=""
            >
              <option value="">Select a template...</option>
              {getAllTemplates().map((template) => (
                <option key={template.id} value={template.id}>
                  {template.name}
                </option>
              ))}
            </select>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
              Pre-built teams for different scenarios
            </p>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
          <div className="flex flex-col sm:flex-row gap-4">
            <button
              type="button"
              onClick={handleStartTournament}
              disabled={!valid || tournamentRunning}
              className="flex-1 px-8 py-4 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white text-lg font-bold rounded-lg transition-colors shadow-lg hover:shadow-xl"
            >
              {tournamentRunning ? 'Running...' : 'Start Tournament →'}
            </button>
            <button
              type="button"
              className="px-8 py-4 bg-gray-600 hover:bg-gray-700 text-white font-semibold rounded-lg transition-colors"
            >
              Save Team
            </button>
          </div>
          {!valid && (
            <p className="text-sm text-red-600 dark:text-red-400 mt-3 text-center">
              Team must have at least 2 agents to start tournament
            </p>
          )}
        </div>

        {/* Info Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
            <div className="text-3xl mb-2">🎮</div>
            <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-1">
              Click to Select
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Click any position to choose an agent archetype
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
            <div className="text-3xl mb-2">🏆</div>
            <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-1">
              100 Scenarios
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Test your team against diverse challenges
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
            <div className="text-3xl mb-2">📊</div>
            <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-1">
              Deep Analysis
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              See which agents contribute most to success
            </p>
          </div>
        </div>
      </div>

      {/* Agent Selector Modal */}
      <AgentSelectorModal
        isOpen={showAgentModal}
        onClose={() => {
          setShowAgentModal(false);
          setSelectedPosition(null);
        }}
        onSelect={handleAgentSelect}
        position={selectedPosition ?? 0}
        currentArchetype={
          selectedPosition !== null ? team.positions[selectedPosition] : null
        }
      />

      {/* Tournament Progress */}
      {tournamentRunning && tournamentProgress && (
        <TournamentRunner
          progress={tournamentProgress}
          onCancel={handleCancelTournament}
        />
      )}

      {/* Tournament Results */}
      {tournamentResult && (
        <TournamentResults
          result={tournamentResult}
          onClose={handleCloseResults}
          onRunAgain={handleRunAgain}
          onSave={() => {
            // Already saved in the engine
          }}
        />
      )}
    </div>
  );
}
