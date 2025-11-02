/**
 * Team Builder Storage Utilities
 *
 * Local storage management for teams and tournament results.
 */

import type {
  TeamComposition,
  TournamentResult,
  TeamBuilderStorage,
  AgentArchetype,
  TeamTemplate,
} from '../types/teamBuilder';
import { AGENT_ARCHETYPES } from './agentArchetypes';
import { TEAM_TEMPLATES } from './teamTemplates';

/**
 * Storage keys
 */
const STORAGE_KEYS = {
  TEAMS: 'bucket_brigade_teams',
  TOURNAMENTS: 'bucket_brigade_tournaments',
  ACTIVE_TEAM: 'bucket_brigade_active_team',
} as const;

/**
 * Generate unique ID
 */
function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Load all teams from storage
 */
export function loadTeams(): Record<string, TeamComposition> {
  try {
    const data = sessionStorage.getItem(STORAGE_KEYS.TEAMS);
    return data ? JSON.parse(data) : {};
  } catch (error) {
    console.error('Failed to load teams:', error);
    return {};
  }
}

/**
 * Save teams to storage
 */
export function saveTeams(teams: Record<string, TeamComposition>): void {
  try {
    sessionStorage.setItem(STORAGE_KEYS.TEAMS, JSON.stringify(teams));
  } catch (error) {
    console.error('Failed to save teams:', error);
  }
}

/**
 * Load all tournament results from storage
 */
export function loadTournaments(): Record<string, TournamentResult> {
  try {
    const data = sessionStorage.getItem(STORAGE_KEYS.TOURNAMENTS);
    return data ? JSON.parse(data) : {};
  } catch (error) {
    console.error('Failed to load tournaments:', error);
    return {};
  }
}

/**
 * Save tournament results to storage
 */
export function saveTournaments(tournaments: Record<string, TournamentResult>): void {
  try {
    sessionStorage.setItem(STORAGE_KEYS.TOURNAMENTS, JSON.stringify(tournaments));
  } catch (error) {
    console.error('Failed to save tournaments:', error);
  }
}

/**
 * Get active team ID
 */
export function getActiveTeamId(): string | null {
  try {
    return sessionStorage.getItem(STORAGE_KEYS.ACTIVE_TEAM);
  } catch (error) {
    console.error('Failed to get active team:', error);
    return null;
  }
}

/**
 * Set active team ID
 */
export function setActiveTeamId(teamId: string | null): void {
  try {
    if (teamId) {
      sessionStorage.setItem(STORAGE_KEYS.ACTIVE_TEAM, teamId);
    } else {
      sessionStorage.removeItem(STORAGE_KEYS.ACTIVE_TEAM);
    }
  } catch (error) {
    console.error('Failed to set active team:', error);
  }
}

/**
 * Create new team
 */
export function createTeam(name: string = 'New Team'): TeamComposition {
  return {
    id: generateId(),
    name,
    positions: Array(10).fill(null),
    createdAt: Date.now(),
    modifiedAt: Date.now(),
    tournamentHistory: [],
  };
}

/**
 * Create team from template
 */
export function createTeamFromTemplate(template: TeamTemplate): TeamComposition {
  const positions: (AgentArchetype | null)[] = template.archetypeIds.map(
    (id) => AGENT_ARCHETYPES[id] || null,
  );

  // Pad with nulls if less than 10 positions
  while (positions.length < 10) {
    positions.push(null);
  }

  return {
    id: generateId(),
    name: template.name,
    positions: positions.slice(0, 10),
    createdAt: Date.now(),
    modifiedAt: Date.now(),
    tournamentHistory: [],
  };
}

/**
 * Save team to storage
 */
export function saveTeam(team: TeamComposition): void {
  const teams = loadTeams();
  teams[team.id] = {
    ...team,
    modifiedAt: Date.now(),
  };
  saveTeams(teams);
}

/**
 * Load team by ID
 */
export function loadTeam(teamId: string): TeamComposition | null {
  const teams = loadTeams();
  return teams[teamId] || null;
}

/**
 * Delete team
 */
export function deleteTeam(teamId: string): void {
  const teams = loadTeams();
  delete teams[teamId];
  saveTeams(teams);

  // Clear active team if it was deleted
  if (getActiveTeamId() === teamId) {
    setActiveTeamId(null);
  }
}

/**
 * Get all teams as array
 */
export function getAllTeams(): TeamComposition[] {
  const teams = loadTeams();
  return Object.values(teams).sort((a, b) => b.modifiedAt - a.modifiedAt);
}

/**
 * Update team position
 */
export function updateTeamPosition(
  teamId: string,
  position: number,
  archetype: AgentArchetype | null,
): TeamComposition | null {
  const team = loadTeam(teamId);
  if (!team) return null;

  team.positions[position] = archetype;
  saveTeam(team);
  return team;
}

/**
 * Update team name
 */
export function updateTeamName(teamId: string, name: string): TeamComposition | null {
  const team = loadTeam(teamId);
  if (!team) return null;

  team.name = name;
  saveTeam(team);
  return team;
}

/**
 * Add tournament result to team history
 */
export function addTournamentToTeam(teamId: string, tournamentId: string): void {
  const team = loadTeam(teamId);
  if (!team) return;

  team.tournamentHistory.push(tournamentId);
  saveTeam(team);
}

/**
 * Save tournament result
 */
export function saveTournamentResult(result: TournamentResult): void {
  const tournaments = loadTournaments();
  tournaments[result.id] = result;
  saveTournaments(tournaments);

  // Add to team history
  addTournamentToTeam(result.teamId, result.id);
}

/**
 * Load tournament result by ID
 */
export function loadTournamentResult(tournamentId: string): TournamentResult | null {
  const tournaments = loadTournaments();
  return tournaments[tournamentId] || null;
}

/**
 * Get all tournament results for a team
 */
export function getTeamTournaments(teamId: string): TournamentResult[] {
  const team = loadTeam(teamId);
  if (!team) return [];

  const tournaments = loadTournaments();
  return team.tournamentHistory
    .map((id) => tournaments[id])
    .filter((t) => t != null)
    .sort((a, b) => b.timestamp - a.timestamp);
}

/**
 * Delete tournament result
 */
export function deleteTournamentResult(tournamentId: string): void {
  const tournaments = loadTournaments();
  delete tournaments[tournamentId];
  saveTournaments(tournaments);
}

/**
 * Get team size (non-null positions)
 */
export function getTeamSize(team: TeamComposition): number {
  return team.positions.filter((p) => p != null).length;
}

/**
 * Validate team (at least 2 agents)
 */
export function isTeamValid(team: TeamComposition): boolean {
  return getTeamSize(team) >= 2;
}

/**
 * Clear all data
 */
export function clearAllData(): void {
  try {
    sessionStorage.removeItem(STORAGE_KEYS.TEAMS);
    sessionStorage.removeItem(STORAGE_KEYS.TOURNAMENTS);
    sessionStorage.removeItem(STORAGE_KEYS.ACTIVE_TEAM);
  } catch (error) {
    console.error('Failed to clear data:', error);
  }
}

/**
 * Export data as JSON
 */
export function exportData(): string {
  const data: TeamBuilderStorage = {
    teams: loadTeams(),
    tournaments: loadTournaments(),
    activeTeamId: getActiveTeamId(),
    templates: TEAM_TEMPLATES,
  };
  return JSON.stringify(data, null, 2);
}

/**
 * Import data from JSON
 */
export function importData(jsonData: string): boolean {
  try {
    const data: TeamBuilderStorage = JSON.parse(jsonData);

    if (data.teams) {
      saveTeams(data.teams);
    }
    if (data.tournaments) {
      saveTournaments(data.tournaments);
    }
    if (data.activeTeamId) {
      setActiveTeamId(data.activeTeamId);
    }

    return true;
  } catch (error) {
    console.error('Failed to import data:', error);
    return false;
  }
}
