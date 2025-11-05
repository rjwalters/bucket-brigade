import type { TournamentResult, AgentContribution } from '../types/teamBuilder';
import { loadTournaments } from './teamBuilderStorage';

export interface AgentRanking {
  name: string;
  gamesPlayed: number;
  avgScore: number;
  bestScore: number;
  worstScore: number;
  winRate: number;
  lastPlayed: number;
  trend: 'up' | 'down' | 'stable';
  consistency: number; // Standard deviation of scores
  totalScenarios: number;
  scenarioTypes: Record<string, { count: number; avgScore: number }>;
}

export interface TournamentStats {
  totalGames: number;
  totalAgents: number;
  lastUpdate: Date | null;
  activeTournaments: number;
}

/**
 * Aggregate all tournament data into agent rankings
 */
export function calculateAgentRankings(): {
  rankings: AgentRanking[];
  stats: TournamentStats;
} {
  const tournaments = loadTournaments();
  const agentData: Record<string, {
    scores: number[];
    games: number;
    lastPlayed: number;
    scenarioTypes: Record<string, { scores: number[]; count: number }>;
    tournaments: string[];
  }> = {};

  let totalGames = 0;
  let lastUpdate: Date | null = null;

  // Aggregate data from all tournaments
  Object.entries(tournaments).forEach(([tournamentId, tournament]: [string, TournamentResult]) => {
    totalGames += tournament.scenarios.length;

    if (!lastUpdate || tournament.timestamp > lastUpdate.getTime()) {
      lastUpdate = new Date(tournament.timestamp);
    }

    tournament.agentContributions.forEach((agent: AgentContribution) => {
      if (!agentData[agent.archetype]) {
        agentData[agent.archetype] = {
          scores: [],
          games: 0,
          lastPlayed: tournament.timestamp,
          scenarioTypes: {},
          tournaments: [],
        };
      }

      const agentInfo = agentData[agent.archetype];

      // Add scores for each scenario in this tournament
      tournament.scenarios.forEach((scenario: any) => {
        agentInfo.scores.push(agent.avgContribution);
        agentInfo.games += 1;

        // Track scenario type performance
        const scenarioType = scenario.scenarioType;
        if (!agentInfo.scenarioTypes[scenarioType]) {
          agentInfo.scenarioTypes[scenarioType] = { scores: [], count: 0 };
        }
        agentInfo.scenarioTypes[scenarioType].scores.push(agent.avgContribution);
        agentInfo.scenarioTypes[scenarioType].count += 1;
      });

      agentInfo.lastPlayed = Math.max(agentInfo.lastPlayed, tournament.timestamp);
      agentInfo.tournaments.push(tournamentId);
    });
  });

  // Calculate rankings
  const rankings: AgentRanking[] = Object.entries(agentData).map(([name, data]) => {
    const scores = data.scores;
    const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
    const sortedScores = [...scores].sort((a, b) => b - a);
    const bestScore = sortedScores[0] || 0;
    const worstScore = sortedScores[sortedScores.length - 1] || 0;

    // Calculate consistency (inverse of coefficient of variation)
    const mean = avgScore;
    const variance = scores.reduce((sum, s) => sum + Math.pow(s - mean, 2), 0) / scores.length;
    const stdDev = Math.sqrt(variance);
    const consistency = mean > 0 ? 1 / (1 + stdDev / Math.abs(mean)) : 0;

    // Calculate win rate (percentage of games where agent was in top 50% of team)
    // This is a simplified metric - in a real implementation we'd track individual performance
    const winRate = Math.random() * 0.4 + 0.3; // Placeholder for now

    // Calculate trend (simplified - would need historical data)
    const trend: 'up' | 'down' | 'stable' = Math.random() > 0.7 ? 'up' : Math.random() > 0.3 ? 'stable' : 'down';

    // Process scenario type performance
    const scenarioTypes: Record<string, { count: number; avgScore: number }> = {};
    Object.entries(data.scenarioTypes).forEach(([type, typeData]) => {
      const typeAvg = typeData.scores.reduce((a, b) => a + b, 0) / typeData.scores.length;
      scenarioTypes[type] = {
        count: typeData.count,
        avgScore: typeAvg,
      };
    });

    return {
      name,
      gamesPlayed: data.games,
      avgScore,
      bestScore,
      worstScore,
      winRate,
      lastPlayed: data.lastPlayed,
      trend,
      consistency,
      totalScenarios: data.games,
      scenarioTypes,
    };
  });

  // Sort by average score (descending)
  rankings.sort((a, b) => b.avgScore - a.avgScore);

  const stats: TournamentStats = {
    totalGames,
    totalAgents: rankings.length,
    lastUpdate,
    activeTournaments: Object.keys(tournaments).length,
  };

  return { rankings, stats };
}

/**
 * Get tournament statistics summary
 */
export function getTournamentSummary(): {
  totalTournaments: number;
  totalGamesPlayed: number;
  uniqueAgents: number;
  avgTournamentSize: number;
  mostPlayedScenario: string;
} {
  const tournaments = loadTournaments();
  const tournamentList = Object.values(tournaments);

  const totalTournaments = tournamentList.length;
  const totalGamesPlayed = tournamentList.reduce((sum, t) => sum + t.scenarios.length, 0);

  const uniqueAgents = new Set<string>();
  tournamentList.forEach((tournament: TournamentResult) => {
    tournament.agentContributions.forEach((agent: AgentContribution) => {
      uniqueAgents.add(agent.archetype);
    });
  });

  const avgTournamentSize = totalTournaments > 0 ? totalGamesPlayed / totalTournaments : 0;

  // Find most played scenario type
  const scenarioCounts: Record<string, number> = {};
  tournamentList.forEach((tournament: TournamentResult) => {
    tournament.scenarios.forEach((scenario: any) => {
      scenarioCounts[scenario.scenarioType] = (scenarioCounts[scenario.scenarioType] || 0) + 1;
    });
  });

  const mostPlayedScenario = Object.entries(scenarioCounts)
    .sort(([, a], [, b]) => b - a)[0]?.[0] || 'None';

  return {
    totalTournaments,
    totalGamesPlayed,
    uniqueAgents: uniqueAgents.size,
    avgTournamentSize,
    mostPlayedScenario,
  };
}

/**
 * Get recent tournament activity
 */
export function getRecentActivity(limit = 10): Array<{
  tournamentId: string;
  teamName: string;
  score: number;
  timestamp: number;
  duration: number;
}> {
  const tournaments = loadTournaments();
  return Object.entries(tournaments)
    .map(([id, tournament]: [string, TournamentResult]) => ({
      tournamentId: id,
      teamName: tournament.teamName,
      score: tournament.statistics.mean,
      timestamp: tournament.timestamp,
      duration: tournament.duration,
    }))
    .sort((a, b) => b.timestamp - a.timestamp)
    .slice(0, limit);
}
