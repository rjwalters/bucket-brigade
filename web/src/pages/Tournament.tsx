import { useState, useEffect, useRef } from 'react';
import { Trophy, Play, Pause, RotateCcw, TrendingUp, Users, Clock, Target, Activity } from 'lucide-react';
import { createTournamentEngine } from '../utils/tournamentEngine';
import { getAllTeams, loadTeams, loadTournaments, saveTournaments } from '../utils/teamBuilderStorage';
import { calculateAgentRankings, getTournamentSummary, getRecentActivity } from '../utils/tournamentAnalysis';
import { getAllArchetypes } from '../utils/agentArchetypes';

export default function Tournament() {
  const [isRunning, setIsRunning] = useState(false);
  const [agentRankings, setAgentRankings] = useState<any[]>([]);
  const [totalGames, setTotalGames] = useState(0);
  const [tournamentSummary, setTournamentSummary] = useState<any>(null);
  const [recentActivity, setRecentActivity] = useState<any[]>([]);
  const tournamentEngineRef = useRef(createTournamentEngine());
  const intervalRef = useRef<number | null>(null);
  const isRunningRef = useRef(false);

  // Load existing tournament data
  useEffect(() => {
    console.log('🏆 Tournament component mounted, loading data...');
    loadExistingData();
  }, []);

  const loadExistingData = () => {
    console.log('📊 Loading tournament data...');
    const { rankings, stats } = calculateAgentRankings();
    const summary = getTournamentSummary();
    const activity = getRecentActivity();

    console.log(`📈 Loaded ${rankings.length} agent rankings, ${stats.totalGames} total games, ${stats.activeTournaments} tournaments`);

    setAgentRankings(rankings);
    setTotalGames(stats.totalGames);
    setTournamentSummary(summary);
    setRecentActivity(activity);
  };

  const runBackgroundTournament = async (): Promise<boolean> => {
    if (!isRunningRef.current) {
      console.log('⚠️ Tournament not running, skipping...');
      return false;
    }

    const startTime = Date.now();
    console.log('🎯 Starting background tournament...');

    try {
      // Generate a random team from available archetypes
      const teamSize = Math.floor(Math.random() * 7) + 4; // 4-10 agents
      const availableArchetypes = getAllArchetypes();
      const positions = [];

      for (let i = 0; i < teamSize; i++) {
        const randomArchetype = availableArchetypes[Math.floor(Math.random() * availableArchetypes.length)];
        positions.push(randomArchetype);
      }

      const randomTeam = {
        id: `random-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        name: `Random Team (${teamSize} agents)`,
        positions,
        createdAt: Date.now(),
        modifiedAt: Date.now(),
        tournamentHistory: [],
      };

      console.log(`🎲 Generated random team: ${randomTeam.name} with agents: ${positions.map((p: any) => p.name).join(', ')}`);

      // Run a small tournament (5 scenarios)
      console.log('🏃 Running tournament with 5 scenarios...');
      const result = await tournamentEngineRef.current.runTournament(randomTeam, {
        teamId: randomTeam.id,
        numScenarios: 5,
        seed: Math.floor(Math.random() * 1000000),
      });

      const duration = Date.now() - startTime;
      console.log(`✅ Tournament completed in ${duration}ms: ${result.teamName}`);
      console.log(`📈 Results: ${result.scenarios.length} scenarios, avg score: ${result.statistics.mean.toFixed(2)}, total games: ${result.scenarios.length}`);

      // Update rankings
      console.log('🔄 Updating rankings...');
      loadExistingData();
      console.log('✅ Rankings updated');

      return true; // Tournament completed successfully

    } catch (error) {
      const duration = Date.now() - startTime;
      console.error(`❌ Tournament failed after ${duration}ms:`, error);
      return false; // Tournament failed
    }
  };

  const startTournament = () => {
    console.log('🚀 Starting tournament engine...');
    setIsRunning(true);
    isRunningRef.current = true;
    console.log('🏃 Running continuous tournaments...');
    // Start the tournament loop
    runTournamentLoop();
  };

  const runTournamentLoop = async () => {
    console.log('🔄 Starting tournament loop (continuous mode)');
    let tournamentCount = 0;
    let attemptCount = 0;
    const startTime = Date.now();

    while (isRunningRef.current) {
      attemptCount++;
      const tournamentCompleted = await runBackgroundTournament();

      if (tournamentCompleted) {
        tournamentCount++;

        // Log progress every 10 tournaments
        if (tournamentCount % 10 === 0) {
          const elapsed = Date.now() - startTime;
          const rate = (tournamentCount / elapsed) * 1000; // tournaments per second
          console.log(`📊 Tournament progress: ${tournamentCount} completed (${rate.toFixed(1)}/sec)`);
        }
      } else {
        // Tournament failed - this shouldn't happen with random team generation
        console.error('❌ Tournament failed unexpectedly - continuing...');
      }

      // Small delay to keep UI responsive
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    console.log(`🏁 Tournament loop stopped after ${tournamentCount} tournaments (${attemptCount} attempts)`);
  };

  const stopTournament = () => {
    console.log('🛑 Stopping tournament engine...');
    setIsRunning(false);
    isRunningRef.current = false;
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
      console.log('✅ Cleared tournament interval');
    }
  };

  const resetRankings = () => {
    console.log('🔄 Resetting tournament rankings...');
    const teams = loadTeams();
    const tournaments = loadTournaments();

    console.log(`🗑️ Clearing tournament history from ${Object.keys(teams).length} teams and ${Object.keys(tournaments).length} tournaments`);

    // Clear tournament history (but keep teams)
    Object.values(teams).forEach((team: any) => {
      team.tournamentHistory = [];
    });

    // Clear all tournaments
    saveTournaments({});

    console.log('✅ Tournament data cleared, reloading...');
    loadExistingData();
  };

  const getRankIcon = (rank: number) => {
    if (rank === 1) return '🥇';
    if (rank === 2) return '🥈';
    if (rank === 3) return '🥉';
    return `#${rank}`;
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up': return <TrendingUp className="w-4 h-4 text-green-500" />;
      case 'down': return <TrendingUp className="w-4 h-4 text-red-500 rotate-180" />;
      default: return <div className="w-4 h-4" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-content-primary flex items-center gap-3">
            <Trophy className="w-8 h-8 text-yellow-500" />
            Agent Tournament
          </h1>
          <p className="text-content-secondary mt-1">
            Continuous ranking of agent performance across randomized scenarios
          </p>
        </div>

        <div className="flex items-center gap-3">
          {!isRunning ? (
            <button
              onClick={startTournament}
              className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
            >
              <Play className="w-4 h-4" />
              Start Continuous Tournament
            </button>
          ) : (
            <button
              onClick={stopTournament}
              className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
            >
              <Pause className="w-4 h-4" />
              Stop Tournament
            </button>
          )}

          <button
            onClick={resetRankings}
            className="flex items-center gap-2 px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            Reset Rankings
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-surface-secondary rounded-lg p-4">
          <div className="flex items-center gap-3">
            <Users className="w-5 h-5 text-blue-500" />
            <div>
              <p className="text-sm text-content-secondary">Agents Ranked</p>
              <p className="text-2xl font-bold text-content-primary">{agentRankings.length}</p>
            </div>
          </div>
        </div>

        <div className="bg-surface-secondary rounded-lg p-4">
          <div className="flex items-center gap-3">
            <Target className="w-5 h-5 text-green-500" />
            <div>
              <p className="text-sm text-content-secondary">Total Games</p>
              <p className="text-2xl font-bold text-content-primary">{totalGames}</p>
            </div>
          </div>
        </div>

        <div className="bg-surface-secondary rounded-lg p-4">
          <div className="flex items-center gap-3">
            <Activity className="w-5 h-5 text-purple-500" />
            <div>
              <p className="text-sm text-content-secondary">Tournaments</p>
              <p className="text-2xl font-bold text-content-primary">
                {tournamentSummary?.totalTournaments || 0}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-surface-secondary rounded-lg p-4">
          <div className="flex items-center gap-3">
            <Clock className="w-5 h-5 text-orange-500" />
            <div>
              <p className="text-sm text-content-secondary">Status</p>
              <p className="text-lg font-semibold text-content-primary">
                {isRunning ? 'Running' : 'Stopped'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Rankings Table */}
      <div className="bg-surface-secondary rounded-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-outline-primary">
          <h2 className="text-xl font-semibold text-content-primary">Agent Rankings</h2>
          <p className="text-sm text-content-secondary">
            Agents ranked by average marginal contribution across all tournament games
          </p>
        </div>

        {agentRankings.length === 0 ? (
          <div className="px-6 py-12 text-center">
            <Trophy className="w-12 h-12 text-content-tertiary mx-auto mb-4" />
            <h3 className="text-lg font-medium text-content-primary mb-2">Ready to Start Tournament</h3>
            <p className="text-content-secondary mb-4">
              Use the "Start Continuous Tournament" button above to begin automatically generating random teams and ranking agent performance across cooperative scenarios.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-surface-tertiary">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-content-secondary uppercase tracking-wider">
                    Rank
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-content-secondary uppercase tracking-wider">
                    Agent
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-content-secondary uppercase tracking-wider">
                    Avg Score
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-content-secondary uppercase tracking-wider">
                    Best Score
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-content-secondary uppercase tracking-wider">
                    Games
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-content-secondary uppercase tracking-wider">
                    Consistency
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-content-secondary uppercase tracking-wider">
                    Trend
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-outline-primary">
                {agentRankings.map((agent, index) => (
                  <tr key={agent.name} className="hover:bg-surface-tertiary/50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="text-lg font-bold text-content-primary">
                        {getRankIcon(index + 1)}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="font-medium text-content-primary">{agent.name}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="font-mono text-content-primary">
                        {agent.avgScore.toFixed(2)}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="font-mono text-green-600">
                        {agent.bestScore.toFixed(2)}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-content-secondary">
                      {agent.gamesPlayed}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="font-mono text-content-primary">
                        {(agent.consistency * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      {getTrendIcon(agent.trend)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Recent Activity */}
      {recentActivity.length > 0 && (
        <div className="bg-surface-secondary rounded-lg overflow-hidden">
          <div className="px-6 py-4 border-b border-outline-primary">
            <h3 className="text-lg font-semibold text-content-primary">Recent Tournament Activity</h3>
            <p className="text-sm text-content-secondary">
              Latest tournament results and team performances
            </p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-surface-tertiary">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-content-secondary uppercase tracking-wider">
                    Team
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-content-secondary uppercase tracking-wider">
                    Score
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-content-secondary uppercase tracking-wider">
                    Duration
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-content-secondary uppercase tracking-wider">
                    Time
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-outline-primary">
                {recentActivity.slice(0, 5).map((activity) => (
                  <tr key={activity.tournamentId} className="hover:bg-surface-tertiary/50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="font-medium text-content-primary">{activity.teamName}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="font-mono text-content-primary">
                        {activity.score.toFixed(2)}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-content-secondary">
                      {(activity.duration / 1000).toFixed(1)}s
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-content-secondary">
                      {new Date(activity.timestamp).toLocaleTimeString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tournament Info */}
      <div className="bg-surface-secondary rounded-lg p-6">
        <h3 className="text-lg font-semibold text-content-primary mb-3">How Rankings Work</h3>
        <div className="text-sm text-content-secondary space-y-2">
          <p>
            <strong>Marginal Contribution:</strong> Each agent's ranking is based on their estimated marginal
            contribution to team performance. This measures how much value an agent adds when included in random teams.
          </p>
          <p>
            <strong>Continuous Tournaments:</strong> Background tournaments run automatically, testing agents in
            mixed teams across diverse scenarios to ensure robust rankings.
          </p>
          <p>
            <strong>Scenario Diversity:</strong> Agents are evaluated across all 9 scenario types (cooperation,
            social dilemmas, coordination challenges) to prevent overfitting to specific game dynamics.
          </p>
          <p>
            <strong>Consistency Metric:</strong> Shows how reliably an agent performs across different team compositions
            and scenarios (higher is more consistent).
          </p>
        </div>
      </div>
    </div>
  );
}
