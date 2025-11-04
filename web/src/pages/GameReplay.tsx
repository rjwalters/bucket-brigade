import React, { useState, useEffect, useCallback } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { Home, Loader2, X, Trash2, Download } from 'lucide-react';
import type { GameReplay } from '../types';
import { loadGameReplays, saveGameReplays } from '../utils/storage';
import { runGameBatch } from '../utils/runSimulation';
import Town from '../components/Town';
import AgentLayer from '../components/AgentLayer';
import ReplayControls from '../components/ReplayControls';
import GameSidebar from '../components/GameSidebar';
import GameAnalysis from '../components/GameAnalysis';

const GameReplayPage: React.FC = () => {
  const { gameId } = useParams<{ gameId?: string }>();
  const navigate = useNavigate();
  const [games, setGames] = useState<GameReplay[]>([]);
  const [selectedGame, setSelectedGame] = useState<GameReplay | null>(null);
  const [currentNight, setCurrentNight] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1000); // milliseconds
  const [phase, setPhase] = useState<'day' | 'night'>('day');
  const [isSimulating, setIsSimulating] = useState(false);
  const [simulationProgress, setSimulationProgress] = useState({ completed: 0, total: 0 });
  const [simulationStats, setSimulationStats] = useState<any>(null);

  // Run simulation if gameId is 'new'
  useEffect(() => {
    if (gameId === 'new') {
      runNewSimulation();
    } else {
      loadExistingGames();
    }
  }, [gameId]);

  const runNewSimulation = async () => {
    setIsSimulating(true);

    try {
      // Load team and scenario from sessionStorage
      const teamStr = sessionStorage.getItem('selected_team');
      const scenarioStr = sessionStorage.getItem('selected_scenario');

      if (!teamStr || !scenarioStr) {
        console.error('No team or scenario selected');
        navigate('/');
        return;
      }

      const team = JSON.parse(teamStr);
      const scenario = JSON.parse(scenarioStr);

      // Run batch simulation (100 games)
      const result = await runGameBatch(
        team,
        scenario,
        100,
        (completed, total) => {
          setSimulationProgress({ completed, total });
        }
      );

      // Save statistics
      setSimulationStats(result.statistics);

      // Pick a representative game (one close to median performance)
      const sortedGames = [...result.games].sort((a, b) => {
        const scoreA = a.nights.reduce((sum, night) => sum + night.rewards.reduce((s, r) => s + r, 0), 0);
        const scoreB = b.nights.reduce((sum, night) => sum + night.rewards.reduce((s, r) => s + r, 0), 0);
        return scoreA - scoreB;
      });
      const medianGame = sortedGames[Math.floor(sortedGames.length / 2)];

      // Add statistics to the game
      const gameWithStats = {
        ...medianGame,
        statistics: {
          avgAgentScores: result.statistics.avgAgentScores,
          stdErrAgentScores: result.statistics.stdErrAgentScores,
          avgFinalScore: result.statistics.avgFinalScore,
          stdErrFinalScore: result.statistics.stdErrFinalScore,
          numGames: result.statistics.numGames
        }
      };

      // Save the representative game to localStorage
      const existingGames = loadGameReplays();
      const updatedGames = [...existingGames, gameWithStats];
      saveGameReplays(updatedGames);

      // Display the game
      setGames(updatedGames);
      setSelectedGame(gameWithStats);
      setIsSimulating(false);

      // Navigate to the new game
      navigate(`/replay/${updatedGames.length - 1}`);
    } catch (error) {
      console.error('Simulation failed:', error);
      setIsSimulating(false);
      navigate('/');
    }
  };

  const loadExistingGames = () => {
    const loadedGames = loadGameReplays();
    setGames(loadedGames);

    if (gameId && loadedGames[parseInt(gameId)]) {
      setSelectedGame(loadedGames[parseInt(gameId)]);
    } else if (loadedGames.length > 0) {
      setSelectedGame(loadedGames[0]);
    }
  };

  const currentNightData = selectedGame?.nights[currentNight];

  // Auto-play functionality
  useEffect(() => {
    if (!isPlaying || !selectedGame) return;

    const interval = setInterval(() => {
      setCurrentNight(prev => {
        if (prev >= selectedGame.nights.length - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, speed);

    return () => clearInterval(interval);
  }, [isPlaying, selectedGame, speed]);

  const handlePlayPause = useCallback(() => {
    setIsPlaying(!isPlaying);
  }, [isPlaying]);

  const handleReset = useCallback(() => {
    setCurrentNight(0);
    setIsPlaying(false);
  }, []);

  const handlePrevNight = useCallback(() => {
    setCurrentNight(prev => Math.max(0, prev - 1));
  }, []);

  const handleNextNight = useCallback(() => {
    if (!selectedGame) return;
    setCurrentNight(prev => Math.min(selectedGame.nights.length - 1, prev + 1));
  }, [selectedGame]);

  const handleSpeedChange = useCallback((newSpeed: number) => {
    setSpeed(newSpeed);
  }, []);

  const handleGameSelect = useCallback((game: GameReplay) => {
    setSelectedGame(game);
    setCurrentNight(0);
    setIsPlaying(false);
  }, []);

  const handleDeleteGame = useCallback((index: number, event: React.MouseEvent) => {
    event.stopPropagation(); // Prevent selecting the game when deleting
    const updatedGames = games.filter((_, i) => i !== index);
    saveGameReplays(updatedGames);
    setGames(updatedGames);

    // If we deleted the selected game, select another or none
    if (selectedGame === games[index]) {
      if (updatedGames.length > 0) {
        setSelectedGame(updatedGames[0]);
      } else {
        setSelectedGame(null);
      }
    }
  }, [games, selectedGame]);

  const handleDeleteAll = useCallback(() => {
    if (window.confirm('Are you sure you want to delete all game replays? This cannot be undone.')) {
      saveGameReplays([]);
      setGames([]);
      setSelectedGame(null);
    }
  }, []);

  const handleExportAll = useCallback(() => {
    const dataStr = JSON.stringify(games, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `bucket-brigade-replays-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }, [games]);

  // Show loading screen during simulation
  if (isSimulating) {
    const progress = simulationProgress.total > 0
      ? (simulationProgress.completed / simulationProgress.total) * 100
      : 0;

    return (
      <div className="text-center py-12">
        <div className="max-w-md mx-auto">
          <Loader2 className="w-16 h-16 text-blue-600 dark:text-blue-400 mx-auto mb-4 animate-spin" />
          <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-4">
            Running Simulation
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            Simulating 100 games to gather statistics...
          </p>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-4 mb-2">
            <div
              className="bg-blue-600 h-4 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            {simulationProgress.completed} / {simulationProgress.total} games completed
          </p>
        </div>
      </div>
    );
  }

  if (games.length === 0 && !isSimulating) {
    return (
      <div className="text-center py-12">
        <div className="max-w-md mx-auto">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-4">
            No Games Available
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            Run a game from the dashboard to see results here.
          </p>
          <Link to="/" className="btn-primary inline-flex items-center">
            <Home className="w-4 h-4 mr-2" />
            Back to Dashboard
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Game Replay</h1>
        <Link to="/" className="btn-secondary inline-flex items-center">
          <Home className="w-4 h-4 mr-2" />
          Dashboard
        </Link>
      </div>

      {/* Game Selection */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Game Replays ({games.length})</h2>
          <div className="flex gap-2">
            <button
              onClick={handleExportAll}
              disabled={games.length === 0}
              className="btn-secondary flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              title="Export all games as JSON"
            >
              <Download className="w-4 h-4" />
              Export All
            </button>
            <button
              onClick={handleDeleteAll}
              disabled={games.length === 0}
              className="btn-secondary flex items-center gap-2 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-950 disabled:opacity-50 disabled:cursor-not-allowed"
              title="Delete all game replays"
            >
              <Trash2 className="w-4 h-4" />
              Delete All
            </button>
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {games.map((game, index) => {
            const timestamp = game.timestamp ? new Date(game.timestamp) : null;
            const teamName = game.teamName || 'Unknown Team';
            const scenarioName = game.scenarioName || 'Custom Scenario';

            return (
              <div
                key={index}
                onClick={() => handleGameSelect(game)}
                className={`relative p-4 rounded-lg border-2 cursor-pointer text-left transition-colors ${
                  selectedGame === game
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-950 dark:border-blue-400'
                    : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                }`}
              >
                {/* Delete button */}
                <button
                  onClick={(e) => handleDeleteGame(index, e)}
                  className="absolute top-2 right-2 p-1 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-500 dark:text-gray-400 hover:text-red-600 dark:hover:text-red-400 transition-colors"
                  title="Delete this replay"
                >
                  <X className="w-4 h-4" />
                </button>

                {/* Game info */}
                <div className="font-medium text-gray-900 dark:text-gray-100 pr-6">{teamName}</div>
                <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">{scenarioName}</div>
                <div className="text-xs text-gray-500 dark:text-gray-500 mt-2">
                  {timestamp ? timestamp.toLocaleString() : 'No timestamp'}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                  {game.nights.length} nights • {game.scenario.num_agents} agents
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  β={game.scenario.beta.toFixed(2)} • κ={game.scenario.kappa.toFixed(2)}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Game Replay */}
      {selectedGame && currentNightData && (
        <>
          {/* Replay Controls */}
          <ReplayControls
            currentNight={currentNight}
            totalNights={selectedGame.nights.length}
            isPlaying={isPlaying}
            speed={speed}
            onPlayPause={handlePlayPause}
            onReset={handleReset}
            onPrev={handlePrevNight}
            onNext={handleNextNight}
            onSpeedChange={handleSpeedChange}
            nights={selectedGame.nights}
          />

          {/* Game Visualization and Info */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Game Visualization */}
            <div className="lg:col-span-2">
              <div className="card">
                <div className="relative flex justify-center">
                  {/* Day/Night Indicator - Top Right */}
                  <div className="absolute top-2 right-2 text-3xl z-10">
                    {phase === 'day' ? '☀️' : '🌙'}
                  </div>

                  {/* Town (houses) */}
                  <Town
                    houses={currentNightData.houses}
                    numAgents={selectedGame.scenario.num_agents}
                    archetypes={selectedGame.archetypes}
                  />

                  {/* Agent Layer */}
                  <AgentLayer
                    locations={currentNightData.locations}
                    signals={currentNightData.signals}
                    actions={currentNightData.actions}
                    onPhaseChange={setPhase}
                  />
                </div>
              </div>
            </div>

            {/* Game Sidebar */}
            <div>
              <GameSidebar
                scenario={selectedGame.scenario}
                currentNightData={currentNightData}
                allNights={selectedGame.nights}
                archetypes={selectedGame.archetypes}
                statistics={selectedGame.statistics}
              />
            </div>
          </div>

          {/* Post-Game Analysis - Show when reached the end */}
          {currentNight === selectedGame.nights.length - 1 && (
            <GameAnalysis game={selectedGame} />
          )}
        </>
      )}
    </div>
  );
};

export default GameReplayPage;
