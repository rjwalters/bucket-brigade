import React, { useState, useEffect, useCallback } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { Home } from 'lucide-react';
import type { GameReplay } from '../types';
import { loadGameReplays, saveGameReplays } from '../utils/storage';
import { runGameBatch } from '../utils/runSimulation';
import ReplayControls from '../components/ReplayControls';
import GameSidebar from '../components/GameSidebar';
import GameAnalysis from '../components/GameAnalysis';
import GameReplayTable from '../components/replay/GameReplayTable';
import GameStateVisualizer from '../components/replay/GameStateVisualizer';
import SimulationLoading from '../components/replay/SimulationLoading';
import { useReplayPlayback } from '../hooks/useReplayPlayback';

const GameReplayPage: React.FC = () => {
  const { gameId } = useParams<{ gameId?: string }>();
  const navigate = useNavigate();
  const [games, setGames] = useState<GameReplay[]>([]);
  const [selectedGame, setSelectedGame] = useState<GameReplay | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [simulationProgress, setSimulationProgress] = useState({ completed: 0, total: 0 });

  // Playback state and controls (encapsulated in hook)
  const {
    currentStep,
    currentNight,
    phase,
    displayNight,
    totalSteps,
    isPlaying,
    speed,
    togglePlayPause,
    reset,
    stepForward,
    stepBackward,
    setSpeed,
  } = useReplayPlayback(selectedGame);

  // Load/run on gameId change
  useEffect(() => {
    if (gameId === 'new') {
      runNewSimulation();
    } else {
      loadExistingGames();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
      const result = await runGameBatch(team, scenario, 100, (completed, total) => {
        setSimulationProgress({ completed, total });
      });

      // Pick a representative game (one close to median performance)
      const sortedGames = [...result.games].sort((a, b) => {
        const scoreA = a.nights.reduce(
          (sum, night) => sum + night.rewards.reduce((s, r) => s + r, 0),
          0
        );
        const scoreB = b.nights.reduce(
          (sum, night) => sum + night.rewards.reduce((s, r) => s + r, 0),
          0
        );
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
          numGames: result.statistics.numGames,
        },
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

  // Derived display state
  const currentNightData = selectedGame?.nights[currentNight];
  const displayNightData = selectedGame?.nights[displayNight];

  // For step 0 (Day 0), show initial state with all houses safe.
  const initialHouses = selectedGame ? Array(10).fill(0) : [];
  const displayHouses =
    currentStep === 0 && phase === 'day' ? initialHouses : displayNightData?.houses || [];

  const handleGameSelect = useCallback((game: GameReplay) => {
    setSelectedGame(game);
  }, []);

  const handleDeleteGame = useCallback(
    (index: number, event: React.MouseEvent) => {
      event.stopPropagation();
      const updatedGames = games.filter((_, i) => i !== index);
      saveGameReplays(updatedGames);
      setGames(updatedGames);

      if (selectedGame === games[index]) {
        setSelectedGame(updatedGames.length > 0 ? updatedGames[0] : null);
      }
    },
    [games, selectedGame]
  );

  const handleDeleteAll = useCallback(() => {
    if (
      window.confirm(
        'Are you sure you want to delete all game replays? This cannot be undone.'
      )
    ) {
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
    return (
      <SimulationLoading
        completed={simulationProgress.completed}
        total={simulationProgress.total}
      />
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
      <GameReplayTable
        games={games}
        selectedGame={selectedGame}
        onSelect={handleGameSelect}
        onDelete={handleDeleteGame}
        onDeleteAll={handleDeleteAll}
        onExportAll={handleExportAll}
      />

      {/* Game Replay */}
      {selectedGame && currentNightData && (
        <>
          {/* Replay Controls */}
          <ReplayControls
            currentStep={currentStep}
            totalSteps={totalSteps}
            currentNight={currentNight}
            totalNights={selectedGame.nights.length}
            isPlaying={isPlaying}
            speed={speed}
            phase={phase}
            onPlayPause={togglePlayPause}
            onReset={reset}
            onPrev={stepBackward}
            onNext={stepForward}
            onSpeedChange={setSpeed}
            nights={selectedGame.nights}
          />

          {/* Game Visualization and Info */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Game Visualization */}
            <div className="lg:col-span-2">
              <GameStateVisualizer
                selectedGame={selectedGame}
                currentNightData={currentNightData}
                displayHouses={displayHouses}
                phase={phase}
              />
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
