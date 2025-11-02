import React, { useState, useEffect, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import { Home } from 'lucide-react';
import type { GameReplay } from '../types';
import { loadGameReplays } from '../utils/storage';
import Town from '../components/Town';
import AgentLayer from '../components/AgentLayer';
import ReplayControls from '../components/ReplayControls';
import GameSidebar from '../components/GameSidebar';

const GameReplay: React.FC = () => {
  const { gameId } = useParams<{ gameId?: string }>();
  const [games, setGames] = useState<GameReplay[]>([]);
  const [selectedGame, setSelectedGame] = useState<GameReplay | null>(null);
  const [currentNight, setCurrentNight] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1000); // milliseconds

  useEffect(() => {
    const loadedGames = loadGameReplays();
    setGames(loadedGames);

    if (gameId && loadedGames[parseInt(gameId)]) {
      setSelectedGame(loadedGames[parseInt(gameId)]);
    } else if (loadedGames.length > 0) {
      setSelectedGame(loadedGames[0]);
    }
  }, [gameId]);

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

  if (games.length === 0) {
    return (
      <div className="text-center py-12">
        <div className="max-w-md mx-auto">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">No Games Available</h2>
          <p className="text-gray-600 mb-6">
            You need to upload game replay data first. Run batch experiments and import the results.
          </p>
          <Link to="/" className="btn-primary">
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
        <h1 className="text-2xl font-bold text-gray-900">Game Replay</h1>
        <Link to="/" className="btn-secondary">
          <Home className="w-4 h-4 mr-2" />
          Dashboard
        </Link>
      </div>

      {/* Game Selection */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Select Game</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {games.map((game, index) => (
            <button
              key={index}
              onClick={() => handleGameSelect(game)}
              className={`p-4 rounded-lg border-2 text-left transition-colors ${
                selectedGame === game
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <div className="font-medium text-gray-900">Game #{index}</div>
              <div className="text-sm text-gray-600">
                {game.nights.length} nights • {game.scenario.num_agents} agents
              </div>
              <div className="text-sm text-gray-600">
                β={game.scenario.beta.toFixed(2)} • κ={game.scenario.kappa.toFixed(2)}
              </div>
            </button>
          ))}
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
          />

          {/* Game Visualization and Info */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Game Visualization */}
            <div className="lg:col-span-2">
              <div className="card">
                <div className="relative">
                  {/* Town (houses) */}
                  <Town houses={currentNightData.houses} />

                  {/* Agent Layer */}
                  <AgentLayer
                    locations={currentNightData.locations}
                    signals={currentNightData.signals}
                    actions={currentNightData.actions}
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
              />
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default GameReplay;
