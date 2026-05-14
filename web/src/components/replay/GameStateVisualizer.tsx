import React from 'react';
import type { GameNight, GameReplay, HouseState } from '../../types';
import Town from '../Town';
import AgentLayer from '../AgentLayer';

interface GameStateVisualizerProps {
  selectedGame: GameReplay;
  currentNightData: GameNight;
  displayHouses: HouseState[];
  phase: 'day' | 'night';
}

/**
 * Visualizes the game state at a given moment in the replay.
 * Composes Town (houses + fires) and AgentLayer (agent positions + signals)
 * with a day/night indicator overlay.
 */
const GameStateVisualizerComponent: React.FC<GameStateVisualizerProps> = ({
  selectedGame,
  currentNightData,
  displayHouses,
  phase,
}) => {
  return (
    <div className="card">
      <div className="relative mx-auto" style={{ width: '640px', height: '640px' }}>
        {/* Day/Night Indicator - Top Right */}
        <div className="absolute top-2 right-2 text-3xl z-10">
          {phase === 'day' ? '☀️' : '🌙'}
        </div>

        {/* Town (houses) - show appropriate state based on phase */}
        <Town
          houses={displayHouses}
          numAgents={selectedGame.scenario.num_agents}
          archetypes={selectedGame.archetypes}
        />

        {/* Agent Layer */}
        <AgentLayer
          locations={currentNightData.locations}
          signals={currentNightData.signals}
          actions={currentNightData.actions}
          phase={phase}
        />
      </div>
    </div>
  );
};

const GameStateVisualizer = React.memo(GameStateVisualizerComponent);
GameStateVisualizer.displayName = 'GameStateVisualizer';

export default GameStateVisualizer;
