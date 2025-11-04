import React, { useState, useEffect } from 'react';

interface AgentLayerProps {
  locations: number[]; // Agent positions (house indices)
  signals: number[]; // Agent signals (0=REST, 1=WORK)
  actions: number[][]; // Agent actions [[house, mode], ...]
  className?: string;
  onPhaseChange?: (phase: 'day' | 'night') => void;
}

const AgentLayer: React.FC<AgentLayerProps> = ({
  locations,
  signals,
  actions,
  className = '',
  onPhaseChange
}) => {
  // Day/night phase animation state
  const [phase, setPhase] = useState<'day' | 'night'>('day');
  const [animationKey, setAnimationKey] = useState(0);

  // Reset animation when locations change (new night)
  useEffect(() => {
    setPhase('day');
    if (onPhaseChange) onPhaseChange('day');
    setAnimationKey(prev => prev + 1);

    // Transition to night phase after day phase animation
    const dayTimer = setTimeout(() => {
      setPhase('night');
      if (onPhaseChange) onPhaseChange('night');
    }, 1000); // 1 second for day phase

    return () => clearTimeout(dayTimer);
  }, [locations, signals, actions, onPhaseChange]);

  const centerX = 320; // Updated to match Town component (symmetric)
  const centerY = 320; // Updated to match Town component (symmetric)

  // Helper function to get house center position (including "ghost house" at center for day phase)
  const getHouseCenterPosition = (houseIndex: number): { x: number, y: number } => {
    if (houseIndex === -1) {
      // Ghost house at center for day phase signaling
      return { x: centerX, y: centerY };
    } else {
      // Regular houses in a circle
      const houseAngle = (houseIndex / 10) * 2 * Math.PI - Math.PI / 2;
      const houseRadius = 210; // Updated to match Town component (5% larger)
      return {
        x: centerX + houseRadius * Math.cos(houseAngle),
        y: centerY + houseRadius * Math.sin(houseAngle)
      };
    }
  };

  // Helper function to get agent position around a house
  const getAgentPositionAroundHouse = (houseCenterX: number, houseCenterY: number, agentId: number, spotRadius: number): { x: number, y: number } => {
    // Position agent at one of 10 spots around the house perimeter (36 degrees apart)
    const spotAngle = (agentId / 10) * 2 * Math.PI;
    return {
      x: houseCenterX + spotRadius * Math.cos(spotAngle),
      y: houseCenterY + spotRadius * Math.sin(spotAngle)
    };
  };

  // Create positions for agents
  const getAgentPositions = () => {
    const positions: Array<{
      x: number;
      y: number;
      agentId: number;
      houseIndex: number;
      signal: number;
      action: number[];
    }> = [];

    locations.forEach((houseIndex, agentId) => {
      let x, y;

      if (phase === 'day') {
        // During day: agents gather at center "ghost house" in angular positions
        const houseCenter = getHouseCenterPosition(-1); // -1 for center ghost house
        const agentPos = getAgentPositionAroundHouse(houseCenter.x, houseCenter.y, agentId, 35);
        x = agentPos.x;
        y = agentPos.y;
      } else {
        // During night: agents move to their target houses in angular positions
        const targetHouse = actions[agentId] ? actions[agentId][0] : houseIndex;
        const houseCenter = getHouseCenterPosition(targetHouse);
        const agentPos = getAgentPositionAroundHouse(houseCenter.x, houseCenter.y, agentId, 35);
        x = agentPos.x;
        y = agentPos.y;
      }

      positions.push({
        x,
        y,
        agentId,
        houseIndex,
        signal: signals[agentId],
        action: actions[agentId] || [houseIndex, 0]
      });
    });

    return positions;
  };

  const agentPositions = getAgentPositions();

  const getSignalSymbol = (signal: number) => {
    return signal === 1 ? '🔥' : '💤';
  };

  const getSignalClass = (signal: number) => {
    return signal === 1 ? 'signal-work' : 'signal-rest';
  };

  // Group agents by house for hover information
  const agentsByHouse = agentPositions.reduce((acc, agent) => {
    if (!acc[agent.houseIndex]) {
      acc[agent.houseIndex] = [];
    }
    acc[agent.houseIndex].push(agent);
    return acc;
  }, {} as Record<number, typeof agentPositions>);

  return (
    <div className={`agent-layer ${className}`}>
      <svg width="640" height="640" viewBox="0 0 640 640" className="agent-svg absolute inset-0">

        {agentPositions.map((agent) => (
          <g
            key={`agent-${agent.agentId}-${animationKey}`}
            className="agent-group"
            data-agent-id={agent.agentId}
          >
            {/* Agent dot with smooth animation */}
            <circle
              cx={agent.x}
              cy={agent.y}
              r="12"
              className="agent-dot fill-blue-500 stroke-white stroke-2 cursor-pointer hover:stroke-blue-300"
              style={{
                transition: 'cx 0.8s ease-in-out, cy 0.8s ease-in-out',
              }}
            >
              <title>{`Agent ${agent.agentId} at house ${agent.action[0]}`}</title>
            </circle>

            {/* Signal indicator - only visible during day phase */}
            {phase === 'day' && (
              <text
                x={agent.x}
                y={agent.y - 18}
                textAnchor="middle"
                className={`agent-signal select-none ${getSignalClass(agent.signal)}`}
                style={{ transition: 'x 0.8s ease-in-out, y 0.8s ease-in-out', fontSize: '1.2rem' }}
              >
                {getSignalSymbol(agent.signal)}
              </text>
            )}

            {/* Action indicator - only visible during night phase */}
            {phase === 'night' && (
              <text
                x={agent.x}
                y={agent.y - 18}
                textAnchor="middle"
                className="select-none"
                style={{ transition: 'x 0.8s ease-in-out, y 0.8s ease-in-out', fontSize: '1.2rem' }}
              >
                {agent.action[1] === 1 ? '🚒' : '😴'}
              </text>
            )}

            {/* Agent ID label */}
            <text
              x={agent.x}
              y={agent.y + 20}
              textAnchor="middle"
              className="agent-id text-sm font-medium text-blue-600 fill-current select-none"
            >
              {agent.agentId}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
};

export default AgentLayer;
