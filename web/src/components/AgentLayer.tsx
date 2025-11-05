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
      const houseRadius = 231; // Updated to match Town component (10% larger than original 210)
      return {
        x: centerX + houseRadius * Math.cos(houseAngle),
        y: centerY + houseRadius * Math.sin(houseAngle)
      };
    }
  };

  // Helper function to get agent position around a house
  const getAgentPositionAroundHouse = (houseCenterX: number, houseCenterY: number, agentId: number, spotRadius: number, totalAgents: number): { x: number, y: number } => {
    // Position agent at evenly spaced spots around the house perimeter
    const spotAngle = (agentId / totalAgents) * 2 * Math.PI;
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
  idX: number;
  idY: number;
  agentId: number;
  houseIndex: number;
  signal: number;
  action: number[];
      lied: boolean;
  }> = [];

    locations.forEach((houseIndex, agentId) => {
  let x, y, idX, idY, houseCenterX, houseCenterY;

  if (phase === 'day') {
    // During day: agents position near their target houses to signal destination
    const targetHouse = actions[agentId] ? actions[agentId][0] : houseIndex;
    const houseCenter = getHouseCenterPosition(targetHouse);
    houseCenterX = houseCenter.x;
    houseCenterY = houseCenter.y;
    const agentPos = getAgentPositionAroundHouse(houseCenterX, houseCenterY, agentId, 60, locations.length);
    x = agentPos.x;
    y = agentPos.y;
    // Position ID at larger radius along same angle
    const idAngle = (agentId / locations.length) * 2 * Math.PI;
    const idRadius = 60 + 20; // 20 units beyond agent position
    idX = houseCenterX + idRadius * Math.cos(idAngle);
    idY = houseCenterY + idRadius * Math.sin(idAngle);
  } else {
  // During night: agents move to their target houses in angular positions
  const targetHouse = actions[agentId] ? actions[agentId][0] : houseIndex;
  const houseCenter = getHouseCenterPosition(targetHouse);
  houseCenterX = houseCenter.x;
  houseCenterY = houseCenter.y;
  const agentPos = getAgentPositionAroundHouse(houseCenterX, houseCenterY, agentId, 60, locations.length);
    x = agentPos.x;
      y = agentPos.y;
        // Position ID at larger radius along same angle
        const idAngle = (agentId / locations.length) * 2 * Math.PI;
        const idRadius = 60 + 20; // 20 units beyond agent position
        idX = houseCenterX + idRadius * Math.cos(idAngle);
        idY = houseCenterY + idRadius * Math.sin(idAngle);
      }

      // Check if agent lied (signal doesn't match action)
      const agentAction = actions[agentId] || [houseIndex, 0];
      const lied = signals[agentId] !== agentAction[1];

      positions.push({
        x,
        y,
        idX,
        idY,
        agentId,
        houseIndex,
        signal: signals[agentId],
        action: agentAction,
        lied
      });
    });

    return positions;
  };

  const agentPositions = getAgentPositions();

  const getSignalSymbol = (signal: number) => {
    return signal === 1 ? '🚒' : '😴';
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
              r="14.4"
              className={`agent-dot ${phase === 'day' ? 'fill-yellow-400' : 'fill-slate-400'} ${phase === 'night' && agent.lied ? 'stroke-red-500 stroke-4' : 'stroke-white stroke-2'} cursor-pointer hover:stroke-blue-300`}
              style={{
                transition: 'cx 0.8s ease-in-out, cy 0.8s ease-in-out',
              }}
            >
              <title>{`Agent ${agent.agentId} at house ${agent.action[0]}`}</title>
            </circle>

            {/* Action emoji inside the agent dot */}
            <g
            transform={`translate(${agent.x}, ${agent.y})`}
            style={{
            transition: 'transform 0.8s ease-in-out'
            }}
            >
            <text
            x="0"
            y="0"
            textAnchor="middle"
              dominantBaseline="middle"
            className="select-none"
              style={{
                  fontSize: '0.9rem'
                }}
              >
                {phase === 'day' ? getSignalSymbol(agent.signal) : (agent.action[1] === 1 ? '🚒' : '😴')}
              </text>
            </g>

            {/* Agent ID label */}
            <g
              transform={`translate(${agent.idX}, ${agent.idY})`}
              style={{
                transition: 'transform 0.8s ease-in-out'
              }}
            >
            <text
              x="0"
                y="0"
                textAnchor="middle"
                dominantBaseline="middle"
                className="agent-id text-sm font-medium text-blue-600 fill-current select-none"
              >
                {agent.agentId}
              </text>
            </g>
          </g>
        ))}
      </svg>
    </div>
  );
};

export default AgentLayer;
