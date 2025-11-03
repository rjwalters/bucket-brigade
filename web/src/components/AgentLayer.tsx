import React, { useState, useEffect } from 'react';

interface AgentLayerProps {
  locations: number[]; // Agent positions (house indices)
  signals: number[]; // Agent signals (0=REST, 1=WORK)
  actions: number[][]; // Agent actions [[house, mode], ...]
  className?: string;
}

const AgentLayer: React.FC<AgentLayerProps> = ({
  locations,
  signals,
  actions,
  className = ''
}) => {
  // Day/night phase animation state
  const [phase, setPhase] = useState<'day' | 'night'>('day');
  const [animationKey, setAnimationKey] = useState(0);

  // Reset animation when locations change (new night)
  useEffect(() => {
    setPhase('day');
    setAnimationKey(prev => prev + 1);

    // Transition to night phase after day phase animation
    const dayTimer = setTimeout(() => {
      setPhase('night');
    }, 1000); // 1 second for day phase

    return () => clearTimeout(dayTimer);
  }, [locations, signals, actions]);

  const centerX = 150;
  const centerY = 150;

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
        // During day: agents gather at center in a small circle
        const numAgents = locations.length;
        const agentAngle = (agentId / numAgents) * 2 * Math.PI;
        const gatherRadius = 20; // Small circle at center
        x = centerX + gatherRadius * Math.cos(agentAngle);
        y = centerY + gatherRadius * Math.sin(agentAngle);
      } else {
        // During night: agents move to their target houses
        const targetHouse = actions[agentId] ? actions[agentId][0] : houseIndex;
        const angle = (targetHouse / 10) * 2 * Math.PI - Math.PI / 2;
        const houseRadius = 120;
        x = centerX + houseRadius * Math.cos(angle);
        y = centerY + houseRadius * Math.sin(angle);
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
      <svg width="300" height="300" className="agent-svg absolute inset-0">
        {/* Day/Night Indicator */}
        <g>
          <text
            x={centerX}
            y="20"
            textAnchor="middle"
            className="text-2xl select-none"
          >
            {phase === 'day' ? '☀️' : '🌙'}
          </text>
          <text
            x={centerX}
            y="38"
            textAnchor="middle"
            className="text-xs fill-gray-600"
          >
            {phase === 'day' ? 'Day: Signaling' : 'Night: Working'}
          </text>
        </g>

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
              r="8"
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
                y={agent.y - 12}
                textAnchor="middle"
                className={`agent-signal text-sm select-none ${getSignalClass(agent.signal)}`}
                style={{ transition: 'x 0.8s ease-in-out, y 0.8s ease-in-out' }}
              >
                {getSignalSymbol(agent.signal)}
              </text>
            )}

            {/* Action indicator - only visible during night phase */}
            {phase === 'night' && (
              <text
                x={agent.x}
                y={agent.y - 12}
                textAnchor="middle"
                className="text-sm select-none"
                style={{ transition: 'x 0.8s ease-in-out, y 0.8s ease-in-out' }}
              >
                {agent.action[1] === 1 ? '🚒' : '😴'}
              </text>
            )}

            {/* Agent ID label */}
            <text
              x={agent.x}
              y={agent.y + 16}
              textAnchor="middle"
              className="agent-id text-xs font-medium text-blue-600 fill-current select-none"
            >
              {agent.agentId}
            </text>
          </g>
        ))}
      </svg>

      {/* House hover information */}
      <div className="house-info absolute inset-0 pointer-events-none">
        {Object.entries(agentsByHouse).map(([houseIndex, agents]) => {
          const houseAngle = (parseInt(houseIndex) / 10) * 2 * Math.PI - Math.PI / 2;
          const infoX = 150 + 160 * Math.cos(houseAngle);
          const infoY = 150 + 160 * Math.sin(houseAngle);

          const workingAgents = agents.filter(a => a.action[1] === 1);
          const restingAgents = agents.filter(a => a.action[1] === 0);

          if (agents.length === 0) return null;

          return (
            <div
              key={`house-info-${houseIndex}`}
              className="house-info-popup absolute bg-black bg-opacity-75 text-white text-xs rounded px-2 py-1 pointer-events-auto opacity-0 hover:opacity-100 transition-opacity"
              style={{
                left: infoX,
                top: infoY,
                transform: 'translate(-50%, -50%)'
              }}
            >
              <div>House {houseIndex}</div>
              {workingAgents.length > 0 && (
                <div>Working: {workingAgents.map(a => a.agentId).join(', ')}</div>
              )}
              {restingAgents.length > 0 && (
                <div>Resting: {restingAgents.map(a => a.agentId).join(', ')}</div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default AgentLayer;
