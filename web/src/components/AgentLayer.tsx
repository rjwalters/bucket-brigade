import React from 'react';

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
  // Create positions for agents orbiting around houses
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
      // Position agents in a small orbit around their target house
      const angle = (houseIndex / 10) * 2 * Math.PI - Math.PI / 2;
      const centerX = 150;
      const centerY = 150;

      // Offset agents slightly from the house center
      const agentRadius = 35; // Slightly outside house radius
      const x = centerX + agentRadius * Math.cos(angle);
      const y = centerY + agentRadius * Math.sin(angle);

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
        {agentPositions.map((agent) => (
          <g
            key={`agent-${agent.agentId}`}
            className="agent-group"
            data-agent-id={agent.agentId}
          >
            {/* Agent dot */}
            <circle
              cx={agent.x}
              cy={agent.y}
              r="8"
              className="agent-dot fill-blue-500 stroke-white stroke-2 cursor-pointer hover:stroke-blue-300 transition-all"
            >
              <title>{`Agent ${agent.agentId} at house ${agent.houseIndex}`}</title>
            </circle>

            {/* Signal indicator */}
            <text
              x={agent.x}
              y={agent.y - 12}
              textAnchor="middle"
              className={`agent-signal text-sm select-none ${getSignalClass(agent.signal)}`}
            >
              {getSignalSymbol(agent.signal)}
            </text>

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
