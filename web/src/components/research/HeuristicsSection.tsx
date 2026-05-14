import { memo } from 'react';
import type { HeuristicResults } from '../../types/research';

export interface HeuristicsSectionProps {
  heuristics: HeuristicResults;
}

function HeuristicsSectionImpl({ heuristics }: HeuristicsSectionProps) {
  if (!heuristics.ranking) return null;

  const homogeneous = heuristics.ranking.homogeneous;
  const mixed = heuristics.ranking.mixed;
  const showHomogeneous = homogeneous && homogeneous.length > 0;
  const showMixed = mixed && mixed.length > 0;

  if (!showHomogeneous && !showMixed) return null;

  return (
    <div className="mb-8">
      <h2 className="text-2xl font-bold mb-4 text-content-primary">Heuristic Archetypes</h2>
      <div className="grid md:grid-cols-2 gap-6">
        {showHomogeneous && (
          <div className="bg-surface-secondary p-6 rounded-lg shadow border border-outline-primary">
            <h3 className="text-xl font-semibold mb-4 text-content-primary">Homogeneous Teams</h3>
            <div className="space-y-3">
              {homogeneous.map((team) => {
                const maxPayoff = Math.max(...homogeneous.map((t) => t.mean_payoff));
                return (
                  <div key={team.name}>
                    <div className="flex justify-between mb-1">
                      <span className="font-medium capitalize text-content-primary">
                        {team.name.replace(/_/g, ' ')}
                      </span>
                      <span className="font-bold text-content-primary">
                        {team.mean_payoff.toFixed(1)}
                      </span>
                    </div>
                    <div className="w-full bg-surface-tertiary rounded-full h-3">
                      <div
                        className="bg-green-500 h-3 rounded-full"
                        style={{
                          width: `${(team.mean_payoff / maxPayoff) * 100}%`,
                        }}
                      />
                    </div>
                    <div className="text-xs text-content-secondary mt-1">
                      ±{team.std_payoff.toFixed(1)}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {showMixed && (
          <div className="bg-surface-secondary p-6 rounded-lg shadow border border-outline-primary">
            <h3 className="text-xl font-semibold mb-4 text-content-primary">Mixed Teams</h3>
            <div className="space-y-3">
              {mixed.slice(0, 5).map((team) => {
                const maxPayoff = Math.max(...mixed.map((t) => t.mean_payoff));
                return (
                  <div key={team.composition}>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm text-content-primary">{team.composition}</span>
                      <span className="font-bold text-content-primary">
                        {team.mean_payoff.toFixed(1)}
                      </span>
                    </div>
                    <div className="w-full bg-surface-tertiary rounded-full h-3">
                      <div
                        className="bg-purple-500 h-3 rounded-full"
                        style={{
                          width: `${(team.mean_payoff / maxPayoff) * 100}%`,
                        }}
                      />
                    </div>
                    <div className="text-xs text-content-secondary mt-1">
                      ±{team.std_payoff.toFixed(1)}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export const HeuristicsSection = memo(HeuristicsSectionImpl);
