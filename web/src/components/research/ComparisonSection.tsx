import { memo } from 'react';
import { Info } from 'lucide-react';
import type { ComparisonResults, NashResults } from '../../types/research';
import type { ArchetypeParams } from '../../data/archetypes.generated';
import { AgentRadarChart } from '../AgentRadarChart';
import { genomeToParams } from '../../utils/genome';

export interface ComparisonSectionProps {
  comparison: ComparisonResults;
  nash?: NashResults;
}

function ComparisonSectionImpl({ comparison, nash }: ComparisonSectionProps) {
  const maxRankingPayoff = Math.max(...comparison.ranking.map((r) => r.mean_payoff));

  return (
    <div className="mb-8">
      <h2 className="text-2xl font-bold mb-4 text-content-primary">Strategy Comparison</h2>

      {/* Agent Profiles - Radar Charts */}
      <div className="mb-6 bg-surface-secondary p-6 rounded-lg shadow border border-outline-primary">
        <h3 className="text-xl font-semibold mb-4 text-content-primary">Strategy Profiles</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {comparison.ranking.map((result, idx) => {
            const strategyGenome = comparison.strategies[result.name];
            const params = genomeToParams(strategyGenome);
            return (
              <div key={result.name} className="text-center">
                <div className="flex items-center justify-center gap-2 mb-2">
                  <div className="text-xl font-bold text-content-tertiary">#{idx + 1}</div>
                  <h4 className="text-lg font-semibold capitalize text-content-primary">
                    {result.name.replace(/_/g, ' ')}
                  </h4>
                </div>
                <div className="text-sm text-content-secondary mb-2">
                  Avg: {result.mean_payoff.toFixed(2)} ± {result.std_payoff.toFixed(2)}
                </div>
                <AgentRadarChart params={params} />
              </div>
            );
          })}
          {/* Add Nash equilibrium if available */}
          {nash && nash.equilibrium.support_size === 1 && (
            <div className="text-center">
              <div className="flex items-center justify-center gap-2 mb-2">
                <div className="text-xl font-bold text-content-tertiary">
                  #{comparison.ranking.length + 1}
                </div>
                <h4 className="text-lg font-semibold capitalize text-content-primary">
                  Nash Equilibrium
                </h4>
              </div>
              <div className="text-sm text-content-secondary mb-2">
                Payoff: {nash.equilibrium.expected_payoff.toFixed(2)}
              </div>
              <AgentRadarChart
                params={nash.equilibrium.strategy_pool[0].parameters as ArchetypeParams}
              />
            </div>
          )}
          {/* Placeholder for third strategy if needed */}
          {comparison.ranking.length < 2 && !nash && (
            <div className="text-center flex flex-col items-center justify-center min-h-[300px] bg-surface-tertiary rounded-lg border-2 border-dashed border-outline-primary">
              <div className="text-content-tertiary mb-2">
                <Info className="w-12 h-12 opacity-30" />
              </div>
              <h4 className="text-lg font-semibold text-content-secondary">Additional Strategy</h4>
              <p className="text-sm text-content-tertiary mt-1">Coming Soon</p>
            </div>
          )}
        </div>
      </div>

      <div className="bg-surface-secondary p-6 rounded-lg shadow border border-outline-primary">
        <h3 className="text-xl font-semibold mb-4 text-content-primary">Tournament Results</h3>
        <div className="space-y-4">
          {comparison.ranking.map((result, idx) => (
            <div key={result.name} className="flex items-center gap-4">
              <div className="text-2xl font-bold text-content-tertiary w-8">#{idx + 1}</div>
              <div className="flex-1">
                <div className="flex justify-between items-center mb-1">
                  <span className="font-semibold capitalize text-content-primary">
                    {result.name.replace(/_/g, ' ')}
                  </span>
                  <span className="text-lg font-bold text-content-primary">
                    {result.mean_payoff.toFixed(2)}
                  </span>
                </div>
                <div className="w-full bg-surface-tertiary rounded-full h-4">
                  <div
                    className="bg-blue-500 h-4 rounded-full"
                    style={{
                      width: `${(result.mean_payoff / maxRankingPayoff) * 100}%`,
                    }}
                  />
                </div>
                <div className="text-sm text-content-secondary mt-1">
                  ±{result.std_payoff.toFixed(2)}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Strategy Distance */}
        {Object.keys(comparison.distances).length > 0 && (
          <div className="mt-6">
            <h4 className="text-lg font-semibold mb-2 text-content-primary">Strategy Distances</h4>
            <div className="space-y-2">
              {Object.entries(comparison.distances).map(([key, distance]) => (
                <div key={key} className="flex justify-between text-sm">
                  <span className="capitalize text-content-secondary">
                    {key.replace(/_/g, ' ').replace(/vs/g, '↔')}
                  </span>
                  <span className="font-mono text-content-primary">{distance.toFixed(3)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export const ComparisonSection = memo(ComparisonSectionImpl);
