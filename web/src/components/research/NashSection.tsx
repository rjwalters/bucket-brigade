import { memo } from 'react';
import { Info } from 'lucide-react';
import type { NashResults } from '../../types/research';

export interface NashSectionProps {
  nash: NashResults;
  onShowInfo: () => void;
}

function NashSectionImpl({ nash, onShowInfo }: NashSectionProps) {
  return (
    <div className="mb-8">
      <div className="flex items-center gap-2 mb-4">
        <h2 className="text-2xl font-bold text-content-primary">Nash Equilibrium Analysis</h2>
        <button
          onClick={onShowInfo}
          className="p-1 rounded-full hover:bg-surface-tertiary text-content-secondary hover:text-content-primary transition-colors"
          title="Technical details"
        >
          <Info className="w-5 h-5" />
        </button>
      </div>

      {/* Key Metrics */}
      <div className="grid md:grid-cols-4 gap-4 mb-6">
        <div className="bg-surface-secondary p-6 rounded-lg shadow border border-outline-primary">
          <div className="text-sm text-content-secondary mb-1">Equilibrium Type</div>
          <div className="text-2xl font-bold capitalize text-content-primary">
            {nash.equilibrium.type}
          </div>
        </div>
        <div className="bg-surface-secondary p-6 rounded-lg shadow border border-outline-primary">
          <div className="text-sm text-content-secondary mb-1">Expected Payoff</div>
          <div className="text-2xl font-bold text-content-primary">
            {nash.equilibrium.expected_payoff.toFixed(2)}
          </div>
        </div>
        <div className="bg-surface-secondary p-6 rounded-lg shadow border border-outline-primary">
          <div className="text-sm text-content-secondary mb-1">Cooperation Rate</div>
          <div className="text-2xl font-bold text-green-600 dark:text-green-400">
            {(nash.interpretation.cooperation_rate * 100).toFixed(0)}%
          </div>
        </div>
        <div className="bg-surface-secondary p-6 rounded-lg shadow border border-outline-primary">
          <div className="text-sm text-content-secondary mb-1">Convergence Time</div>
          <div className="text-2xl font-bold text-content-primary">
            {nash.convergence?.elapsed_time?.toFixed(1) ?? 'N/A'}s
          </div>
        </div>
      </div>

      {/* Strategy Pool */}
      <div className="bg-surface-secondary p-6 rounded-lg shadow mb-6 border border-outline-primary">
        <h3 className="text-xl font-semibold mb-4 text-content-primary">Equilibrium Strategies</h3>
        <div className="space-y-6">
          {nash.equilibrium.strategy_pool.map((strategy) => (
            <div key={strategy.index} className="border-l-4 border-blue-500 pl-4">
              <div className="flex items-center justify-between mb-3">
                <div>
                  <h4 className="text-lg font-bold text-content-primary">
                    {strategy.classification}
                  </h4>
                  <div className="text-sm text-content-secondary">
                    Probability: {(strategy.probability * 100).toFixed(0)}%
                    {strategy.closest_archetype !== strategy.classification && (
                      <span className="ml-2">
                        (Closest to {strategy.closest_archetype}, distance:{' '}
                        {strategy.archetype_distance.toFixed(3)})
                      </span>
                    )}
                  </div>
                </div>
              </div>

              {/* Agent Parameters Visualization */}
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                {Object.entries(strategy.parameters).map(([param, value]) => (
                  <div key={param}>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="capitalize text-content-secondary">
                        {param.replace(/_/g, ' ')}
                      </span>
                      <span className="font-mono font-semibold text-content-primary">
                        {(value as number).toFixed(2)}
                      </span>
                    </div>
                    <div className="w-full bg-surface-tertiary rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${
                          (value as number) > 0.66
                            ? 'bg-green-500'
                            : (value as number) > 0.33
                              ? 'bg-yellow-500'
                              : 'bg-red-500'
                        }`}
                        style={{ width: `${(value as number) * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Algorithm Details */}
      <div className="bg-surface-secondary p-6 rounded-lg shadow border border-outline-primary">
        <h3 className="text-xl font-semibold mb-4 text-content-primary">Computation Details</h3>
        <div className="grid md:grid-cols-3 gap-4">
          <div>
            <div className="text-sm text-content-secondary">Algorithm</div>
            <div className="font-bold capitalize text-content-primary">
              {nash.algorithm.method.replace(/_/g, ' ')}
            </div>
          </div>
          <div>
            <div className="text-sm text-content-secondary">Simulations</div>
            <div className="font-bold text-content-primary">{nash.algorithm.num_simulations}</div>
          </div>
          <div>
            <div className="text-sm text-content-secondary">Iterations</div>
            <div className="font-bold text-content-primary">
              {nash.convergence?.iterations ?? 'N/A'} / {nash.algorithm.max_iterations}
            </div>
          </div>
          <div>
            <div className="text-sm text-content-secondary">Epsilon</div>
            <div className="font-bold text-content-primary">{nash.algorithm.epsilon}</div>
          </div>
          <div>
            <div className="text-sm text-content-secondary">Converged</div>
            <div
              className={`font-bold ${
                nash.convergence?.converged
                  ? 'text-green-600 dark:text-green-400'
                  : 'text-red-600 dark:text-red-400'
              }`}
            >
              {nash.convergence?.converged ? 'Yes' : 'No'}
            </div>
          </div>
          <div>
            <div className="text-sm text-content-secondary">Support Size</div>
            <div className="font-bold text-content-primary">{nash.equilibrium.support_size}</div>
          </div>
        </div>
      </div>
    </div>
  );
}

export const NashSection = memo(NashSectionImpl);
