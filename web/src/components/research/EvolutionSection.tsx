import { memo } from 'react';
import { Info } from 'lucide-react';
import type { BestAgent, EvolutionTrace } from '../../types/research';

export interface EvolutionSectionProps {
  trace: EvolutionTrace;
  best: BestAgent;
  onShowInfo: () => void;
}

function EvolutionSectionImpl({ trace, best, onShowInfo }: EvolutionSectionProps) {
  const generations = trace.generations;
  const maxBestFit = Math.max(...generations.map((g) => g.best_fitness));
  const minBestFit = Math.min(...generations.map((g) => g.best_fitness));
  const minMeanFit = Math.min(...generations.map((g) => g.mean_fitness));

  return (
    <div className="mb-8">
      <div className="flex items-center gap-2 mb-4">
        <h2 className="text-2xl font-bold text-content-primary">Evolutionary Optimization</h2>
        <button
          onClick={onShowInfo}
          className="p-1 rounded-full hover:bg-surface-tertiary text-content-secondary hover:text-content-primary transition-colors"
          title="Technical details"
        >
          <Info className="w-5 h-5" />
        </button>
      </div>
      <div className="bg-surface-secondary p-6 rounded-lg shadow border border-outline-primary">
        <div className="grid md:grid-cols-3 gap-4 mb-6">
          <div>
            <div className="text-sm text-content-secondary">Best Fitness</div>
            <div className="text-2xl font-bold text-content-primary">
              {best.fitness.toFixed(2)}
            </div>
          </div>
          <div>
            <div className="text-sm text-content-secondary">Generation</div>
            <div className="text-2xl font-bold text-content-primary">{best.generation}</div>
          </div>
          <div>
            <div className="text-sm text-content-secondary">Time</div>
            <div className="text-2xl font-bold text-content-primary">
              {trace.convergence?.elapsed_time?.toFixed(1) ?? 'N/A'}s
            </div>
          </div>
        </div>

        {/* Evolution Progress Chart */}
        <div className="mb-6">
          <h4 className="text-lg font-semibold mb-2 text-content-primary">
            Fitness Over Generations
          </h4>
          <div className="relative h-48 bg-surface-primary rounded p-4">
            <svg className="w-full h-full" viewBox="0 0 400 150" preserveAspectRatio="none">
              {/* Grid lines */}
              {[0, 1, 2, 3, 4].map((i) => (
                <line
                  key={i}
                  x1="0"
                  y1={i * 37.5}
                  x2="400"
                  y2={i * 37.5}
                  stroke="currentColor"
                  strokeOpacity="0.1"
                />
              ))}

              {/* Best fitness line */}
              <polyline
                points={generations
                  .map((gen, idx) => {
                    const x = (idx / (generations.length - 1)) * 400;
                    const y =
                      150 -
                      ((gen.best_fitness - minBestFit) / (maxBestFit - minBestFit || 1)) * 150;
                    return `${x},${y}`;
                  })
                  .join(' ')}
                fill="none"
                stroke="rgb(59, 130, 246)"
                strokeWidth="2"
              />

              {/* Mean fitness line */}
              <polyline
                points={generations
                  .map((gen, idx) => {
                    const x = (idx / (generations.length - 1)) * 400;
                    const y =
                      150 -
                      ((gen.mean_fitness - minMeanFit) / (maxBestFit - minMeanFit || 1)) * 150;
                    return `${x},${y}`;
                  })
                  .join(' ')}
                fill="none"
                stroke="rgb(156, 163, 175)"
                strokeWidth="2"
                strokeDasharray="5,5"
              />
            </svg>
          </div>
          <div className="flex justify-center gap-6 mt-2 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-4 h-0.5 bg-blue-500" />
              <span className="text-content-secondary">Best Fitness</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-0.5 bg-gray-400 border-dashed" />
              <span className="text-content-secondary">Mean Fitness</span>
            </div>
          </div>
        </div>

        {/* Best Agent Parameters */}
        <div>
          <h4 className="text-lg font-semibold mb-3 text-content-primary">
            Best Agent Parameters
          </h4>
          <div className="grid grid-cols-2 gap-3">
            {Object.entries(best.parameters).map(([param, value]) => (
              <div key={param}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="capitalize text-content-secondary">
                    {param.replace(/_/g, ' ')}
                  </span>
                  <span className="font-mono text-content-primary">
                    {(value as number).toFixed(3)}
                  </span>
                </div>
                <div className="w-full bg-surface-tertiary rounded-full h-2">
                  <div
                    className="bg-orange-500 h-2 rounded-full"
                    style={{ width: `${(value as number) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export const EvolutionSection = memo(EvolutionSectionImpl);
