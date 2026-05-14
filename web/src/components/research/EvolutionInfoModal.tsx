import { memo } from 'react';
import { X } from 'lucide-react';

export interface EvolutionInfoModalProps {
  open: boolean;
  onClose: () => void;
}

function EvolutionInfoModalImpl({ open, onClose }: EvolutionInfoModalProps) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4 modal-overlay">
      <div className="bg-surface-primary rounded-lg shadow-xl max-w-3xl w-full max-h-[90vh] overflow-y-auto modal-content">
        <div className="sticky top-0 bg-surface-primary border-b border-outline-primary p-6 flex items-center justify-between">
          <h3 className="text-2xl font-bold text-content-primary">
            Evolutionary Optimization: Technical Details
          </h3>
          <button
            onClick={onClose}
            className="p-2 rounded-full hover:bg-surface-tertiary transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        <div className="p-6 space-y-6">
          <div>
            <h4 className="text-lg font-semibold mb-3 text-content-primary">
              Genetic Algorithm Overview
            </h4>
            <p className="text-content-secondary mb-3">
              We employ a genetic algorithm to evolve agent strategies that maximize expected
              payoff in the Bucket Brigade game. Each agent is represented by a 10-dimensional
              parameter vector (genome) that encodes behavioral traits.
            </p>
            <p className="text-content-secondary">
              The algorithm maintains a population of candidate strategies and iteratively
              improves them through selection, crossover, and mutation operators inspired by
              biological evolution.
            </p>
          </div>

          <div>
            <h4 className="text-lg font-semibold mb-3 text-content-primary">Fitness Function</h4>
            <p className="text-content-secondary mb-3">
              The fitness of an agent strategy θ is measured as the expected reward over
              multiple game simulations:
            </p>
            <div className="bg-surface-tertiary p-4 rounded-lg font-mono text-sm mb-3">
              f(θ) = 𝔼[R(θ)] = (1/N) Σᵢ Rᵢ(θ)
            </div>
            <p className="text-content-secondary text-sm">
              where N is the number of evaluation games and Rᵢ(θ) is the total reward in game i.
            </p>
          </div>

          <div>
            <h4 className="text-lg font-semibold mb-3 text-content-primary">
              Evolutionary Operators
            </h4>
            <div className="space-y-3">
              <div>
                <h5 className="font-semibold text-content-primary mb-1">Selection (Tournament)</h5>
                <p className="text-content-secondary text-sm">
                  Parents are selected via tournament selection with size k=3. We randomly sample
                  k individuals and select the one with highest fitness. This balances exploration
                  and exploitation.
                </p>
              </div>
              <div>
                <h5 className="font-semibold text-content-primary mb-1">Crossover (Uniform)</h5>
                <p className="text-content-secondary text-sm mb-2">
                  Two parents θ₁, θ₂ produce offspring θ' where each parameter is inherited from
                  parent 1 with probability p=0.5:
                </p>
                <div className="bg-surface-tertiary p-3 rounded-lg font-mono text-xs">
                  θ'ᵢ = θ₁ᵢ if rand() &lt; 0.5, else θ₂ᵢ
                </div>
              </div>
              <div>
                <h5 className="font-semibold text-content-primary mb-1">Mutation (Gaussian)</h5>
                <p className="text-content-secondary text-sm mb-2">
                  Each parameter is mutated with probability pₘ=0.1 by adding Gaussian noise:
                </p>
                <div className="bg-surface-tertiary p-3 rounded-lg font-mono text-xs mb-2">
                  θ'ᵢ = clip(θᵢ + 𝒩(0, σ²), 0, 1)
                </div>
                <p className="text-content-secondary text-sm">
                  where σ=0.1 is the mutation scale and clip ensures parameters remain in [0,1].
                </p>
              </div>
            </div>
          </div>

          <div>
            <h4 className="text-lg font-semibold mb-3 text-content-primary">Elitism</h4>
            <p className="text-content-secondary">
              The top 5 individuals are preserved unchanged across generations, ensuring monotonic
              improvement in best fitness and preventing loss of good solutions due to stochastic
              noise.
            </p>
          </div>

          <div>
            <h4 className="text-lg font-semibold mb-3 text-content-primary">
              Convergence Criteria
            </h4>
            <p className="text-content-secondary">
              Evolution terminates after a fixed number of generations (typically 15,000). We
              track population diversity using the average pairwise Euclidean distance between
              genomes to monitor convergence.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export const EvolutionInfoModal = memo(EvolutionInfoModalImpl);
