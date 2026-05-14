import { memo } from 'react';
import { X } from 'lucide-react';

export interface NashInfoModalProps {
  open: boolean;
  onClose: () => void;
}

function NashInfoModalImpl({ open, onClose }: NashInfoModalProps) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4 modal-overlay">
      <div className="bg-surface-primary rounded-lg shadow-xl max-w-3xl w-full max-h-[90vh] overflow-y-auto modal-content">
        <div className="sticky top-0 bg-surface-primary border-b border-outline-primary p-6 flex items-center justify-between">
          <h3 className="text-2xl font-bold text-content-primary">
            Nash Equilibrium Analysis: Technical Details
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
              Symmetric Game Formulation
            </h4>
            <p className="text-content-secondary mb-3">
              We model the Bucket Brigade as a symmetric multi-agent game where all players share
              the same strategy space Θ (the 10-dimensional parameter space). A Nash equilibrium
              is a probability distribution over strategies such that no player can improve their
              expected payoff by unilaterally deviating.
            </p>
            <div className="bg-surface-tertiary p-4 rounded-lg font-mono text-sm mb-3">
              π* ∈ arg max_π 𝔼<sub>θ~π</sub>[u(θ, π*)]
            </div>
            <p className="text-content-secondary text-sm">
              where u(θ, π) is the expected payoff when playing strategy θ against population
              distribution π.
            </p>
          </div>

          <div>
            <h4 className="text-lg font-semibold mb-3 text-content-primary">
              Support Enumeration Method
            </h4>
            <p className="text-content-secondary mb-3">
              We use a support enumeration algorithm that searches over candidate support sets S
              ⊂ Θ. For each support size k, we:
            </p>
            <ol className="list-decimal list-inside space-y-2 text-content-secondary text-sm mb-3">
              <li>Sample k strategies from the parameter space</li>
              <li>
                Compute the payoff matrix A where A<sub>ij</sub> = u(θᵢ, θⱼ)
              </li>
              <li>Solve for the mixed strategy equilibrium over these k strategies</li>
              <li>
                Verify the equilibrium condition: all support strategies have equal payoff and
                all non-support strategies have weakly lower payoff
              </li>
            </ol>
          </div>

          <div>
            <h4 className="text-lg font-semibold mb-3 text-content-primary">
              Payoff Estimation via Monte Carlo
            </h4>
            <p className="text-content-secondary mb-3">
              Each payoff matrix entry requires estimating the expected payoff of strategy θᵢ
              when playing against θⱼ:
            </p>
            <div className="bg-surface-tertiary p-4 rounded-lg font-mono text-sm mb-3">
              u(θᵢ, θⱼ) ≈ (1/M) Σₘ R<sub>m</sub>(θᵢ | opponent=θⱼ)
            </div>
            <p className="text-content-secondary text-sm">
              where M is the number of Monte Carlo simulations per strategy pair (typically
              1000).
            </p>
          </div>

          <div>
            <h4 className="text-lg font-semibold mb-3 text-content-primary">
              Linear Programming Solution
            </h4>
            <p className="text-content-secondary mb-3">
              Given a candidate support S with payoff matrix A, we find the equilibrium
              probabilities by solving:
            </p>
            <div className="bg-surface-tertiary p-4 rounded-lg space-y-2 font-mono text-xs mb-3">
              <div>maximize: v</div>
              <div>subject to: Aᵀp ≥ v·1</div>
              <div className="ml-16">1ᵀp = 1</div>
              <div className="ml-16">p ≥ 0</div>
            </div>
            <p className="text-content-secondary text-sm">
              where p is the probability distribution over the support and v is the equilibrium
              payoff.
            </p>
          </div>

          <div>
            <h4 className="text-lg font-semibold mb-3 text-content-primary">
              Equilibrium Classification
            </h4>
            <p className="text-content-secondary mb-3">
              We classify equilibria based on their support size and strategic characteristics:
            </p>
            <ul className="list-disc list-inside space-y-1 text-content-secondary text-sm">
              <li>
                <strong>Pure equilibrium</strong>: Single strategy with probability 1.0
              </li>
              <li>
                <strong>Mixed equilibrium</strong>: Multiple strategies in support with varying
                probabilities
              </li>
              <li>
                <strong>Cooperation rate</strong>: Expected probability of working (weighted by
                strategy probabilities and work_tendency parameters)
              </li>
            </ul>
          </div>

          <div>
            <h4 className="text-lg font-semibold mb-3 text-content-primary">
              Convergence Guarantees
            </h4>
            <p className="text-content-secondary">
              For finite games with symmetric strategy spaces, Nash equilibria are guaranteed to
              exist (by Brouwer's fixed-point theorem). Our algorithm systematically explores
              support sizes k=1,2,...,K<sub>max</sub> and verifies equilibrium conditions
              numerically with tolerance ε=0.001.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export const NashInfoModal = memo(NashInfoModalImpl);
