import { useState, useEffect } from 'react';
import { Info, X } from 'lucide-react';
import {
  SCENARIOS,
  type ScenarioName,
  type HeuristicResults,
  type EvolutionTrace,
  type BestAgent,
  type ComparisonResults,
  type NashResults,
} from '../types/research';
import { AgentRadarChart } from '../components/AgentRadarChart';
import type { ArchetypeParams } from '../data/archetypes.generated';

interface ScenarioData {
  config: any;
  heuristics?: HeuristicResults;
  evolution?: {
    trace: EvolutionTrace;
    best: BestAgent;
  };
  nash?: NashResults;
  comparison?: ComparisonResults;
}

// Convert genome array to ArchetypeParams object
// Genome order: [honesty_bias, work_tendency, neighbor_help_bias, own_house_priority,
//                risk_aversion, coordination_weight, exploration_rate, fatigue_memory,
//                rest_reward_bias, altruism_factor]
function genomeToParams(genome: number[]): ArchetypeParams {
  return {
    honesty_bias: genome[0],
    work_tendency: genome[1],
    neighbor_help_bias: genome[2],
    own_house_priority: genome[3],
    risk_aversion: genome[4],
    coordination_weight: genome[5],
    exploration_rate: genome[6],
    fatigue_memory: genome[7],
    rest_reward_bias: genome[8],
    altruism_factor: genome[9],
  };
}

export default function ScenarioResearch() {
  const [selectedScenario, setSelectedScenario] = useState<ScenarioName>('greedy_neighbor');
  // Removed tab state - now showing all sections stacked vertically
  const [data, setData] = useState<ScenarioData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showEvolutionInfo, setShowEvolutionInfo] = useState(false);
  const [showNashInfo, setShowNashInfo] = useState(false);

  useEffect(() => {
    loadScenarioData(selectedScenario);
  }, [selectedScenario]);

  const loadScenarioData = async (scenario: ScenarioName) => {
    setLoading(true);
    setError(null);

    try {
      const basePath = import.meta.env.BASE_URL || '/';
      const scenarioPath = `${basePath}research/scenarios/${scenario}`;

      // Load all data in parallel
      const [config, nash, heuristics, evolutionTrace, evolutionBest, comparison] = await Promise.all([
        fetch(`${scenarioPath}/config.json`).then(r => r.ok ? r.json() : null),
        fetch(`${scenarioPath}/nash/equilibrium.json`).then(r => r.ok ? r.json() : null),
        fetch(`${scenarioPath}/heuristics/results.json`).then(r => r.ok ? r.json() : null),
        fetch(`${scenarioPath}/evolved/evolution_trace.json`).then(r => r.ok ? r.json() : null),
        fetch(`${scenarioPath}/evolved/best_agent.json`).then(r => r.ok ? r.json() : null),
        fetch(`${scenarioPath}/comparison/comparison.json`).then(r => r.ok ? r.json() : null),
      ]);

      setData({
        config,
        nash,
        heuristics,
        evolution: evolutionTrace && evolutionBest ? { trace: evolutionTrace, best: evolutionBest } : undefined,
        comparison,
      });
    } catch (err) {
      setError(`Failed to load scenario data: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="text-center">
          <div className="text-xl text-content-primary">Loading scenario data...</div>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="text-center text-red-500 dark:text-red-400">
          <div className="text-xl">{error || 'Failed to load data'}</div>
        </div>
      </div>
    );
  }

  return (
  <div className="container mx-auto px-4 py-8">
  <h1 className="text-4xl font-bold mb-8 text-content-primary">Scenario Research</h1>

  {/* Scenario Selector */}
  <div className="mb-8">
  <label className="block text-sm font-medium mb-2 text-content-primary">Select Scenario</label>
  <select
  value={selectedScenario}
  onChange={(e) => setSelectedScenario(e.target.value as ScenarioName)}
  className="w-full md:w-auto px-4 py-2 border border-outline-primary rounded-lg bg-surface-secondary text-content-primary"
  >
      {SCENARIOS.map((scenario) => (
            <option key={scenario} value={scenario}>
              {scenario.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
            </option>
          ))}
        </select>
      </div>

      {/* Scenario Overview */}
      {data.config && (
        <div className="mb-8 p-6 bg-surface-tertiary rounded-lg">
          <h2 className="text-2xl font-bold mb-4 text-content-primary">{data.config.description}</h2>
          <p className="text-content-secondary mb-4">{data.config.story}</p>

          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div>
              <div className="text-sm text-content-secondary">Fire Spread (β)</div>
              <div className="text-xl font-bold text-content-primary">{data.config.parameters.beta}</div>
            </div>
            <div>
              <div className="text-sm text-content-secondary">Extinguish (κ)</div>
              <div className="text-xl font-bold text-content-primary">{data.config.parameters.kappa}</div>
            </div>
            <div>
              <div className="text-sm text-content-secondary">Reward/A (saved)</div>
              <div className="text-xl font-bold text-content-primary">{data.config.parameters.A}</div>
            </div>
            <div>
              <div className="text-sm text-content-secondary">Penalty/L (ruined)</div>
              <div className="text-xl font-bold text-content-primary">{data.config.parameters.L}</div>
            </div>
            <div>
              <div className="text-sm text-content-secondary">Work Cost (c)</div>
              <div className="text-xl font-bold text-content-primary">{data.config.parameters.c}</div>
            </div>
            <div>
              <div className="text-sm text-content-secondary">Ignition Prob (ρ)</div>
              <div className="text-xl font-bold text-content-primary">{data.config.parameters.rho_ignite}</div>
            </div>
            <div>
              <div className="text-sm text-content-secondary">Min Nights</div>
              <div className="text-xl font-bold text-content-primary">{data.config.parameters.N_min}</div>
            </div>
            <div>
              <div className="text-sm text-content-secondary">Spark Prob</div>
              <div className="text-xl font-bold text-content-primary">{data.config.parameters.p_spark}</div>
            </div>
            <div>
              <div className="text-sm text-content-secondary">Spark Duration</div>
              <div className="text-xl font-bold text-content-primary">{data.config.parameters.N_spark}</div>
            </div>
            <div>
              <div className="text-sm text-content-secondary">Agents</div>
              <div className="text-xl font-bold text-content-primary">{data.config.parameters.num_agents}</div>
            </div>
          </div>
        </div>
      )}

      {/* Comparison Results - Show this first as it's the most important */}
      {data.comparison && (
        <div className="mb-8">
          <h2 className="text-2xl font-bold mb-4 text-content-primary">Strategy Comparison</h2>

          {/* Agent Profiles - Radar Charts */}
          <div className="mb-6 bg-surface-secondary p-6 rounded-lg shadow border border-outline-primary">
            <h3 className="text-xl font-semibold mb-4 text-content-primary">Strategy Profiles</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {data.comparison.ranking.map((result, idx) => {
                const strategyGenome = data.comparison!.strategies[result.name];
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
              {data.nash && data.nash.equilibrium.support_size === 1 && (
                <div className="text-center">
                  <div className="flex items-center justify-center gap-2 mb-2">
                    <div className="text-xl font-bold text-content-tertiary">#{data.comparison.ranking.length + 1}</div>
                    <h4 className="text-lg font-semibold capitalize text-content-primary">
                      Nash Equilibrium
                    </h4>
                  </div>
                  <div className="text-sm text-content-secondary mb-2">
                    Payoff: {data.nash.equilibrium.expected_payoff.toFixed(2)}
                  </div>
                  <AgentRadarChart params={data.nash.equilibrium.strategy_pool[0].parameters as ArchetypeParams} />
                </div>
              )}
              {/* Placeholder for third strategy if needed */}
              {data.comparison.ranking.length < 2 && !data.nash && (
                <div className="text-center flex flex-col items-center justify-center min-h-[300px] bg-surface-tertiary rounded-lg border-2 border-dashed border-outline-primary">
                  <div className="text-content-tertiary mb-2">
                    <Info className="w-12 h-12 opacity-30" />
                  </div>
                  <h4 className="text-lg font-semibold text-content-secondary">
                    Additional Strategy
                  </h4>
                  <p className="text-sm text-content-tertiary mt-1">
                    Coming Soon
                  </p>
                </div>
              )}
            </div>
          </div>

          <div className="bg-surface-secondary p-6 rounded-lg shadow border border-outline-primary">
            <h3 className="text-xl font-semibold mb-4 text-content-primary">Tournament Results</h3>
            <div className="space-y-4">
              {data.comparison.ranking.map((result, idx) => (
                <div key={result.name} className="flex items-center gap-4">
                  <div className="text-2xl font-bold text-content-tertiary w-8">#{idx + 1}</div>
                  <div className="flex-1">
                    <div className="flex justify-between items-center mb-1">
                      <span className="font-semibold capitalize text-content-primary">{result.name.replace(/_/g, ' ')}</span>
                      <span className="text-lg font-bold text-content-primary">{result.mean_payoff.toFixed(2)}</span>
                    </div>
                    <div className="w-full bg-surface-tertiary rounded-full h-4">
                      <div
                        className="bg-blue-500 h-4 rounded-full"
                        style={{
                          width: `${(result.mean_payoff / Math.max(...data.comparison!.ranking.map(r => r.mean_payoff))) * 100}%`
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
            {Object.keys(data.comparison.distances).length > 0 && (
              <div className="mt-6">
                <h4 className="text-lg font-semibold mb-2 text-content-primary">Strategy Distances</h4>
                <div className="space-y-2">
                  {Object.entries(data.comparison.distances).map(([key, distance]) => (
                    <div key={key} className="flex justify-between text-sm">
                      <span className="capitalize text-content-secondary">{key.replace(/_/g, ' ').replace(/vs/g, '↔')}</span>
                      <span className="font-mono text-content-primary">{distance.toFixed(3)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Heuristic Results */}
      {data.heuristics && data.heuristics.ranking && (
        <div className="mb-8">
          <h2 className="text-2xl font-bold mb-4 text-content-primary">Heuristic Archetypes</h2>
          <div className="grid md:grid-cols-2 gap-6">
            {/* Homogeneous Teams */}
            {data.heuristics.ranking.homogeneous && data.heuristics.ranking.homogeneous.length > 0 && (
              <div className="bg-surface-secondary p-6 rounded-lg shadow border border-outline-primary">
                <h3 className="text-xl font-semibold mb-4 text-content-primary">Homogeneous Teams</h3>
                <div className="space-y-3">
                  {data.heuristics.ranking.homogeneous.map((team) => (
                    <div key={team.name}>
                      <div className="flex justify-between mb-1">
                        <span className="font-medium capitalize text-content-primary">{team.name.replace(/_/g, ' ')}</span>
                        <span className="font-bold text-content-primary">{team.mean_payoff.toFixed(1)}</span>
                      </div>
                      <div className="w-full bg-surface-tertiary rounded-full h-3">
                        <div
                          className="bg-green-500 h-3 rounded-full"
                          style={{
                            width: `${(team.mean_payoff / Math.max(...data.heuristics!.ranking.homogeneous.map(t => t.mean_payoff))) * 100}%`
                          }}
                        />
                      </div>
                      <div className="text-xs text-content-secondary mt-1">±{team.std_payoff.toFixed(1)}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Mixed Teams */}
            {data.heuristics.ranking.mixed && data.heuristics.ranking.mixed.length > 0 && (
              <div className="bg-surface-secondary p-6 rounded-lg shadow border border-outline-primary">
                <h3 className="text-xl font-semibold mb-4 text-content-primary">Mixed Teams</h3>
                <div className="space-y-3">
                  {data.heuristics.ranking.mixed.slice(0, 5).map((team) => (
                    <div key={team.composition}>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm text-content-primary">{team.composition}</span>
                        <span className="font-bold text-content-primary">{team.mean_payoff.toFixed(1)}</span>
                      </div>
                      <div className="w-full bg-surface-tertiary rounded-full h-3">
                        <div
                          className="bg-purple-500 h-3 rounded-full"
                          style={{
                            width: `${(team.mean_payoff / Math.max(...data.heuristics!.ranking.mixed.map(t => t.mean_payoff))) * 100}%`
                          }}
                        />
                      </div>
                      <div className="text-xs text-content-secondary mt-1">±{team.std_payoff.toFixed(1)}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Evolution Results */}
      {data.evolution && (
        <div className="mb-8">
          <div className="flex items-center gap-2 mb-4">
            <h2 className="text-2xl font-bold text-content-primary">Evolutionary Optimization</h2>
            <button
              onClick={() => setShowEvolutionInfo(true)}
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
                <div className="text-2xl font-bold text-content-primary">{data.evolution.best.fitness.toFixed(2)}</div>
              </div>
              <div>
                <div className="text-sm text-content-secondary">Generation</div>
                <div className="text-2xl font-bold text-content-primary">{data.evolution.best.generation}</div>
              </div>
              <div>
                <div className="text-sm text-content-secondary">Time</div>
                <div className="text-2xl font-bold text-content-primary">{data.evolution.trace.convergence?.elapsed_time?.toFixed(1) ?? 'N/A'}s</div>
              </div>
            </div>

            {/* Evolution Progress Chart */}
            <div className="mb-6">
              <h4 className="text-lg font-semibold mb-2 text-content-primary">Fitness Over Generations</h4>
              <div className="relative h-48 bg-surface-primary rounded p-4">
                <svg className="w-full h-full" viewBox="0 0 400 150" preserveAspectRatio="none">
                  {/* Grid lines */}
                  {[0, 1, 2, 3, 4].map(i => (
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
                    points={data.evolution!.trace.generations
                      .map((gen, idx) => {
                        const x = (idx / (data.evolution!.trace.generations.length - 1)) * 400;
                        const maxFit = Math.max(...data.evolution!.trace.generations.map(g => g.best_fitness));
                        const minFit = Math.min(...data.evolution!.trace.generations.map(g => g.best_fitness));
                        const y = 150 - ((gen.best_fitness - minFit) / (maxFit - minFit || 1)) * 150;
                        return `${x},${y}`;
                      })
                      .join(' ')}
                    fill="none"
                    stroke="rgb(59, 130, 246)"
                    strokeWidth="2"
                  />

                  {/* Mean fitness line */}
                  <polyline
                    points={data.evolution!.trace.generations
                      .map((gen, idx) => {
                        const x = (idx / (data.evolution!.trace.generations.length - 1)) * 400;
                        const maxFit = Math.max(...data.evolution!.trace.generations.map(g => g.best_fitness));
                        const minFit = Math.min(...data.evolution!.trace.generations.map(g => g.mean_fitness));
                        const y = 150 - ((gen.mean_fitness - minFit) / (maxFit - minFit || 1)) * 150;
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
              <h4 className="text-lg font-semibold mb-3 text-content-primary">Best Agent Parameters</h4>
              <div className="grid grid-cols-2 gap-3">
                {Object.entries(data.evolution.best.parameters).map(([param, value]) => (
                  <div key={param}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="capitalize text-content-secondary">{param.replace(/_/g, ' ')}</span>
                      <span className="font-mono text-content-primary">{(value as number).toFixed(3)}</span>
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
      )}

      {/* Nash Equilibrium Results */}
      {data.nash && (
        <div className="mb-8">
          <div className="flex items-center gap-2 mb-4">
            <h2 className="text-2xl font-bold text-content-primary">Nash Equilibrium Analysis</h2>
            <button
              onClick={() => setShowNashInfo(true)}
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
              <div className="text-2xl font-bold capitalize text-content-primary">{data.nash.equilibrium.type}</div>
            </div>
            <div className="bg-surface-secondary p-6 rounded-lg shadow border border-outline-primary">
              <div className="text-sm text-content-secondary mb-1">Expected Payoff</div>
              <div className="text-2xl font-bold text-content-primary">{data.nash.equilibrium.expected_payoff.toFixed(2)}</div>
            </div>
            <div className="bg-surface-secondary p-6 rounded-lg shadow border border-outline-primary">
              <div className="text-sm text-content-secondary mb-1">Cooperation Rate</div>
              <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                {(data.nash.interpretation.cooperation_rate * 100).toFixed(0)}%
              </div>
            </div>
            <div className="bg-surface-secondary p-6 rounded-lg shadow border border-outline-primary">
              <div className="text-sm text-content-secondary mb-1">Convergence Time</div>
              <div className="text-2xl font-bold text-content-primary">{data.nash.convergence?.elapsed_time?.toFixed(1) ?? 'N/A'}s</div>
            </div>
          </div>

          {/* Strategy Pool */}
          <div className="bg-surface-secondary p-6 rounded-lg shadow mb-6 border border-outline-primary">
            <h3 className="text-xl font-semibold mb-4 text-content-primary">Equilibrium Strategies</h3>
            <div className="space-y-6">
              {data.nash.equilibrium.strategy_pool.map((strategy) => (
                <div key={strategy.index} className="border-l-4 border-blue-500 pl-4">
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <h4 className="text-lg font-bold text-content-primary">{strategy.classification}</h4>
                      <div className="text-sm text-content-secondary">
                        Probability: {(strategy.probability * 100).toFixed(0)}%
                        {strategy.closest_archetype !== strategy.classification && (
                          <span className="ml-2">
                            (Closest to {strategy.closest_archetype}, distance: {strategy.archetype_distance.toFixed(3)})
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
                          <span className="font-mono font-semibold text-content-primary">{(value as number).toFixed(2)}</span>
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
                <div className="font-bold capitalize text-content-primary">{data.nash.algorithm.method.replace(/_/g, ' ')}</div>
              </div>
              <div>
                <div className="text-sm text-content-secondary">Simulations</div>
                <div className="font-bold text-content-primary">{data.nash.algorithm.num_simulations}</div>
              </div>
              <div>
                <div className="text-sm text-content-secondary">Iterations</div>
                <div className="font-bold text-content-primary">{data.nash.convergence?.iterations ?? 'N/A'} / {data.nash.algorithm.max_iterations}</div>
              </div>
              <div>
                <div className="text-sm text-content-secondary">Epsilon</div>
                <div className="font-bold text-content-primary">{data.nash.algorithm.epsilon}</div>
              </div>
              <div>
                <div className="text-sm text-content-secondary">Converged</div>
                <div className={`font-bold ${data.nash.convergence?.converged ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                  {data.nash.convergence?.converged ? 'Yes' : 'No'}
                </div>
              </div>
              <div>
                <div className="text-sm text-content-secondary">Support Size</div>
                <div className="font-bold text-content-primary">{data.nash.equilibrium.support_size}</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Research Insights - Method-Specific */}
      {data.config?.method_insights && Object.keys(data.config.method_insights).length > 0 && (
        <div className="mb-8">
          <h2 className="text-2xl font-bold mb-6 text-content-primary">Research Insights</h2>

          {/* Nash Equilibrium Insights */}
          {data.config.method_insights.nash && data.config.method_insights.nash.length > 0 && (
            <div className="mb-8">
              <h3 className="text-xl font-bold mb-4 text-purple-700 dark:text-purple-300">
                Nash Equilibrium Analysis
              </h3>
              <div className="space-y-6">
                {data.config.method_insights.nash.map((insight: any, idx: number) => (
                  <div key={idx} className="bg-surface-secondary p-6 rounded-lg shadow border-l-4 border-purple-500">
                    <h4 className="text-lg font-semibold mb-3 text-purple-700 dark:text-purple-300">
                      {insight.question}
                    </h4>

                    <div className="mb-4">
                      <div className="text-sm font-semibold text-content-secondary uppercase mb-2">
                        Finding
                      </div>
                      <p className="text-base text-content-primary">
                        {insight.finding}
                      </p>
                    </div>

                    {insight.evidence && insight.evidence.length > 0 && (
                      <div className="mb-4">
                        <div className="text-sm font-semibold text-content-secondary uppercase mb-2">
                          Evidence
                        </div>
                        <ul className="space-y-1">
                          {insight.evidence.map((evidence: string, evidx: number) => (
                            <li key={evidx} className="flex items-start text-sm">
                              <span className="text-purple-500 mr-2 mt-1">•</span>
                              <span className="text-content-secondary">{evidence}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded">
                      <div className="text-sm font-semibold text-purple-600 dark:text-purple-300 uppercase mb-1">
                        Implication
                      </div>
                      <p className="text-sm text-content-primary italic">
                        {insight.implication}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Evolution Insights */}
          {data.config.method_insights.evolution && data.config.method_insights.evolution.length > 0 && (
            <div className="mb-8">
              <h3 className="text-xl font-bold mb-4 text-green-700 dark:text-green-300">
                Evolutionary Optimization
              </h3>
              <div className="space-y-6">
                {data.config.method_insights.evolution.map((insight: any, idx: number) => (
                  <div key={idx} className="bg-surface-secondary p-6 rounded-lg shadow border-l-4 border-green-500">
                    <h4 className="text-lg font-semibold mb-3 text-green-700 dark:text-green-300">
                      {insight.question}
                    </h4>

                    <div className="mb-4">
                      <div className="text-sm font-semibold text-content-secondary uppercase mb-2">
                        Finding
                      </div>
                      <p className="text-base text-content-primary">
                        {insight.finding}
                      </p>
                    </div>

                    {insight.evidence && insight.evidence.length > 0 && (
                      <div className="mb-4">
                        <div className="text-sm font-semibold text-content-secondary uppercase mb-2">
                          Evidence
                        </div>
                        <ul className="space-y-1">
                          {insight.evidence.map((evidence: string, evidx: number) => (
                            <li key={evidx} className="flex items-start text-sm">
                              <span className="text-green-500 mr-2 mt-1">•</span>
                              <span className="text-content-secondary">{evidence}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded">
                      <div className="text-sm font-semibold text-green-600 dark:text-green-300 uppercase mb-1">
                        Implication
                      </div>
                      <p className="text-sm text-content-primary italic">
                        {insight.implication}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Comparative Insights */}
          {data.config.method_insights.comparative && data.config.method_insights.comparative.length > 0 && (
            <div className="mb-8">
              <h3 className="text-xl font-bold mb-4 text-blue-700 dark:text-blue-300">
                Comparative Analysis
              </h3>
              <div className="space-y-6">
                {data.config.method_insights.comparative.map((insight: any, idx: number) => (
                  <div key={idx} className="bg-surface-secondary p-6 rounded-lg shadow border-l-4 border-blue-500">
                    <h4 className="text-lg font-semibold mb-3 text-blue-700 dark:text-blue-300">
                      {insight.question}
                    </h4>

                    <div className="mb-4">
                      <div className="text-sm font-semibold text-content-secondary uppercase mb-2">
                        Finding
                      </div>
                      <p className="text-base text-content-primary">
                        {insight.finding}
                      </p>
                    </div>

                    {insight.evidence && insight.evidence.length > 0 && (
                      <div className="mb-4">
                        <div className="text-sm font-semibold text-content-secondary uppercase mb-2">
                          Evidence
                        </div>
                        <ul className="space-y-1">
                          {insight.evidence.map((evidence: string, evidx: number) => (
                            <li key={evidx} className="flex items-start text-sm">
                              <span className="text-blue-500 mr-2 mt-1">•</span>
                              <span className="text-content-secondary">{evidence}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded">
                      <div className="text-sm font-semibold text-blue-600 dark:text-blue-300 uppercase mb-1">
                        Implication
                      </div>
                      <p className="text-sm text-content-primary italic">
                        {insight.implication}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* PPO Training Insights (placeholder for future) */}
          {data.config.method_insights.ppo && data.config.method_insights.ppo.length > 0 && (
            <div className="mb-8">
              <h3 className="text-xl font-bold mb-4 text-orange-700 dark:text-orange-300">
                PPO Training Analysis
              </h3>
              <div className="space-y-6">
                {data.config.method_insights.ppo.map((insight: any, idx: number) => (
                  <div key={idx} className="bg-surface-secondary p-6 rounded-lg shadow border-l-4 border-orange-500">
                    <h4 className="text-lg font-semibold mb-3 text-orange-700 dark:text-orange-300">
                      {insight.question}
                    </h4>

                    <div className="mb-4">
                      <div className="text-sm font-semibold text-content-secondary uppercase mb-2">
                        Finding
                      </div>
                      <p className="text-base text-content-primary">
                        {insight.finding}
                      </p>
                    </div>

                    {insight.evidence && insight.evidence.length > 0 && (
                      <div className="mb-4">
                        <div className="text-sm font-semibold text-content-secondary uppercase mb-2">
                          Evidence
                        </div>
                        <ul className="space-y-1">
                          {insight.evidence.map((evidence: string, evidx: number) => (
                            <li key={evidx} className="flex items-start text-sm">
                              <span className="text-orange-500 mr-2 mt-1">•</span>
                              <span className="text-content-secondary">{evidence}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded">
                      <div className="text-sm font-semibold text-orange-600 dark:text-orange-300 uppercase mb-1">
                        Implication
                      </div>
                      <p className="text-sm text-content-primary italic">
                        {insight.implication}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Legacy Research Insights (fallback) */}
      {(!data.config?.method_insights || Object.keys(data.config.method_insights).length === 0) &&
       data.config?.research_insights && data.config.research_insights.length > 0 && (
        <div className="mb-8">
          <h2 className="text-2xl font-bold mb-6 text-content-primary">Research Insights</h2>
          <div className="space-y-6">
            {data.config.research_insights.map((insight: any, idx: number) => (
              <div key={idx} className="bg-surface-secondary p-6 rounded-lg shadow border-l-4 border-blue-500">
                <h3 className="text-xl font-semibold mb-3 text-blue-700 dark:text-blue-300">
                  {insight.question}
                </h3>

                <div className="mb-4">
                  <div className="text-sm font-semibold text-content-secondary uppercase mb-2">
                    Finding
                  </div>
                  <p className="text-lg text-content-primary">
                    {insight.finding}
                  </p>
                </div>

                {insight.evidence && insight.evidence.length > 0 && (
                  <div className="mb-4">
                    <div className="text-sm font-semibold text-content-secondary uppercase mb-2">
                      Evidence
                    </div>
                    <ul className="space-y-2">
                      {insight.evidence.map((evidence: string, evidx: number) => (
                        <li key={evidx} className="flex items-start">
                          <span className="text-green-500 mr-2 mt-1">•</span>
                          <span className="text-content-secondary">{evidence}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded">
                  <div className="text-sm font-semibold text-blue-600 dark:text-blue-300 uppercase mb-1">
                    Implication
                  </div>
                  <p className="text-content-primary italic">
                    {insight.implication}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Research Questions (fallback if no insights) */}
      {(!data.config?.research_insights || data.config.research_insights.length === 0) &&
       data.config?.research_questions && data.config.research_questions.length > 0 && (
        <div className="mb-8">
          <h2 className="text-2xl font-bold mb-4 text-content-primary">Research Questions</h2>
          <div className="bg-surface-secondary p-6 rounded-lg shadow border border-outline-primary">
            <ul className="list-disc list-inside space-y-2">
              {data.config.research_questions.map((question: string, idx: number) => (
                <li key={idx} className="text-content-secondary">{question}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* Evolution Info Modal */}
      {showEvolutionInfo && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4 modal-overlay">
          <div className="bg-surface-primary rounded-lg shadow-xl max-w-3xl w-full max-h-[90vh] overflow-y-auto modal-content">
            <div className="sticky top-0 bg-surface-primary border-b border-outline-primary p-6 flex items-center justify-between">
              <h3 className="text-2xl font-bold text-content-primary">Evolutionary Optimization: Technical Details</h3>
              <button
                onClick={() => setShowEvolutionInfo(false)}
                className="p-2 rounded-full hover:bg-surface-tertiary transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-6 space-y-6">
              <div>
                <h4 className="text-lg font-semibold mb-3 text-content-primary">Genetic Algorithm Overview</h4>
                <p className="text-content-secondary mb-3">
                  We employ a genetic algorithm to evolve agent strategies that maximize expected payoff in the Bucket Brigade game. Each agent is represented by a 10-dimensional parameter vector (genome) that encodes behavioral traits.
                </p>
                <p className="text-content-secondary">
                  The algorithm maintains a population of candidate strategies and iteratively improves them through selection, crossover, and mutation operators inspired by biological evolution.
                </p>
              </div>

              <div>
                <h4 className="text-lg font-semibold mb-3 text-content-primary">Fitness Function</h4>
                <p className="text-content-secondary mb-3">
                  The fitness of an agent strategy θ is measured as the expected reward over multiple game simulations:
                </p>
                <div className="bg-surface-tertiary p-4 rounded-lg font-mono text-sm mb-3">
                  f(θ) = 𝔼[R(θ)] = (1/N) Σᵢ Rᵢ(θ)
                </div>
                <p className="text-content-secondary text-sm">
                  where N is the number of evaluation games and Rᵢ(θ) is the total reward in game i.
                </p>
              </div>

              <div>
                <h4 className="text-lg font-semibold mb-3 text-content-primary">Evolutionary Operators</h4>
                <div className="space-y-3">
                  <div>
                    <h5 className="font-semibold text-content-primary mb-1">Selection (Tournament)</h5>
                    <p className="text-content-secondary text-sm">
                      Parents are selected via tournament selection with size k=3. We randomly sample k individuals and select the one with highest fitness. This balances exploration and exploitation.
                    </p>
                  </div>
                  <div>
                    <h5 className="font-semibold text-content-primary mb-1">Crossover (Uniform)</h5>
                    <p className="text-content-secondary text-sm mb-2">
                      Two parents θ₁, θ₂ produce offspring θ' where each parameter is inherited from parent 1 with probability p=0.5:
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
                  The top 5 individuals are preserved unchanged across generations, ensuring monotonic improvement in best fitness and preventing loss of good solutions due to stochastic noise.
                </p>
              </div>

              <div>
                <h4 className="text-lg font-semibold mb-3 text-content-primary">Convergence Criteria</h4>
                <p className="text-content-secondary">
                  Evolution terminates after a fixed number of generations (typically 15,000). We track population diversity using the average pairwise Euclidean distance between genomes to monitor convergence.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Nash Info Modal */}
      {showNashInfo && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4 modal-overlay">
          <div className="bg-surface-primary rounded-lg shadow-xl max-w-3xl w-full max-h-[90vh] overflow-y-auto modal-content">
            <div className="sticky top-0 bg-surface-primary border-b border-outline-primary p-6 flex items-center justify-between">
              <h3 className="text-2xl font-bold text-content-primary">Nash Equilibrium Analysis: Technical Details</h3>
              <button
                onClick={() => setShowNashInfo(false)}
                className="p-2 rounded-full hover:bg-surface-tertiary transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-6 space-y-6">
              <div>
                <h4 className="text-lg font-semibold mb-3 text-content-primary">Symmetric Game Formulation</h4>
                <p className="text-content-secondary mb-3">
                  We model the Bucket Brigade as a symmetric multi-agent game where all players share the same strategy space Θ (the 10-dimensional parameter space). A Nash equilibrium is a probability distribution over strategies such that no player can improve their expected payoff by unilaterally deviating.
                </p>
                <div className="bg-surface-tertiary p-4 rounded-lg font-mono text-sm mb-3">
                  π* ∈ arg max_π 𝔼<sub>θ~π</sub>[u(θ, π*)]
                </div>
                <p className="text-content-secondary text-sm">
                  where u(θ, π) is the expected payoff when playing strategy θ against population distribution π.
                </p>
              </div>

              <div>
                <h4 className="text-lg font-semibold mb-3 text-content-primary">Support Enumeration Method</h4>
                <p className="text-content-secondary mb-3">
                  We use a support enumeration algorithm that searches over candidate support sets S ⊂ Θ. For each support size k, we:
                </p>
                <ol className="list-decimal list-inside space-y-2 text-content-secondary text-sm mb-3">
                  <li>Sample k strategies from the parameter space</li>
                  <li>Compute the payoff matrix A where A<sub>ij</sub> = u(θᵢ, θⱼ)</li>
                  <li>Solve for the mixed strategy equilibrium over these k strategies</li>
                  <li>Verify the equilibrium condition: all support strategies have equal payoff and all non-support strategies have weakly lower payoff</li>
                </ol>
              </div>

              <div>
                <h4 className="text-lg font-semibold mb-3 text-content-primary">Payoff Estimation via Monte Carlo</h4>
                <p className="text-content-secondary mb-3">
                  Each payoff matrix entry requires estimating the expected payoff of strategy θᵢ when playing against θⱼ:
                </p>
                <div className="bg-surface-tertiary p-4 rounded-lg font-mono text-sm mb-3">
                  u(θᵢ, θⱼ) ≈ (1/M) Σₘ R<sub>m</sub>(θᵢ | opponent=θⱼ)
                </div>
                <p className="text-content-secondary text-sm">
                  where M is the number of Monte Carlo simulations per strategy pair (typically 1000).
                </p>
              </div>

              <div>
                <h4 className="text-lg font-semibold mb-3 text-content-primary">Linear Programming Solution</h4>
                <p className="text-content-secondary mb-3">
                  Given a candidate support S with payoff matrix A, we find the equilibrium probabilities by solving:
                </p>
                <div className="bg-surface-tertiary p-4 rounded-lg space-y-2 font-mono text-xs mb-3">
                  <div>maximize: v</div>
                  <div>subject to: Aᵀp ≥ v·1</div>
                  <div className="ml-16">1ᵀp = 1</div>
                  <div className="ml-16">p ≥ 0</div>
                </div>
                <p className="text-content-secondary text-sm">
                  where p is the probability distribution over the support and v is the equilibrium payoff.
                </p>
              </div>

              <div>
                <h4 className="text-lg font-semibold mb-3 text-content-primary">Equilibrium Classification</h4>
                <p className="text-content-secondary mb-3">
                  We classify equilibria based on their support size and strategic characteristics:
                </p>
                <ul className="list-disc list-inside space-y-1 text-content-secondary text-sm">
                  <li><strong>Pure equilibrium</strong>: Single strategy with probability 1.0</li>
                  <li><strong>Mixed equilibrium</strong>: Multiple strategies in support with varying probabilities</li>
                  <li><strong>Cooperation rate</strong>: Expected probability of working (weighted by strategy probabilities and work_tendency parameters)</li>
                </ul>
              </div>

              <div>
                <h4 className="text-lg font-semibold mb-3 text-content-primary">Convergence Guarantees</h4>
                <p className="text-content-secondary">
                  For finite games with symmetric strategy spaces, Nash equilibria are guaranteed to exist (by Brouwer's fixed-point theorem). Our algorithm systematically explores support sizes k=1,2,...,K<sub>max</sub> and verifies equilibrium conditions numerically with tolerance ε=0.001.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
