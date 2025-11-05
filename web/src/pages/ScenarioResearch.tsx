import { useState, useEffect } from 'react';
import {
  SCENARIOS,
  type ScenarioName,
  type HeuristicResults,
  type EvolutionTrace,
  type BestAgent,
  type ComparisonResults,
  type NashResults,
} from '../types/research';

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

export default function ScenarioResearch() {
  const [selectedScenario, setSelectedScenario] = useState<ScenarioName>('greedy_neighbor');
  const [activeTab, setActiveTab] = useState<'nash' | 'evolution' | 'heuristics'>('evolution');
  const [data, setData] = useState<ScenarioData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadScenarioData(selectedScenario);
  }, [selectedScenario, activeTab]);

  const loadScenarioData = async (scenario: ScenarioName) => {
    setLoading(true);
    setError(null);

    try {
      const basePath = import.meta.env.BASE_URL || '/';
      const scenarioPath = `${basePath}research/scenarios/${scenario}`;

      // Load config first
      const config = await fetch(`${scenarioPath}/config.json`).then(r => r.ok ? r.json() : null);

      // Load data based on active tab
      let nash, heuristics, evolutionTrace, evolutionBest, comparison;

      if (activeTab === 'nash') {
        nash = await fetch(`${scenarioPath}/nash/equilibrium.json`).then(r => r.ok ? r.json() : null);
      } else if (activeTab === 'evolution') {
        [evolutionTrace, evolutionBest] = await Promise.all([
          fetch(`${scenarioPath}/evolved/evolution_trace.json`).then(r => r.ok ? r.json() : null),
          fetch(`${scenarioPath}/evolved/best_agent.json`).then(r => r.ok ? r.json() : null),
        ]);
      } else if (activeTab === 'heuristics') {
        [heuristics, comparison] = await Promise.all([
          fetch(`${scenarioPath}/heuristics/results.json`).then(r => r.ok ? r.json() : null),
          fetch(`${scenarioPath}/comparison/comparison.json`).then(r => r.ok ? r.json() : null),
        ]);
      }

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
          <div className="text-xl">Loading scenario data...</div>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="text-center text-red-500">
          <div className="text-xl">{error || 'Failed to load data'}</div>
        </div>
      </div>
    );
  }

  const tabs = [
  { id: 'nash', label: 'Nash Equilibrium', icon: '🎯' },
  { id: 'evolution', label: 'Evolution', icon: '🧬' },
    { id: 'heuristics', label: 'Heuristics', icon: '🧠' },
  ] as const;

  return (
  <div className="container mx-auto px-4 py-8">
  <h1 className="text-4xl font-bold mb-8">Scenario Research</h1>

  {/* Scenario Selector */}
  <div className="mb-8">
  <label className="block text-sm font-medium mb-2">Select Scenario</label>
  <select
  value={selectedScenario}
  onChange={(e) => setSelectedScenario(e.target.value as ScenarioName)}
  className="w-full md:w-auto px-4 py-2 border rounded-lg bg-white dark:bg-gray-800"
  >
      {SCENARIOS.map((scenario) => (
            <option key={scenario} value={scenario}>
              {scenario.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
            </option>
          ))}
        </select>
      </div>

      {/* Tab Navigation */}
      <div className="mb-8">
        <nav className="flex space-x-1 bg-gray-100 dark:bg-gray-800 p-1 rounded-lg">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm'
                  : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
              }`}
            >
              <span className="mr-2">{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Scenario Overview */}
      {data.config && (
        <div className="mb-8 p-6 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h2 className="text-2xl font-bold mb-4">{data.config.description}</h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">{data.config.story}</p>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <div className="text-sm text-gray-500">Fire Spread (β)</div>
              <div className="text-xl font-bold">{data.config.parameters.beta}</div>
            </div>
            <div>
              <div className="text-sm text-gray-500">Extinguish (κ)</div>
              <div className="text-xl font-bold">{data.config.parameters.kappa}</div>
            </div>
            <div>
              <div className="text-sm text-gray-500">Work Cost (c)</div>
              <div className="text-xl font-bold">{data.config.parameters.c}</div>
            </div>
            <div>
              <div className="text-sm text-gray-500">Agents</div>
              <div className="text-xl font-bold">{data.config.parameters.num_agents}</div>
            </div>
          </div>
        </div>
      )}

      {/* Comparison Results - Show this first as it's the most important */}
      {data.comparison && (
        <div className="mb-8">
          <h2 className="text-2xl font-bold mb-4">Strategy Comparison</h2>
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
            <h3 className="text-xl font-semibold mb-4">Tournament Results</h3>
            <div className="space-y-4">
              {data.comparison.ranking.map((result, idx) => (
                <div key={result.name} className="flex items-center gap-4">
                  <div className="text-2xl font-bold text-gray-400 w-8">#{idx + 1}</div>
                  <div className="flex-1">
                    <div className="flex justify-between items-center mb-1">
                      <span className="font-semibold capitalize">{result.name.replace(/_/g, ' ')}</span>
                      <span className="text-lg font-bold">{result.mean_payoff.toFixed(2)}</span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-4">
                      <div
                        className="bg-blue-500 h-4 rounded-full"
                        style={{
                          width: `${(result.mean_payoff / Math.max(...data.comparison!.ranking.map(r => r.mean_payoff))) * 100}%`
                        }}
                      />
                    </div>
                    <div className="text-sm text-gray-500 mt-1">
                      ±{result.std_payoff.toFixed(2)}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Strategy Distance */}
            {Object.keys(data.comparison.distances).length > 0 && (
              <div className="mt-6">
                <h4 className="text-lg font-semibold mb-2">Strategy Distances</h4>
                <div className="space-y-2">
                  {Object.entries(data.comparison.distances).map(([key, distance]) => (
                    <div key={key} className="flex justify-between text-sm">
                      <span className="capitalize">{key.replace(/_/g, ' ').replace(/vs/g, '↔')}</span>
                      <span className="font-mono">{distance.toFixed(3)}</span>
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
          <h2 className="text-2xl font-bold mb-4">Heuristic Archetypes</h2>
          <div className="grid md:grid-cols-2 gap-6">
            {/* Homogeneous Teams */}
            {data.heuristics.ranking.homogeneous && data.heuristics.ranking.homogeneous.length > 0 && (
              <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
                <h3 className="text-xl font-semibold mb-4">Homogeneous Teams</h3>
                <div className="space-y-3">
                  {data.heuristics.ranking.homogeneous.map((team) => (
                    <div key={team.name}>
                      <div className="flex justify-between mb-1">
                        <span className="font-medium capitalize">{team.name.replace(/_/g, ' ')}</span>
                        <span className="font-bold">{team.mean_payoff.toFixed(1)}</span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                        <div
                          className="bg-green-500 h-3 rounded-full"
                          style={{
                            width: `${(team.mean_payoff / Math.max(...data.heuristics!.ranking.homogeneous.map(t => t.mean_payoff))) * 100}%`
                          }}
                        />
                      </div>
                      <div className="text-xs text-gray-500 mt-1">±{team.std_payoff.toFixed(1)}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Mixed Teams */}
            {data.heuristics.ranking.mixed && data.heuristics.ranking.mixed.length > 0 && (
              <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
                <h3 className="text-xl font-semibold mb-4">Mixed Teams</h3>
                <div className="space-y-3">
                  {data.heuristics.ranking.mixed.slice(0, 5).map((team) => (
                    <div key={team.composition}>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm">{team.composition}</span>
                        <span className="font-bold">{team.mean_payoff.toFixed(1)}</span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                        <div
                          className="bg-purple-500 h-3 rounded-full"
                          style={{
                            width: `${(team.mean_payoff / Math.max(...data.heuristics!.ranking.mixed.map(t => t.mean_payoff))) * 100}%`
                          }}
                        />
                      </div>
                      <div className="text-xs text-gray-500 mt-1">±{team.std_payoff.toFixed(1)}</div>
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
          <h2 className="text-2xl font-bold mb-4">Evolutionary Optimization</h2>
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
            <div className="grid md:grid-cols-3 gap-4 mb-6">
              <div>
                <div className="text-sm text-gray-500">Best Fitness</div>
                <div className="text-2xl font-bold">{data.evolution.best.fitness.toFixed(2)}</div>
              </div>
              <div>
                <div className="text-sm text-gray-500">Generation</div>
                <div className="text-2xl font-bold">{data.evolution.best.generation}</div>
              </div>
              <div>
                <div className="text-sm text-gray-500">Time</div>
                <div className="text-2xl font-bold">{data.evolution.trace.convergence.elapsed_time.toFixed(1)}s</div>
              </div>
            </div>

            {/* Evolution Progress Chart */}
            <div className="mb-6">
              <h4 className="text-lg font-semibold mb-2">Fitness Over Generations</h4>
              <div className="relative h-48 bg-gray-50 dark:bg-gray-900 rounded p-4">
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
                  <span>Best Fitness</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-0.5 bg-gray-400 border-dashed" />
                  <span>Mean Fitness</span>
                </div>
              </div>
            </div>

            {/* Best Agent Parameters */}
            <div>
              <h4 className="text-lg font-semibold mb-3">Best Agent Parameters</h4>
              <div className="grid grid-cols-2 gap-3">
                {Object.entries(data.evolution.best.parameters).map(([param, value]) => (
                  <div key={param}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="capitalize">{param.replace(/_/g, ' ')}</span>
                      <span className="font-mono">{(value as number).toFixed(3)}</span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
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
          <h2 className="text-2xl font-bold mb-4">Nash Equilibrium Analysis</h2>

          {/* Key Metrics */}
          <div className="grid md:grid-cols-4 gap-4 mb-6">
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
              <div className="text-sm text-gray-500 mb-1">Equilibrium Type</div>
              <div className="text-2xl font-bold capitalize">{data.nash.equilibrium.type}</div>
            </div>
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
              <div className="text-sm text-gray-500 mb-1">Expected Payoff</div>
              <div className="text-2xl font-bold">{data.nash.equilibrium.expected_payoff.toFixed(2)}</div>
            </div>
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
              <div className="text-sm text-gray-500 mb-1">Cooperation Rate</div>
              <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                {(data.nash.interpretation.cooperation_rate * 100).toFixed(0)}%
              </div>
            </div>
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
              <div className="text-sm text-gray-500 mb-1">Convergence Time</div>
              <div className="text-2xl font-bold">{data.nash.convergence.elapsed_time.toFixed(1)}s</div>
            </div>
          </div>

          {/* Strategy Pool */}
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow mb-6">
            <h3 className="text-xl font-semibold mb-4">Equilibrium Strategies</h3>
            <div className="space-y-6">
              {data.nash.equilibrium.strategy_pool.map((strategy) => (
                <div key={strategy.index} className="border-l-4 border-blue-500 pl-4">
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <h4 className="text-lg font-bold">{strategy.classification}</h4>
                      <div className="text-sm text-gray-500">
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
                          <span className="capitalize text-gray-600 dark:text-gray-400">
                            {param.replace(/_/g, ' ')}
                          </span>
                          <span className="font-mono font-semibold">{(value as number).toFixed(2)}</span>
                        </div>
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
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
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
            <h3 className="text-xl font-semibold mb-4">Computation Details</h3>
            <div className="grid md:grid-cols-3 gap-4">
              <div>
                <div className="text-sm text-gray-500">Algorithm</div>
                <div className="font-bold capitalize">{data.nash.algorithm.method.replace(/_/g, ' ')}</div>
              </div>
              <div>
                <div className="text-sm text-gray-500">Simulations</div>
                <div className="font-bold">{data.nash.algorithm.num_simulations}</div>
              </div>
              <div>
                <div className="text-sm text-gray-500">Iterations</div>
                <div className="font-bold">{data.nash.convergence.iterations} / {data.nash.algorithm.max_iterations}</div>
              </div>
              <div>
                <div className="text-sm text-gray-500">Epsilon</div>
                <div className="font-bold">{data.nash.algorithm.epsilon}</div>
              </div>
              <div>
                <div className="text-sm text-gray-500">Converged</div>
                <div className={`font-bold ${data.nash.convergence.converged ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                  {data.nash.convergence.converged ? 'Yes' : 'No'}
                </div>
              </div>
              <div>
                <div className="text-sm text-gray-500">Support Size</div>
                <div className="font-bold">{data.nash.equilibrium.support_size}</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Research Insights - Method-Specific */}
      {data.config?.method_insights && Object.keys(data.config.method_insights).length > 0 && (
        <div className="mb-8">
          <h2 className="text-2xl font-bold mb-6">Research Insights</h2>

          {/* Nash Equilibrium Insights */}
          {data.config.method_insights.nash && data.config.method_insights.nash.length > 0 && (
            <div className="mb-8">
              <h3 className="text-xl font-bold mb-4 text-purple-700 dark:text-purple-300">
                Nash Equilibrium Analysis
              </h3>
              <div className="space-y-6">
                {data.config.method_insights.nash.map((insight: any, idx: number) => (
                  <div key={idx} className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow border-l-4 border-purple-500">
                    <h4 className="text-lg font-semibold mb-3 text-purple-700 dark:text-purple-300">
                      {insight.question}
                    </h4>

                    <div className="mb-4">
                      <div className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase mb-2">
                        Finding
                      </div>
                      <p className="text-base text-gray-800 dark:text-gray-200">
                        {insight.finding}
                      </p>
                    </div>

                    {insight.evidence && insight.evidence.length > 0 && (
                      <div className="mb-4">
                        <div className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase mb-2">
                          Evidence
                        </div>
                        <ul className="space-y-1">
                          {insight.evidence.map((evidence: string, evidx: number) => (
                            <li key={evidx} className="flex items-start text-sm">
                              <span className="text-purple-500 mr-2 mt-1">•</span>
                              <span className="text-gray-700 dark:text-gray-300">{evidence}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded">
                      <div className="text-sm font-semibold text-purple-600 dark:text-purple-300 uppercase mb-1">
                        Implication
                      </div>
                      <p className="text-sm text-gray-800 dark:text-gray-200 italic">
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
                  <div key={idx} className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow border-l-4 border-green-500">
                    <h4 className="text-lg font-semibold mb-3 text-green-700 dark:text-green-300">
                      {insight.question}
                    </h4>

                    <div className="mb-4">
                      <div className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase mb-2">
                        Finding
                      </div>
                      <p className="text-base text-gray-800 dark:text-gray-200">
                        {insight.finding}
                      </p>
                    </div>

                    {insight.evidence && insight.evidence.length > 0 && (
                      <div className="mb-4">
                        <div className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase mb-2">
                          Evidence
                        </div>
                        <ul className="space-y-1">
                          {insight.evidence.map((evidence: string, evidx: number) => (
                            <li key={evidx} className="flex items-start text-sm">
                              <span className="text-green-500 mr-2 mt-1">•</span>
                              <span className="text-gray-700 dark:text-gray-300">{evidence}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded">
                      <div className="text-sm font-semibold text-green-600 dark:text-green-300 uppercase mb-1">
                        Implication
                      </div>
                      <p className="text-sm text-gray-800 dark:text-gray-200 italic">
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
                  <div key={idx} className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow border-l-4 border-blue-500">
                    <h4 className="text-lg font-semibold mb-3 text-blue-700 dark:text-blue-300">
                      {insight.question}
                    </h4>

                    <div className="mb-4">
                      <div className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase mb-2">
                        Finding
                      </div>
                      <p className="text-base text-gray-800 dark:text-gray-200">
                        {insight.finding}
                      </p>
                    </div>

                    {insight.evidence && insight.evidence.length > 0 && (
                      <div className="mb-4">
                        <div className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase mb-2">
                          Evidence
                        </div>
                        <ul className="space-y-1">
                          {insight.evidence.map((evidence: string, evidx: number) => (
                            <li key={evidx} className="flex items-start text-sm">
                              <span className="text-blue-500 mr-2 mt-1">•</span>
                              <span className="text-gray-700 dark:text-gray-300">{evidence}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded">
                      <div className="text-sm font-semibold text-blue-600 dark:text-blue-300 uppercase mb-1">
                        Implication
                      </div>
                      <p className="text-sm text-gray-800 dark:text-gray-200 italic">
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
                  <div key={idx} className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow border-l-4 border-orange-500">
                    <h4 className="text-lg font-semibold mb-3 text-orange-700 dark:text-orange-300">
                      {insight.question}
                    </h4>

                    <div className="mb-4">
                      <div className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase mb-2">
                        Finding
                      </div>
                      <p className="text-base text-gray-800 dark:text-gray-200">
                        {insight.finding}
                      </p>
                    </div>

                    {insight.evidence && insight.evidence.length > 0 && (
                      <div className="mb-4">
                        <div className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase mb-2">
                          Evidence
                        </div>
                        <ul className="space-y-1">
                          {insight.evidence.map((evidence: string, evidx: number) => (
                            <li key={evidx} className="flex items-start text-sm">
                              <span className="text-orange-500 mr-2 mt-1">•</span>
                              <span className="text-gray-700 dark:text-gray-300">{evidence}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded">
                      <div className="text-sm font-semibold text-orange-600 dark:text-orange-300 uppercase mb-1">
                        Implication
                      </div>
                      <p className="text-sm text-gray-800 dark:text-gray-200 italic">
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
          <h2 className="text-2xl font-bold mb-6">Research Insights</h2>
          <div className="space-y-6">
            {data.config.research_insights.map((insight: any, idx: number) => (
              <div key={idx} className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow border-l-4 border-blue-500">
                <h3 className="text-xl font-semibold mb-3 text-blue-700 dark:text-blue-300">
                  {insight.question}
                </h3>

                <div className="mb-4">
                  <div className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase mb-2">
                    Finding
                  </div>
                  <p className="text-lg text-gray-800 dark:text-gray-200">
                    {insight.finding}
                  </p>
                </div>

                {insight.evidence && insight.evidence.length > 0 && (
                  <div className="mb-4">
                    <div className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase mb-2">
                      Evidence
                    </div>
                    <ul className="space-y-2">
                      {insight.evidence.map((evidence: string, evidx: number) => (
                        <li key={evidx} className="flex items-start">
                          <span className="text-green-500 mr-2 mt-1">•</span>
                          <span className="text-gray-700 dark:text-gray-300">{evidence}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded">
                  <div className="text-sm font-semibold text-blue-600 dark:text-blue-300 uppercase mb-1">
                    Implication
                  </div>
                  <p className="text-gray-800 dark:text-gray-200 italic">
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
          <h2 className="text-2xl font-bold mb-4">Research Questions</h2>
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
            <ul className="list-disc list-inside space-y-2">
              {data.config.research_questions.map((question: string, idx: number) => (
                <li key={idx} className="text-gray-700 dark:text-gray-300">{question}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
