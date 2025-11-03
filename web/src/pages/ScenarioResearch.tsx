import { useState, useEffect } from 'react';
import {
  SCENARIOS,
  type ScenarioName,
  type HeuristicResults,
  type EvolutionTrace,
  type BestAgent,
  type ComparisonResults,
} from '../types/research';

interface ScenarioData {
  config: any;
  heuristics?: HeuristicResults;
  evolution?: {
    trace: EvolutionTrace;
    best: BestAgent;
  };
  comparison?: ComparisonResults;
}

export default function ScenarioResearch() {
  const [selectedScenario, setSelectedScenario] = useState<ScenarioName>('greedy_neighbor');
  const [data, setData] = useState<ScenarioData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadScenarioData(selectedScenario);
  }, [selectedScenario]);

  const loadScenarioData = async (scenario: ScenarioName) => {
    setLoading(true);
    setError(null);

    try {
      const basePath = import.meta.env.BASE_URL || '/';
      const scenarioPath = `${basePath}research/scenarios/${scenario}`;

      // Load all available data files
      const [config, heuristics, evolutionTrace, evolutionBest, comparison] = await Promise.all([
        fetch(`${scenarioPath}/config.json`).then(r => r.ok ? r.json() : null),
        fetch(`${scenarioPath}/heuristics/results.json`).then(r => r.ok ? r.json() : null),
        fetch(`${scenarioPath}/evolved/evolution_trace.json`).then(r => r.ok ? r.json() : null),
        fetch(`${scenarioPath}/evolved/best_agent.json`).then(r => r.ok ? r.json() : null),
        fetch(`${scenarioPath}/comparison/comparison.json`).then(r => r.ok ? r.json() : null),
      ]);

      setData({
        config,
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
      {data.heuristics && (
        <div className="mb-8">
          <h2 className="text-2xl font-bold mb-4">Heuristic Archetypes</h2>
          <div className="grid md:grid-cols-2 gap-6">
            {/* Homogeneous Teams */}
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

            {/* Mixed Teams */}
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

      {/* Research Questions */}
      {data.config?.research_questions && data.config.research_questions.length > 0 && (
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
