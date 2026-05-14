import { useEffect, useState } from 'react';
import type {
  BestAgent,
  ComparisonResults,
  EvolutionTrace,
  HeuristicResults,
  NashResults,
  ScenarioName,
} from '../types/research';

export interface ScenarioData {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  config: any;
  heuristics?: HeuristicResults;
  evolution?: {
    trace: EvolutionTrace;
    best: BestAgent;
  };
  nash?: NashResults;
  comparison?: ComparisonResults;
}

export interface UseScenarioDataResult {
  data: ScenarioData | null;
  loading: boolean;
  error: string | null;
}

/**
 * Hook that loads all research data for a given scenario in parallel.
 *
 * Fetches config, nash, heuristics, evolution (trace + best agent), and
 * comparison JSON files from the public research assets. Missing files
 * resolve to `null` / `undefined` rather than throwing.
 */
export function useScenarioData(scenario: ScenarioName): UseScenarioDataResult {
  const [data, setData] = useState<ScenarioData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      setLoading(true);
      setError(null);

      try {
        const basePath = import.meta.env.BASE_URL || '/';
        const scenarioPath = `${basePath}research/scenarios/${scenario}`;

        const [config, nash, heuristics, evolutionTrace, evolutionBest, comparison] =
          await Promise.all([
            fetch(`${scenarioPath}/config.json`).then((r) => (r.ok ? r.json() : null)),
            fetch(`${scenarioPath}/nash/equilibrium.json`).then((r) =>
              r.ok ? r.json() : null,
            ),
            fetch(`${scenarioPath}/heuristics/results.json`).then((r) =>
              r.ok ? r.json() : null,
            ),
            fetch(`${scenarioPath}/evolved/evolution_trace.json`).then((r) =>
              r.ok ? r.json() : null,
            ),
            fetch(`${scenarioPath}/evolved/best_agent.json`).then((r) =>
              r.ok ? r.json() : null,
            ),
            fetch(`${scenarioPath}/comparison/comparison.json`).then((r) =>
              r.ok ? r.json() : null,
            ),
          ]);

        if (cancelled) return;

        setData({
          config,
          nash,
          heuristics,
          evolution:
            evolutionTrace && evolutionBest
              ? { trace: evolutionTrace, best: evolutionBest }
              : undefined,
          comparison,
        });
      } catch (err) {
        if (!cancelled) {
          setError(`Failed to load scenario data: ${err}`);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    load();

    return () => {
      cancelled = true;
    };
  }, [scenario]);

  return { data, loading, error };
}
