import { useState } from 'react';
import { type ScenarioName } from '../types/research';
import { useScenarioData } from '../hooks/useScenarioData';
import { ScenarioSelector } from '../components/research/ScenarioSelector';
import { ScenarioOverview } from '../components/research/ScenarioOverview';
import { ComparisonSection } from '../components/research/ComparisonSection';
import { HeuristicsSection } from '../components/research/HeuristicsSection';
import { EvolutionSection } from '../components/research/EvolutionSection';
import { NashSection } from '../components/research/NashSection';
import { ResearchInsightsSection } from '../components/research/ResearchInsightsSection';
import { EvolutionInfoModal } from '../components/research/EvolutionInfoModal';
import { NashInfoModal } from '../components/research/NashInfoModal';

export default function ScenarioResearch() {
  const [selectedScenario, setSelectedScenario] = useState<ScenarioName>('greedy_neighbor');
  const [showEvolutionInfo, setShowEvolutionInfo] = useState(false);
  const [showNashInfo, setShowNashInfo] = useState(false);

  const { data, loading, error } = useScenarioData(selectedScenario);

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

      <ScenarioSelector selected={selectedScenario} onChange={setSelectedScenario} />

      <ScenarioOverview config={data.config} />

      {/* Comparison Results - Show this first as it's the most important */}
      {data.comparison && (
        <ComparisonSection comparison={data.comparison} nash={data.nash} />
      )}

      {/* Heuristic Results */}
      {data.heuristics && <HeuristicsSection heuristics={data.heuristics} />}

      {/* Evolution Results */}
      {data.evolution && (
        <EvolutionSection
          trace={data.evolution.trace}
          best={data.evolution.best}
          onShowInfo={() => setShowEvolutionInfo(true)}
        />
      )}

      {/* Nash Equilibrium Results */}
      {data.nash && (
        <NashSection nash={data.nash} onShowInfo={() => setShowNashInfo(true)} />
      )}

      {/* Research Insights / Questions */}
      <ResearchInsightsSection config={data.config} />

      {/* Info Modals */}
      <EvolutionInfoModal
        open={showEvolutionInfo}
        onClose={() => setShowEvolutionInfo(false)}
      />
      <NashInfoModal open={showNashInfo} onClose={() => setShowNashInfo(false)} />
    </div>
  );
}
