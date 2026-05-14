import { memo } from 'react';

export interface ScenarioOverviewProps {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  config: any;
}

function ScenarioOverviewImpl({ config }: ScenarioOverviewProps) {
  if (!config) return null;

  return (
    <div className="mb-8 p-6 bg-surface-tertiary rounded-lg">
      <h2 className="text-2xl font-bold mb-4 text-content-primary">{config.description}</h2>
      <p className="text-content-secondary mb-4">{config.story}</p>

      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <div>
          <div className="text-sm text-content-secondary">Fire Spread (β)</div>
          <div className="text-xl font-bold text-content-primary">{config.parameters.beta}</div>
        </div>
        <div>
          <div className="text-sm text-content-secondary">Extinguish (κ)</div>
          <div className="text-xl font-bold text-content-primary">{config.parameters.kappa}</div>
        </div>
        <div>
          <div className="text-sm text-content-secondary">Reward/A (saved)</div>
          <div className="text-xl font-bold text-content-primary">{config.parameters.A}</div>
        </div>
        <div>
          <div className="text-sm text-content-secondary">Penalty/L (ruined)</div>
          <div className="text-xl font-bold text-content-primary">{config.parameters.L}</div>
        </div>
        <div>
          <div className="text-sm text-content-secondary">Work Cost (c)</div>
          <div className="text-xl font-bold text-content-primary">{config.parameters.c}</div>
        </div>
        <div>
          <div className="text-sm text-content-secondary">Ignition Prob (ρ)</div>
          <div className="text-xl font-bold text-content-primary">
            {config.parameters.rho_ignite}
          </div>
        </div>
        <div>
          <div className="text-sm text-content-secondary">Min Nights</div>
          <div className="text-xl font-bold text-content-primary">{config.parameters.N_min}</div>
        </div>
        <div>
          <div className="text-sm text-content-secondary">Spark Prob</div>
          <div className="text-xl font-bold text-content-primary">{config.parameters.p_spark}</div>
        </div>
        <div>
          <div className="text-sm text-content-secondary">Spark Duration</div>
          <div className="text-xl font-bold text-content-primary">
            {config.parameters.N_spark}
          </div>
        </div>
        <div>
          <div className="text-sm text-content-secondary">Agents</div>
          <div className="text-xl font-bold text-content-primary">
            {config.parameters.num_agents}
          </div>
        </div>
      </div>
    </div>
  );
}

export const ScenarioOverview = memo(ScenarioOverviewImpl);
