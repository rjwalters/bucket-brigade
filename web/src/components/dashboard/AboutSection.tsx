import React, { useState } from 'react';
import { Info } from 'lucide-react';

/**
 * Collapsible "About Bucket Brigade" panel shown at the bottom of the
 * SimpleDashboard. Owns its own open/closed state so the parent page can
 * stay focused on the simulation configuration flow.
 */
export const AboutSection: React.FC = React.memo(() => {
  const [showInfo, setShowInfo] = useState<boolean>(false);

  return (
    <div className="card bg-gray-50 dark:bg-gray-800 border-gray-200 dark:border-gray-700">
      <button
        onClick={() => setShowInfo(!showInfo)}
        className="w-full flex items-center justify-between text-left"
      >
        <div className="flex items-center gap-2">
          <Info className="w-5 h-5 text-blue-600 dark:text-blue-400" />
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">About Bucket Brigade</h3>
        </div>
        <span className="text-gray-500 dark:text-gray-400">{showInfo ? '▼' : '▶'}</span>
      </button>

      {showInfo && (
        <div className="mt-4 space-y-3 text-sm text-gray-700 dark:text-gray-300">
          <p>
            Bucket Brigade is a multi-agent cooperation game where agents must work together
            to save a ring of 10 houses from fire. Each house is owned by one agent (with ownership
            assigned round-robin). Each night, agents choose which house to help and whether
            to work or rest.
          </p>
          <p>
            <strong>The Challenge:</strong> Fire spreads between neighboring houses, but agents
            can extinguish fires by working. However, work has a cost, creating a tension between
            protecting one's own houses and helping the collective.
          </p>
          <p>
            <strong>Agent Types:</strong> Different archetypes employ different strategies:
            Firefighters prioritize fire suppression, Coordinators plan strategically, Heroes
            take risks, Free Riders minimize effort, and Liars may signal incorrectly.
          </p>
          <p>
            <strong>Research Goal:</strong> Understanding which strategies succeed in different
            scenarios helps us learn about cooperation, trust, and coordination in multi-agent
            systems.
          </p>
        </div>
      )}
    </div>
  );
});

AboutSection.displayName = 'AboutSection';
