import React from 'react';
import { Flame } from 'lucide-react';

// Hero / title section shown at the top of the SimpleDashboard page.
// Pure presentational component (no state).
export const HeroSection: React.FC = React.memo(() => {
  return (
    <div className="text-center bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-950 dark:to-red-950 rounded-lg p-8 border border-orange-200 dark:border-orange-800">
      <div className="flex items-center justify-center mb-4">
        <Flame className="w-12 h-12 text-orange-600 mr-3" />
        <h1 className="text-5xl font-bold text-gray-900 dark:text-gray-100">Bucket Brigade</h1>
      </div>
      <p className="text-xl text-gray-700 dark:text-gray-300 mb-2 max-w-3xl mx-auto">
        Watch cooperation emerge (or fail) in a frontier town facing fire
      </p>
      <p className="text-base text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
        A circle of houses and fire that spreads relentlessly. Will the Agents work together or let the town burn?
      </p>
    </div>
  );
});

HeroSection.displayName = 'HeroSection';
