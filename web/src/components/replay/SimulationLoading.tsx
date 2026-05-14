import React from 'react';
import { Loader2 } from 'lucide-react';

interface SimulationLoadingProps {
  completed: number;
  total: number;
}

const SimulationLoading: React.FC<SimulationLoadingProps> = ({ completed, total }) => {
  const progress = total > 0 ? (completed / total) * 100 : 0;

  return (
    <div className="text-center py-12">
      <div className="max-w-md mx-auto">
        <Loader2 className="w-16 h-16 text-blue-600 dark:text-blue-400 mx-auto mb-4 animate-spin" />
        <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-4">
          Running Simulation
        </h2>
        <p className="text-gray-600 dark:text-gray-400 mb-6">
          Simulating 100 games to gather statistics...
        </p>
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-4 mb-2">
          <div
            className="bg-blue-600 h-4 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          {completed} / {total} games completed
        </p>
      </div>
    </div>
  );
};

export default SimulationLoading;
