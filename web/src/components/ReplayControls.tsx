import React from 'react';
import { Play, Pause, SkipBack, SkipForward, RotateCcw } from 'lucide-react';

interface ReplayControlsProps {
  currentNight: number;
  totalNights: number;
  isPlaying: boolean;
  speed: number;
  onPlayPause: () => void;
  onReset: () => void;
  onPrev: () => void;
  onNext: () => void;
  onSpeedChange: (speed: number) => void;
}

const ReplayControls: React.FC<ReplayControlsProps> = ({
  currentNight,
  totalNights,
  isPlaying,
  speed,
  onPlayPause,
  onReset,
  onPrev,
  onNext,
  onSpeedChange
}) => {
  const speedOptions = [
    { label: '0.5x', value: 2000 },
    { label: '1x', value: 1000 },
    { label: '2x', value: 500 },
    { label: '4x', value: 250 }
  ];

  return (
    <div className="card">
      <div className="flex items-center justify-between">
        {/* Progress Bar */}
        <div className="flex-1 mr-4">
          <div className="flex items-center space-x-2 mb-2">
            <span className="text-sm font-medium text-gray-700">Night Progress</span>
            <span className="text-sm text-gray-500">
              {currentNight + 1} / {totalNights}
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${((currentNight + 1) / totalNights) * 100}%` }}
            ></div>
          </div>
        </div>

        {/* Control Buttons */}
        <div className="flex items-center space-x-2">
          <button
            onClick={onReset}
            className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
            title="Reset to beginning"
          >
            <RotateCcw className="w-5 h-5 text-gray-600" />
          </button>

          <button
            onClick={onPrev}
            disabled={currentNight === 0}
            className="p-2 rounded-lg hover:bg-gray-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title="Previous night"
          >
            <SkipBack className="w-5 h-5 text-gray-600" />
          </button>

          <button
            onClick={onPlayPause}
            className="p-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white transition-colors"
            title={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? (
              <Pause className="w-5 h-5" />
            ) : (
              <Play className="w-5 h-5" />
            )}
          </button>

          <button
            onClick={onNext}
            disabled={currentNight >= totalNights - 1}
            className="p-2 rounded-lg hover:bg-gray-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title="Next night"
          >
            <SkipForward className="w-5 h-5 text-gray-600" />
          </button>
        </div>

        {/* Speed Control */}
        <div className="ml-6 flex items-center space-x-2">
          <span className="text-sm font-medium text-gray-700">Speed:</span>
          <select
            value={speed}
            onChange={(e) => onSpeedChange(parseInt(e.target.value))}
            className="text-sm border border-gray-300 rounded px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {speedOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Status */}
      <div className="mt-4 text-sm text-gray-600">
        {isPlaying ? (
          <span className="text-green-600">▶ Playing at {speedOptions.find(opt => opt.value === speed)?.label}</span>
        ) : (
          <span>⏸ Paused</span>
        )}
      </div>
    </div>
  );
};

export default ReplayControls;
