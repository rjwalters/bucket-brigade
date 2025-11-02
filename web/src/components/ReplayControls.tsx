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
  className?: string;
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
  onSpeedChange,
  className = ''
}) => {
  const speedOptions = [
    { label: '0.5x', value: 2000, multiplier: 0.5 },
    { label: '1x', value: 1000, multiplier: 1 },
    { label: '2x', value: 500, multiplier: 2 },
    { label: '4x', value: 250, multiplier: 4 }
  ];

  const progressPercentage = totalNights > 0 ? ((currentNight + 1) / totalNights) * 100 : 0;

  return (
    <div className={`replay-controls bg-white rounded-lg shadow-md p-6 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Playback Controls</h3>
        <div className="text-sm text-gray-600">
          Night {currentNight + 1} of {totalNights}
        </div>
      </div>

      {/* Progress Bar */}
      <div className="mb-4">
        <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
          <div
            className="bg-gradient-to-r from-blue-500 to-blue-600 h-full rounded-full transition-all duration-300 ease-out"
            style={{ width: `${progressPercentage}%` }}
          ></div>
        </div>
      </div>

      {/* Control Buttons */}
      <div className="flex items-center justify-center space-x-4 mb-4">
        <button
          onClick={onReset}
          className="control-button p-3 rounded-full hover:bg-gray-100 transition-colors"
          title="Reset to beginning"
        >
          <RotateCcw className="w-5 h-5 text-gray-600" />
        </button>

        <button
          onClick={onPrev}
          disabled={currentNight === 0}
          className="control-button p-3 rounded-full hover:bg-gray-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          title="Previous night"
        >
          <SkipBack className="w-5 h-5 text-gray-600" />
        </button>

        <button
          onClick={onPlayPause}
          className="control-button p-4 rounded-full bg-blue-600 hover:bg-blue-700 text-white transition-colors shadow-lg"
          title={isPlaying ? 'Pause replay' : 'Start replay'}
        >
          {isPlaying ? (
            <Pause className="w-6 h-6" />
          ) : (
            <Play className="w-6 h-6 ml-1" />
          )}
        </button>

        <button
          onClick={onNext}
          disabled={currentNight >= totalNights - 1}
          className="control-button p-3 rounded-full hover:bg-gray-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          title="Next night"
        >
          <SkipForward className="w-5 h-5 text-gray-600" />
        </button>
      </div>

      {/* Speed Control */}
      <div className="flex items-center justify-center space-x-2">
        <span className="text-sm font-medium text-gray-700">Speed:</span>
        <select
          value={speed}
          onChange={(e) => onSpeedChange(parseInt(e.target.value))}
          className="text-sm border border-gray-300 rounded px-3 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        >
          {speedOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <span className="text-xs text-gray-500">
          ({speedOptions.find(opt => opt.value === speed)?.multiplier}x speed)
        </span>
      </div>

      {/* Status Indicator */}
      <div className="mt-4 text-center">
        <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm ${
          isPlaying
            ? 'bg-green-100 text-green-800'
            : 'bg-gray-100 text-gray-800'
        }`}>
          <div className={`w-2 h-2 rounded-full mr-2 ${
            isPlaying ? 'bg-green-500 animate-pulse' : 'bg-gray-500'
          }`}></div>
          {isPlaying ? 'Playing' : 'Paused'}
        </div>
      </div>
    </div>
  );
};

export default ReplayControls;
