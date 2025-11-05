import React, { useState, useEffect } from 'react';
import { Play, Pause, SkipBack, SkipForward, RotateCcw } from 'lucide-react';
import type { GameNight } from '../types';

interface ReplayControlsProps {
  currentNight: number;
  totalNights: number;
  isPlaying: boolean;
  speed: number;
  phase?: 'day' | 'night';
  onPlayPause: () => void;
  onReset: () => void;
  onPrev: () => void;
  onNext: () => void;
  onSpeedChange: (speed: number) => void;
  nights?: GameNight[]; // Optional: for showing fire event markers
  className?: string;
}

const ReplayControls: React.FC<ReplayControlsProps> = ({
  currentNight,
  totalNights,
  isPlaying,
  speed,
  phase = 'day',
  onPlayPause,
  onReset,
  onPrev,
  onNext,
  onSpeedChange,
  nights,
  className = ''
}) => {
  const speedOptions = [
    { label: '0.5x', value: 2000, multiplier: 0.5 },
    { label: '1x', value: 1000, multiplier: 1 },
    { label: '2x', value: 500, multiplier: 2 },
    { label: '4x', value: 250, multiplier: 4 }
  ];

  const progressPercentage = totalNights > 0 ? ((currentNight + 1) / totalNights) * 100 : 0;

  // Find nights where fires occurred (any house became burning)
  const fireEventNights = nights ? nights.reduce((acc, night, index) => {
    // Check if any house is burning (state = 1) in this night
    const hasFire = night.houses.some(house => house === 1);
    if (hasFire && index > 0) {
      // Check if this is a NEW fire (wasn't burning in previous night)
      const prevNight = nights[index - 1];
      const newFires = night.houses.some((house, houseIndex) =>
        house === 1 && prevNight.houses[houseIndex] !== 1
      );
      if (newFires) {
        acc.push(index);
      }
    } else if (hasFire && index === 0) {
      acc.push(index);
    }
    return acc;
  }, [] as number[]) : [];

  return (
    <div className={`replay-controls bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Playback Controls</h3>
        <div className="flex items-center space-x-4">
        {/* Day/Night Indicator */}
        <div className={`flex items-center space-x-2 text-sm px-3 py-1 rounded-full ${
        phase === 'day'
          ? 'bg-yellow-100 dark:bg-yellow-900 text-yellow-800 dark:text-yellow-200'
        : 'bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200'
        }`}>
          <span className="text-lg">
            {phase === 'day' ? '☀️' : '🌙'}
        </span>
          <span className="font-semibold">
          {phase === 'day' ? `Day ${currentNight + 1}` : `Night ${currentNight + 1}`}
          </span>
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-400">
            Night {currentNight + 1} of {totalNights}
          </div>
        </div>
      </div>

      {/* Progress Bar with Fire Event Markers */}
      <div className="mb-4 relative">
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 overflow-hidden">
          <div
            className="bg-gradient-to-r from-blue-500 to-blue-600 h-full rounded-full transition-all duration-300 ease-out"
            style={{ width: `${progressPercentage}%` }}
          ></div>
        </div>
        {/* Fire event markers */}
        {fireEventNights.map((nightIndex) => {
          const markerPosition = ((nightIndex + 1) / totalNights) * 100;
          return (
            <div
              key={`fire-${nightIndex}`}
              className="absolute top-0 bottom-0 w-0.5 bg-red-500"
              style={{ left: `${markerPosition}%` }}
              title={`Fire event at night ${nightIndex + 1}`}
            >
              <div className="absolute -top-1 -left-1 w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
            </div>
          );
        })}
      </div>

      {/* Control Buttons */}
      <div className="flex items-center justify-center space-x-4 mb-4">
        <button
          onClick={onReset}
          className="control-button p-3 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          title="Reset to beginning"
        >
          <RotateCcw className="w-5 h-5 text-gray-600 dark:text-gray-400" />
        </button>

        <button
          onClick={onPrev}
          disabled={currentNight === 0}
          className="control-button p-3 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          title="Previous night"
        >
          <SkipBack className="w-5 h-5 text-gray-600 dark:text-gray-400" />
        </button>

        <button
          onClick={onPlayPause}
          className="control-button p-4 rounded-full bg-blue-600 hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 text-white transition-colors shadow-lg"
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
          className="control-button p-3 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          title="Next night"
        >
          <SkipForward className="w-5 h-5 text-gray-600 dark:text-gray-400" />
        </button>
      </div>

      {/* Speed Control */}
      <div className="flex items-center justify-center space-x-2">
        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Speed:</span>
        <select
          value={speed}
          onChange={(e) => onSpeedChange(parseInt(e.target.value))}
          className="text-sm border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 rounded px-3 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400 focus:border-transparent"
        >
          {speedOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <span className="text-xs text-gray-500 dark:text-gray-400">
          ({speedOptions.find(opt => opt.value === speed)?.multiplier}x speed)
        </span>
      </div>

      {/* Status Indicator */}
      <div className="mt-4 text-center">
        <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm ${
          isPlaying
            ? 'bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200'
            : 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200'
        }`}>
          <div className={`w-2 h-2 rounded-full mr-2 ${
            isPlaying ? 'bg-green-500 dark:bg-green-400 animate-pulse' : 'bg-gray-500 dark:bg-gray-400'
          }`}></div>
          {isPlaying ? 'Playing' : 'Paused'}
        </div>
      </div>
    </div>
  );
};

export default ReplayControls;
