import { useState, useEffect, useCallback } from 'react';
import type { GameReplay } from '../types';

export interface UseReplayPlaybackResult {
  currentStep: number;
  currentNight: number;
  phase: 'day' | 'night';
  displayNight: number;
  totalSteps: number;
  isPlaying: boolean;
  speed: number;
  play: () => void;
  pause: () => void;
  togglePlayPause: () => void;
  reset: () => void;
  stepForward: () => void;
  stepBackward: () => void;
  seek: (step: number) => void;
  setSpeed: (speed: number) => void;
}

/**
 * Hook that manages playback state for a game replay.
 *
 * Step indexing: even steps are day phases, odd steps are night phases.
 * - currentNight = floor(currentStep / 2)
 * - phase = 'day' if currentStep is even, otherwise 'night'
 * - Day phase displays the previous night's house state; night phase displays current night's state.
 */
export function useReplayPlayback(
  selectedGame: GameReplay | null
): UseReplayPlaybackResult {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1000); // milliseconds per step

  const totalSteps = selectedGame ? selectedGame.nights.length * 2 : 0;
  const currentNight = Math.floor(currentStep / 2);
  const phase: 'day' | 'night' = currentStep % 2 === 0 ? 'day' : 'night';
  // Day phase shows state from previous night (or initial for day 0).
  // Night phase shows state from current night.
  const displayNight = phase === 'day' ? Math.max(0, currentNight - 1) : currentNight;

  // Reset playback when the selected game changes
  useEffect(() => {
    setCurrentStep(0);
    setIsPlaying(false);
  }, [selectedGame]);

  // Auto-play loop
  useEffect(() => {
    if (!isPlaying || !selectedGame) return;

    const interval = setInterval(() => {
      setCurrentStep((prev) => {
        const maxStep = selectedGame.nights.length * 2 - 1;
        if (prev >= maxStep) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, speed);

    return () => clearInterval(interval);
  }, [isPlaying, selectedGame, speed]);

  const play = useCallback(() => setIsPlaying(true), []);
  const pause = useCallback(() => setIsPlaying(false), []);
  const togglePlayPause = useCallback(() => setIsPlaying((p) => !p), []);

  const reset = useCallback(() => {
    setCurrentStep(0);
    setIsPlaying(false);
  }, []);

  const stepBackward = useCallback(() => {
    setCurrentStep((prev) => Math.max(0, prev - 1));
  }, []);

  const stepForward = useCallback(() => {
    if (!selectedGame) return;
    const maxStep = selectedGame.nights.length * 2 - 1;
    setCurrentStep((prev) => Math.min(maxStep, prev + 1));
  }, [selectedGame]);

  const seek = useCallback(
    (step: number) => {
      if (!selectedGame) return;
      const maxStep = Math.max(0, selectedGame.nights.length * 2 - 1);
      setCurrentStep(Math.min(Math.max(0, step), maxStep));
    },
    [selectedGame]
  );

  return {
    currentStep,
    currentNight,
    phase,
    displayNight,
    totalSteps,
    isPlaying,
    speed,
    play,
    pause,
    togglePlayPause,
    reset,
    stepForward,
    stepBackward,
    seek,
    setSpeed,
  };
}
