import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import type { GameNight, GameReplay, HouseState } from '../../types'
import GameStateVisualizer from './GameStateVisualizer'

// Minimal-but-valid fixture matching the schemas in utils/schemas.ts.
const scenario: GameReplay['scenario'] = {
  beta: 0.3,
  kappa: 1.2,
  A: 5,
  L: 3,
  c: 0.1,
  rho_ignite: 0.4,
  N_min: 10,
  p_spark: 0.05,
  N_spark: 2,
  num_agents: 2,
}

// 10 houses in the ring (required by the schema). 0 = SAFE.
const houses: HouseState[] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

const night: GameNight = {
  night: 0,
  houses,
  signals: [0, 0],
  locations: [0, 1],
  actions: [
    [0, 0],
    [1, 0],
  ],
  rewards: [0, 0],
}

const replay: GameReplay = {
  scenario,
  nights: [night],
  archetypes: ['cooperative', 'cooperative'],
}

describe('GameStateVisualizer', () => {
  it('renders a sun indicator during the day phase', () => {
    render(
      <GameStateVisualizer
        selectedGame={replay}
        currentNightData={night}
        displayHouses={houses}
        phase="day"
      />
    )
    expect(screen.getByText('☀️')).toBeInTheDocument()
  })

  it('renders a moon indicator during the night phase', () => {
    render(
      <GameStateVisualizer
        selectedGame={replay}
        currentNightData={night}
        displayHouses={houses}
        phase="night"
      />
    )
    expect(screen.getByText('🌙')).toBeInTheDocument()
  })
})
