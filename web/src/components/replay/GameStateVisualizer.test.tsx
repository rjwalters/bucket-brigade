import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import type { GameNight, GameReplay, HouseState } from '../../types'
import GameStateVisualizer from './GameStateVisualizer'

// Minimal-but-valid fixture matching the schemas in utils/schemas.ts.
const scenario: GameReplay['scenario'] = {
  prob_fire_spreads_to_neighbor: 0.3,
  prob_solo_agent_extinguishes_fire: 1.2,
  prob_house_catches_fire: 0.05,
  team_reward_house_survives: 5,
  team_penalty_house_burns: 3,
  reward_own_house_survives: 100,
  reward_other_house_survives: 50,
  penalty_own_house_burns: 0,
  penalty_other_house_burns: 0,
  cost_to_work_one_night: 0.1,
  min_nights: 10,
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
