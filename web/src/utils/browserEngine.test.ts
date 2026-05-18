import { describe, it, expect } from 'vitest'
import { BrowserBucketBrigade, type Scenario } from './browserEngine'

/**
 * Issue #252 / #331: two-phase commitment mode is supported by the Rust
 * core/WASM and PyO3 surfaces, but the pure-JS fallback engine does not
 * implement the round-1 signal channel. The constructor must reject
 * two-phase scenarios at construction time so callers can route through
 * the WASM engine instead of silently running simultaneous semantics.
 */

const baseScenario: Scenario = {
  prob_fire_spreads_to_neighbor: 0.15,
  prob_solo_agent_extinguishes_fire: 0.9,
  prob_house_catches_fire: 0.0,
  team_reward_house_survives: 100,
  team_penalty_house_burns: 100,
  reward_own_house_survives: 100,
  reward_other_house_survives: 50,
  penalty_own_house_burns: 0,
  penalty_other_house_burns: 0,
  cost_to_work_one_night: 0.5,
  min_nights: 12,
  num_agents: 4,
}

describe('BrowserBucketBrigade two-phase guard', () => {
  it('throws when commitment_mode is "two_phase"', () => {
    const scenario: Scenario = { ...baseScenario, commitment_mode: 'two_phase' }
    expect(() => new BrowserBucketBrigade(scenario)).toThrowError(
      /does not support commitment_mode="two_phase"/i,
    )
  })

  it('mentions WASM in the error message so callers know how to migrate', () => {
    const scenario: Scenario = { ...baseScenario, commitment_mode: 'two_phase' }
    expect(() => new BrowserBucketBrigade(scenario)).toThrowError(/WASM/i)
  })

  it('accepts simultaneous mode (default, unset)', () => {
    expect(() => new BrowserBucketBrigade(baseScenario)).not.toThrow()
  })

  it('accepts explicit "simultaneous" commitment_mode', () => {
    const scenario: Scenario = { ...baseScenario, commitment_mode: 'simultaneous' }
    expect(() => new BrowserBucketBrigade(scenario)).not.toThrow()
  })
})
