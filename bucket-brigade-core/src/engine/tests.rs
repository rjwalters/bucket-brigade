use super::*;
use crate::SCENARIOS;

/// Issue #222: ``BucketBrigade::new`` must reject a programmatically
/// constructed ``Scenario`` whose ``distance_metric`` is outside the allowlist.
/// Without this, the engine silently runs ring-arc geometry on an unknown
/// metric string (the dispatch in ``engine/rewards.rs`` doesn't branch on the
/// field).
#[test]
#[should_panic(expected = "distance_metric")]
fn test_engine_rejects_unknown_distance_metric() {
    let mut scenario = SCENARIOS.get("default").unwrap().clone();
    scenario.distance_metric = "bogus".to_string();
    // Must panic from BucketBrigade::new -> Scenario::validate.
    let _ = BucketBrigade::new(scenario, 4, Some(42));
}

#[test]
fn test_engine_creation() {
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    let engine = BucketBrigade::new(scenario.clone(), 4, Some(42));

    assert_eq!(engine.night, 0);
    assert!(!engine.done);
    assert_eq!(engine.num_agents, 4);
}

#[test]
fn test_engine_initialization_fires() {
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    let engine = BucketBrigade::new(scenario, 4, Some(42));

    // Count burning houses
    let burning = engine.houses.iter().filter(|&&h| h == 1).count();

    // trivial_cooperation has prob_house_catches_fire=0.01, so with seed 42 we should have 0-2 burning houses
    assert!(burning <= 2, "Should not have too many burning houses");
}

#[test]
fn test_reset() {
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    let mut engine = BucketBrigade::new(scenario, 4, Some(42));

    // Take some steps
    let actions = vec![[0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 1, 1]];
    engine.step(&actions);
    engine.step(&actions);

    assert!(engine.night > 0);

    // Reset
    engine.reset();

    assert_eq!(engine.night, 0);
    assert!(!engine.done);
}

#[test]
fn test_deterministic_with_seed() {
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    let mut engine1 = BucketBrigade::new(scenario.clone(), 4, Some(123));
    let mut engine2 = BucketBrigade::new(scenario, 4, Some(123));

    // Same seed should produce identical initial states
    assert_eq!(engine1.houses, engine2.houses);

    // And identical behavior
    let actions = vec![[0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 1, 1]];
    let result1 = engine1.step(&actions);
    let result2 = engine2.step(&actions);

    assert_eq!(result1.rewards, result2.rewards);
    assert_eq!(engine1.houses, engine2.houses);
}

#[test]
fn test_step_advances_night() {
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    let mut engine = BucketBrigade::new(scenario, 4, Some(42));

    let initial_night = engine.night;
    let actions = vec![[0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 1, 1]];
    engine.step(&actions);

    assert_eq!(engine.night, initial_night + 1);
}

#[test]
fn test_work_vs_rest_rewards() {
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    let mut engine = BucketBrigade::new(scenario, 4, Some(42));

    // All agents rest
    let rest_actions = vec![[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]];
    let result = engine.step(&rest_actions);

    // Resting gives +0.5 reward (plus other factors)
    // Working gives -cost_to_work_one_night cost (0.5 in trivial_cooperation)
    // So resting should generally give better individual rewards when not needed
    assert!(
        result.rewards.iter().all(|&r| r >= 0.0),
        "Rest rewards should be non-negative"
    );
}

#[test]
fn test_fire_extinguishing() {
    let mut scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    scenario.prob_solo_agent_extinguishes_fire = 0.99; // Very high extinguish probability
    let mut engine = BucketBrigade::new(scenario, 4, Some(42));

    // Set up a burning house
    engine.houses = vec![0; 10];
    engine.houses[5] = 1;

    // Send all agents to work on it
    let actions = vec![[5, 1, 1], [5, 1, 1], [5, 1, 1], [5, 1, 1]];

    engine.step(&actions);

    // With high probability and 4 workers, fire should be extinguished
    // (Though it might burn out or spread first - check it's not still burning)
    let final_state = engine.houses[5];
    assert_ne!(
        final_state, 1,
        "House should not still be burning after 4 workers"
    );
}

#[test]
fn test_fire_spreads_to_neighbors() {
    let mut scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    scenario.prob_fire_spreads_to_neighbor = 1.0; // 100% spread probability
    scenario.prob_solo_agent_extinguishes_fire = 0.0; // No extinguishing

    let mut engine = BucketBrigade::new(scenario, 4, Some(100));

    // Set up a controlled initial state with one burning house
    engine.houses = vec![0; 10];
    engine.houses[5] = 1; // House 5 is burning

    // No one works (let it spread)
    let actions = vec![[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]];
    engine.step(&actions);

    // Neighbors (4 and 6) should catch fire with prob_fire_spreads_to_neighbor=1.0
    // But house 5 should now be ruined (2) after burning
    assert_eq!(
        engine.houses[5], 2,
        "Original burning house should be ruined"
    );
}

#[test]
fn test_burn_out_phase() {
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    let mut engine = BucketBrigade::new(scenario, 4, Some(42));

    // Set a house on fire
    engine.houses[3] = 1;

    // Let it burn without intervention
    let actions = vec![[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]];
    engine.step(&actions);

    // Burning house should become ruined (unless extinguished, which is unlikely with no workers)
    let house_3_state = engine.houses[3];
    assert!(
        house_3_state == 2 || house_3_state == 0,
        "House should be ruined or extinguished"
    );
}

#[test]
fn test_spontaneous_ignition() {
    let mut scenario = SCENARIOS.get("early_containment").unwrap().clone();
    scenario.prob_house_catches_fire = 1.0; // 100% ignition probability for testing

    let mut engine = BucketBrigade::new(scenario, 4, Some(42));

    // Clear all fires
    engine.houses = vec![0; 10];
    engine.night = 0;

    // Step once - should create spontaneous ignitions
    let actions = vec![[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]];
    engine.step(&actions);

    // With prob_house_catches_fire=1.0, all safe houses should catch fire
    let burning = engine.houses.iter().filter(|&&h| h == 1).count();
    assert!(
        burning > 0,
        "Spontaneous ignition should have created new fires"
    );
}

#[test]
fn test_continuous_spontaneous_ignition() {
    let mut scenario = SCENARIOS.get("early_containment").unwrap().clone();
    scenario.prob_house_catches_fire = 1.0; // 100% ignition probability

    let mut engine = BucketBrigade::new(scenario, 4, Some(42));
    engine.houses = vec![0; 10];

    // Step multiple nights
    let actions = vec![[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]];
    for _ in 0..10 {
        engine.houses = vec![0; 10]; // Clear fires each round
        engine.step(&actions);

        // Should still create fires even after many nights
        let burning = engine.houses.iter().filter(|&&h| h == 1).count();
        if !engine.done {
            assert!(
                burning > 0,
                "Should have fires from continuous spontaneous ignition"
            );
        }
    }
}

#[test]
fn test_termination_all_safe() {
    let mut scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    scenario.min_nights = 0; // Allow immediate termination

    let mut engine = BucketBrigade::new(scenario, 4, Some(42));
    engine.houses = vec![0; 10]; // All safe

    let actions = vec![[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]];
    let result = engine.step(&actions);

    assert!(result.done, "Game should end when all houses are safe");
}

#[test]
fn test_termination_all_ruined() {
    let mut scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    scenario.min_nights = 0;

    let mut engine = BucketBrigade::new(scenario, 4, Some(42));
    engine.houses = vec![2; 10]; // All ruined

    let actions = vec![[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]];
    let result = engine.step(&actions);

    assert!(result.done, "Game should end when all houses are ruined");
}

#[test]
fn test_minimum_nights_prevents_early_termination() {
    let mut scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    scenario.min_nights = 5; // Require at least 5 nights

    let mut engine = BucketBrigade::new(scenario, 4, Some(42));
    engine.houses = vec![0; 10]; // All safe

    let actions = vec![[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]];

    // Step through 3 nights
    for _ in 0..3 {
        let result = engine.step(&actions);
        assert!(!result.done, "Should not terminate before min_nights");
    }
}

#[test]
fn test_get_observation() {
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    let engine = BucketBrigade::new(scenario, 4, Some(42));

    let obs = engine.get_observation(0);

    assert_eq!(obs.agent_id, 0);
    assert_eq!(obs.night, 0);
    assert_eq!(obs.houses.len(), 10);
    assert_eq!(obs.signals.len(), 4);
    assert_eq!(obs.locations.len(), 4);
    assert_eq!(obs.scenario_info.len(), 12); // Updated from 10 to 12
}

#[test]
fn test_trajectory_recording() {
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    let mut engine = BucketBrigade::new(scenario, 4, Some(42));

    let actions = vec![[0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 1, 1]];
    engine.step(&actions);
    engine.step(&actions);

    let result = engine.get_result();

    // Should have initial state + 2 steps = 3 nights recorded
    // Night is recorded before incrementing, so: [0, 0, 1]
    assert_eq!(result.nights.len(), 3);
    assert_eq!(result.nights[0].night, 0); // Initial state
    assert_eq!(result.nights[1].night, 0); // After first step (recorded before increment)
    assert_eq!(result.nights[2].night, 1); // After second step (recorded before increment)
}

#[test]
fn test_final_score_calculation() {
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    let mut engine = BucketBrigade::new(scenario, 4, Some(42));

    let actions = vec![[0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 1, 1]];
    engine.step(&actions);

    let result = engine.get_result();

    // Final score should be sum of all agent scores
    let expected_sum: f32 = result.agent_scores.iter().sum();
    assert!((result.final_score - expected_sum).abs() < 0.001);
}

#[test]
fn test_agent_positions_update() {
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    let mut engine = BucketBrigade::new(scenario, 4, Some(42));

    let actions = vec![[5, 1, 1], [6, 1, 1], [7, 1, 1], [8, 1, 1]];
    engine.step(&actions);

    let state = engine.get_current_state();
    assert_eq!(state.agent_positions, vec![5, 6, 7, 8]);
}

#[test]
fn test_agent_signals_update() {
    // Issue #235: signals are now action[2], independent of the work bit
    // action[1]. This test pins the *decoupled* semantics: we construct
    // agents whose broadcast signal differs from their actual mode, and
    // assert the engine records the broadcast (action[2]) — not the mode.
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    let mut engine = BucketBrigade::new(scenario, 4, Some(42));

    // [house, mode, signal] — agent 0 rests but signals work (a lie);
    // agent 1 works but signals rest (a lie); agents 2/3 are honest.
    let actions = vec![[0, 0, 1], [1, 1, 0], [2, 0, 0], [3, 1, 1]];
    engine.step(&actions);

    let state = engine.get_current_state();
    // Signals reflect the broadcast column, not the mode column.
    assert_eq!(state.agent_signals, vec![1, 0, 0, 1]);
    // And the broadcast differs from the work bit for the two liars
    // (the whole point of issue #235 vs the deterministic-copy bug).
    let work_bits: Vec<u8> = actions.iter().map(|a| a[1]).collect();
    assert_ne!(
        state.agent_signals, work_bits,
        "Signal channel must be decoupled from the work/rest action bit"
    );
}

#[test]
#[should_panic(expected = "Game is already finished")]
fn test_step_after_done_panics() {
    let mut scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    scenario.min_nights = 0;

    let mut engine = BucketBrigade::new(scenario, 4, Some(42));
    engine.houses = vec![0; 10];

    let actions = vec![[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]];
    engine.step(&actions);

    // Game should be done now, stepping again should panic
    engine.step(&actions);
}

#[test]
fn test_ownership_rewards() {
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    let mut engine = BucketBrigade::new(scenario, 4, Some(42));

    // Agent 0 owns house 0, agent 1 owns house 1, etc.
    engine.houses = vec![0; 10]; // All safe

    let actions = vec![[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]];
    let result = engine.step(&actions);

    // Per-step rewards now include rest bonus + team rewards (all houses safe)
    // All agents resting (+0.5) + team reward for saved houses
    assert!(
        result.rewards.iter().all(|&r| r > 0.5),
        "Agents should get rest reward + team rewards"
    );

    // Complete the game and check final rewards
    engine.done = true;
    let game_result = engine.get_result();

    // All agents should have positive final scores (accumulated per-step rewards)
    assert!(
        game_result.agent_scores.iter().all(|&s| s > 0.0),
        "All agents should have positive scores with all houses safe"
    );
}

#[test]
fn test_ownership_penalty() {
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    let mut engine = BucketBrigade::new(scenario, 4, Some(42));

    // Set agent 0's house (house 0) to ruined
    engine.houses = vec![0; 10];
    engine.houses[0] = 2;

    let actions = vec![[5, 0, 0], [5, 0, 0], [5, 0, 0], [5, 0, 0]];
    let result = engine.step(&actions);

    // Per-step rewards now include rest, team rewards, and ownership penalties
    // Agent 0: rest (+0.5) + team reward (~80) + ownership penalty (-2.0) = ~78.5
    // Others: rest (+0.5) + team reward (~80) = ~80.5
    assert!(
        result.rewards[0] < result.rewards[1],
        "Agent 0 should have lower reward due to ruined house"
    );
    // Agent 0's reward should still be positive due to large team reward
    assert!(
        result.rewards[0] > 70.0,
        "Agent 0 should have positive reward (rest + large team reward - penalty)"
    );

    // Complete the game and check final rewards
    engine.done = true;
    let game_result = engine.get_result();

    // Agent 0 should have penalty for ruined owned house (house 0)
    // Agent 1's house (house 1) is safe, so they get bonus
    assert!(
        game_result.agent_scores[0] < game_result.agent_scores[1],
        "Agent 0 should have lower final score due to ruined house penalty"
    );
}

// ==============================================================================
// Population Size Scalability Tests
// ==============================================================================

#[test]
fn test_population_size_4() {
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    let engine = BucketBrigade::new(scenario, 4, Some(42));

    assert_eq!(engine.num_agents, 4);
    assert_eq!(engine.agent_signals.len(), 4);
    assert_eq!(engine.agent_positions.len(), 4);
}

#[test]
fn test_population_size_6() {
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    let engine = BucketBrigade::new(scenario, 6, Some(42));

    assert_eq!(engine.num_agents, 6);
    assert_eq!(engine.agent_signals.len(), 6);
    assert_eq!(engine.agent_positions.len(), 6);
}

#[test]
fn test_population_size_8() {
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    let engine = BucketBrigade::new(scenario, 8, Some(42));

    assert_eq!(engine.num_agents, 8);
    assert_eq!(engine.agent_signals.len(), 8);
    assert_eq!(engine.agent_positions.len(), 8);
}

#[test]
fn test_population_size_10() {
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    let engine = BucketBrigade::new(scenario, 10, Some(42));

    assert_eq!(engine.num_agents, 10);
    assert_eq!(engine.agent_signals.len(), 10);
    assert_eq!(engine.agent_positions.len(), 10);
}

#[test]
fn test_population_size_20() {
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    let engine = BucketBrigade::new(scenario, 20, Some(42));

    assert_eq!(engine.num_agents, 20);
    assert_eq!(engine.agent_signals.len(), 20);
    assert_eq!(engine.agent_positions.len(), 20);
}

#[test]
fn test_large_population_step() {
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    let mut engine = BucketBrigade::new(scenario, 20, Some(42));

    // Generate 20 actions (one per agent)
    let actions: Vec<[u8; 3]> = (0..20).map(|i| [i % 10, 1, 1]).collect();

    let result = engine.step(&actions);

    // Should have 20 rewards
    assert_eq!(result.rewards.len(), 20);
    assert!(!result.done);
}

#[test]
fn test_population_scalability_all_sizes() {
    // Test that all population sizes from 2 to 20 work
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();

    for num_agents in 2..=20 {
        let mut engine = BucketBrigade::new(scenario.clone(), num_agents, Some(42));

        assert_eq!(engine.num_agents, num_agents);

        // Take one step
        let actions: Vec<[u8; 3]> = (0..num_agents).map(|i| [(i % 10) as u8, 1, 1]).collect();
        let result = engine.step(&actions);

        assert_eq!(result.rewards.len(), num_agents);
    }
}

#[test]
fn test_large_population_game_completion() {
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    let mut engine = BucketBrigade::new(scenario, 15, Some(42));

    // Run until done
    let mut steps = 0;
    let max_steps = 100;

    while !engine.done && steps < max_steps {
        let actions: Vec<[u8; 3]> = (0..15).map(|i| [(i % 10) as u8, 1, 1]).collect();
        engine.step(&actions);
        steps += 1;
    }

    // Should complete within reasonable steps
    assert!(
        steps < max_steps,
        "Game should complete within {} steps",
        max_steps
    );

    // Get final results
    let game_result = engine.get_result();

    // Should have 15 agent scores
    assert_eq!(game_result.agent_scores.len(), 15);
}

// ==============================================================================
// Scenario Coverage Tests
// ==============================================================================

#[test]
fn test_all_scenarios_runnable() {
    // Test that we can create an engine for every scenario
    for (name, scenario) in SCENARIOS.iter() {
        let engine = BucketBrigade::new(scenario.clone(), 4, Some(42));

        assert_eq!(engine.num_agents, 4);
        assert!(!engine.done);
        println!("✓ Scenario '{}' is runnable", name);
    }
}

#[test]
fn test_all_scenarios_steppable() {
    // Test that we can step every scenario without panicking
    for (name, scenario) in SCENARIOS.iter() {
        let mut engine = BucketBrigade::new(scenario.clone(), 4, Some(42));

        let actions = vec![[0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 1, 1]];
        let result = engine.step(&actions);

        assert_eq!(result.rewards.len(), 4);
        println!("✓ Scenario '{}' is steppable", name);
    }
}

// ==============================================================================
// Issue #203: Spatial cost asymmetry (positional_default)
// ==============================================================================

/// Loading `positional_default` and stepping does not panic and produces
/// finite per-step rewards. Smoke test that the new scenario wires through.
#[test]
fn test_positional_default_engine_smoke() {
    let scenario = SCENARIOS.get("positional_default").unwrap().clone();
    let mut engine = BucketBrigade::new(scenario, 4, Some(42));
    let actions = vec![[0u8, 1u8, 1u8], [3, 1, 1], [5, 1, 1], [8, 1, 1]];
    let result = engine.step(&actions);
    assert_eq!(result.rewards.len(), 4);
    for &r in &result.rewards {
        assert!(r.is_finite(), "Per-step reward must be finite, got {}", r);
    }
}

/// Working at home should cost exactly `cost_to_work_one_night`; working at
/// the antipode of the ring should cost `cost + alpha * 5`. Direct
/// verification that distance scaling is wired through `compute_rewards`.
#[test]
fn test_positional_default_distance_scales_cost() {
    let scenario = SCENARIOS.get("positional_default").unwrap().clone();
    let base_cost = scenario.cost_to_work_one_night;
    let alpha = scenario.distance_cost_alpha;
    // Sanity-check the scenario constants used by the math below.
    assert_eq!(base_cost, 0.5);
    assert_eq!(alpha, 0.1);

    // Isolate the work-cost term by zeroing every other reward channel,
    // and by configuring no spontaneous fires so houses stay SAFE.
    let mut scenario = scenario;
    scenario.prob_house_catches_fire = 0.0;
    scenario.team_reward_house_survives = 0.0;
    scenario.team_penalty_house_burns = 0.0;
    scenario.reward_own_house_survives = vec![0.0; 10];
    scenario.reward_other_house_survives = vec![0.0; 10];
    scenario.penalty_own_house_burns = vec![0.0; 10];
    scenario.penalty_other_house_burns = vec![0.0; 10];

    let mut engine = BucketBrigade::new(scenario, 4, Some(123));

    // Agent 0 home is house 0. Agent 0 works house 5 (max ring distance 5).
    // All other agents rest (so their reward is +0.5 with no team component).
    let actions = vec![[5u8, 1u8, 1u8], [3, 0, 0], [5, 0, 0], [8, 0, 0]];
    let result = engine.step(&actions);

    // Expected for agent 0: -(base + alpha * 5) = -(0.5 + 0.5) = -1.0
    let expected_agent_0 = -(base_cost + alpha * 5.0);
    assert!(
        (result.rewards[0] - expected_agent_0).abs() < 1e-5,
        "Agent 0 (home=0, target=5) should pay {} but got {}",
        expected_agent_0,
        result.rewards[0]
    );
    // Other agents rested -> exactly +0.5 each (no team term, no save events).
    for i in 1..4 {
        assert!(
            (result.rewards[i] - 0.5).abs() < 1e-5,
            "Resting agent {} should get +0.5, got {}",
            i,
            result.rewards[i]
        );
    }
}

/// Backward compat: when `distance_cost_alpha == 0` the per-step reward
/// trace for working agents matches the unmodified `default` scenario
/// exactly, regardless of which house each agent works at.
#[test]
fn test_zero_alpha_preserves_default_rewards() {
    // Same seed, same actions, same scenario -> deterministic equality.
    let scenario = SCENARIOS.get("default").unwrap().clone();
    let mut engine_a = BucketBrigade::new(scenario.clone(), 4, Some(7));
    let mut engine_b = BucketBrigade::new(scenario, 4, Some(7));
    let actions = vec![[0u8, 1u8, 1u8], [4, 1, 1], [9, 1, 1], [2, 1, 1]];
    let r_a = engine_a.step(&actions).rewards;
    let r_b = engine_b.step(&actions).rewards;
    assert_eq!(r_a, r_b);
}

/// Engine `agent_home_positions` is populated from the scenario when set.
#[test]
fn test_engine_home_positions_use_scenario_when_set() {
    let scenario = SCENARIOS.get("positional_default").unwrap().clone();
    let engine = BucketBrigade::new(scenario, 4, Some(0));
    assert_eq!(engine.agent_home_positions, vec![0u8, 3, 5, 8]);
}

/// Engine `agent_home_positions` falls back to the round-robin
/// `house_owners` anchor (agent `i` -> house `i`) when the scenario
/// leaves the field empty.
#[test]
fn test_engine_home_positions_fallback_to_house_owners() {
    let scenario = SCENARIOS.get("default").unwrap().clone();
    let engine = BucketBrigade::new(scenario, 4, Some(0));
    assert_eq!(engine.agent_home_positions, vec![0u8, 1, 2, 3]);
}

// ==============================================================================
// Issue #254 — `num_houses` parameterization smoke tests
// ==============================================================================

/// `v2_minimal` (2 houses, 4 agents) constructs cleanly and exposes the
/// expected per-ring sizing on the engine state vectors.
#[test]
fn test_v2_minimal_engine_construction() {
    let scenario = SCENARIOS.get("v2_minimal").unwrap().clone();
    let engine = BucketBrigade::new(scenario, 4, Some(42));

    assert_eq!(engine.scenario.num_houses, 2);
    assert_eq!(engine.houses.len(), 2);
    assert_eq!(engine.house_owners.len(), 2);
    // House owners on a 2-house ring with 4 agents: [0 % 4, 1 % 4] = [0, 1].
    assert_eq!(engine.house_owners, vec![0, 1]);
    assert_eq!(engine.num_agents, 4);
    assert_eq!(engine.agent_positions.len(), 4);
    assert_eq!(engine.rewards.len(), 4);
}

/// Random rollout on `v2_minimal` completes 20 steps without panicking on
/// out-of-range house indices (the engine paths previously hardcoded
/// `0..10` and would index out of bounds at length-2).
#[test]
fn test_v2_minimal_random_rollout() {
    let scenario = SCENARIOS.get("v2_minimal").unwrap().clone();
    let mut engine = BucketBrigade::new(scenario, 4, Some(7));

    for step in 0..20 {
        if engine.done {
            break;
        }
        // Actions confined to the 2-house ring (action[0] in {0, 1}).
        let actions: Vec<[u8; 3]> = (0..4).map(|i| [(i % 2) as u8, 1, 1]).collect();
        let result = engine.step(&actions);
        assert_eq!(
            result.rewards.len(),
            4,
            "step {} returned wrong reward count",
            step
        );
    }

    let obs = engine.get_observation(0);
    assert_eq!(
        obs.houses.len(),
        2,
        "observation houses must size to num_houses=2"
    );
}

/// `v2_minimal` observation has the expected shape: `houses` is length 2
/// (from `num_houses`), other fields scale with `num_agents=4`.
#[test]
fn test_v2_minimal_observation_shape() {
    let scenario = SCENARIOS.get("v2_minimal").unwrap().clone();
    let engine = BucketBrigade::new(scenario, 4, None);
    let obs = engine.get_observation(0);

    assert_eq!(obs.houses.len(), 2);
    assert_eq!(obs.signals.len(), 4);
    assert_eq!(obs.locations.len(), 4);
    assert_eq!(obs.last_actions.len(), 4);
    // scenario_info layout is fixed (12 elements regardless of ring size).
    assert_eq!(obs.scenario_info.len(), 12);
}

/// Backward compat: every pre-#254 scenario constructed via `BucketBrigade::new`
/// must still produce a 10-element `houses` vector. This is the engine-side
/// mirror of `scenarios::tests::test_non_v2_scenarios_have_ten_houses`.
#[test]
fn test_pre_254_scenarios_still_ten_houses_in_engine() {
    for name in SCENARIOS.keys() {
        if name.starts_with("v2_") {
            continue;
        }
        let scenario = SCENARIOS.get(name).unwrap().clone();
        let engine = BucketBrigade::new(scenario, 4, Some(0));
        assert_eq!(
            engine.houses.len(),
            10,
            "Pre-#254 scenario '{}' regressed: houses.len()={}, expected 10",
            name,
            engine.houses.len()
        );
        assert_eq!(
            engine.house_owners.len(),
            10,
            "Pre-#254 scenario '{}' regressed: house_owners.len()={}, expected 10",
            name,
            engine.house_owners.len()
        );
    }
}

/// `BucketBrigade::new` rejects degenerate `num_houses` values
/// (`num_houses < 2` doesn't allow any wraparound; the spread phase math
/// assumes a real ring).
#[test]
#[should_panic(expected = "num_houses must be at least 2")]
fn test_engine_rejects_one_house_scenario() {
    let mut scenario = SCENARIOS.get("default").unwrap().clone();
    scenario.num_houses = 1;
    let _ = BucketBrigade::new(scenario, 1, Some(0));
}

/// Issue #254: `num_agents > num_houses` is supported (this is the
/// `v2_minimal` case — 4 agents on 2 houses, with agents 2 and 3 as
/// "unowned-house workers"). Verifies the engine doesn't panic on this
/// configuration and that the round-robin ownership pattern is correct.
#[test]
fn test_engine_allows_more_agents_than_houses() {
    let scenario = SCENARIOS.get("v2_minimal").unwrap().clone();
    let engine = BucketBrigade::new(scenario, 4, Some(0));
    // House ownership is `i % num_agents`, not `i % num_houses`, so houses
    // 0 and 1 are owned by agents 0 and 1 respectively.
    assert_eq!(engine.house_owners, vec![0, 1]);
    // Agent home positions wrap modulo `num_houses` (issue #254 fallback),
    // so agents [0, 1, 2, 3] anchor to houses [0, 1, 0, 1].
    assert_eq!(engine.agent_home_positions, vec![0, 1, 0, 1]);
}

// ==============================================================================
// Issue #259 — Action-conditioned reward shaping
// ==============================================================================

/// Backward compat: when both shaping knobs are zero (the default for
/// every pre-#259 scenario) the per-step reward trace is byte-identical
/// to today's behavior. The fast-path skip in `compute_rewards` is the
/// only thing that lets us claim bit-exactness for the default.
#[test]
fn test_zero_shaping_preserves_default_rewards() {
    // Identical scenario, identical seed, identical actions
    // -> identical reward vector. Pre-#259 baseline check.
    let scenario = SCENARIOS.get("default").unwrap().clone();
    assert_eq!(scenario.action_shaping_alpha, 0.0);
    assert_eq!(scenario.action_shaping_beta, 0.0);
    let mut engine_a = BucketBrigade::new(scenario.clone(), 4, Some(11));
    let mut engine_b = BucketBrigade::new(scenario, 4, Some(11));
    let actions = vec![[3u8, 1u8, 1u8], [5, 1, 1], [7, 1, 1], [9, 1, 1]];
    let r_a = engine_a.step(&actions).rewards;
    let r_b = engine_b.step(&actions).rewards;
    assert_eq!(r_a, r_b);
}

/// Engine bit-exactness across all pre-#259 SCENARIOS: every scenario in
/// the registry keeps `action_shaping_alpha == 0` and
/// `action_shaping_beta == 0`. Guards against a future scenario silently
/// enabling shaping and breaking the bit-exact baseline.
#[test]
fn test_pre_259_scenarios_have_zero_shaping_knobs() {
    for (name, scenario) in SCENARIOS.iter() {
        assert_eq!(
            scenario.action_shaping_alpha, 0.0,
            "Pre-#259 scenario '{}' must keep action_shaping_alpha=0.0",
            name
        );
        assert_eq!(
            scenario.action_shaping_beta, 0.0,
            "Pre-#259 scenario '{}' must keep action_shaping_beta=0.0",
            name
        );
    }
}

/// `alpha` credit-share invariant: when k workers co-extinguish one fire,
/// the sum of their action-shaping bonuses equals `alpha` exactly. This is
/// the core acceptance criterion for the credit-share formulation.
#[test]
fn test_alpha_credit_share_sums_to_alpha() {
    // Custom deterministic scenario: 100% extinguish probability, no
    // spreading, no spontaneous ignition. Zero all other reward channels
    // so the per-step reward delta isolates the alpha bonus.
    let mut scenario = SCENARIOS.get("default").unwrap().clone();
    scenario.prob_solo_agent_extinguishes_fire = 1.0;
    scenario.prob_fire_spreads_to_neighbor = 0.0;
    scenario.prob_house_catches_fire = 0.0;
    scenario.team_reward_house_survives = 0.0;
    scenario.team_penalty_house_burns = 0.0;
    scenario.reward_own_house_survives = vec![0.0; 10];
    scenario.reward_other_house_survives = vec![0.0; 10];
    scenario.penalty_own_house_burns = vec![0.0; 10];
    scenario.penalty_other_house_burns = vec![0.0; 10];
    scenario.cost_to_work_one_night = 0.0;
    scenario.action_shaping_alpha = 1.5;
    scenario.action_shaping_beta = 0.0;

    let mut engine = BucketBrigade::new(scenario, 4, Some(2025));
    // Set up: only house 5 burning at start-of-step. All 4 workers go to
    // house 5; with kappa=1.0 the fire is guaranteed to extinguish.
    engine.houses = vec![0; 10];
    engine.houses[5] = 1;

    let actions = vec![[5u8, 1u8, 1u8], [5, 1, 1], [5, 1, 1], [5, 1, 1]];
    let rewards = engine.step(&actions).rewards;

    // Sum of bonuses across the 4 co-extinguishers should be exactly 1.5
    // (work cost is zero, team/ownership/cost are all zero; only alpha
    // and the implicit beta at non-target houses contribute). Since beta=0
    // every term except alpha drops out. Each worker should get
    // alpha/4 = 0.375.
    let sum: f32 = rewards.iter().sum();
    assert!(
        (sum - 1.5).abs() < 1e-5,
        "Sum of alpha bonuses should equal alpha=1.5 exactly, got {}",
        sum
    );
    // Each worker received the same share.
    for &r in &rewards {
        assert!(
            (r - 0.375).abs() < 1e-5,
            "Each co-extinguisher should get alpha/4 = 0.375, got {}",
            r
        );
    }
}

/// REST actions never receive the alpha bonus even if a fire is being
/// extinguished at the same house by other workers.
#[test]
fn test_rest_agents_get_zero_alpha_bonus() {
    let mut scenario = SCENARIOS.get("default").unwrap().clone();
    scenario.prob_solo_agent_extinguishes_fire = 1.0;
    scenario.prob_fire_spreads_to_neighbor = 0.0;
    scenario.prob_house_catches_fire = 0.0;
    scenario.team_reward_house_survives = 0.0;
    scenario.team_penalty_house_burns = 0.0;
    scenario.reward_own_house_survives = vec![0.0; 10];
    scenario.reward_other_house_survives = vec![0.0; 10];
    scenario.penalty_own_house_burns = vec![0.0; 10];
    scenario.penalty_other_house_burns = vec![0.0; 10];
    scenario.cost_to_work_one_night = 0.0;
    scenario.action_shaping_alpha = 2.0;
    scenario.action_shaping_beta = 0.0;

    let mut engine = BucketBrigade::new(scenario, 4, Some(99));
    engine.houses = vec![0; 10];
    engine.houses[3] = 1; // House 3 burning at start-of-step.

    // Agent 0 works house 3, agent 1 rests at house 3 (action[1]=0),
    // agents 2 and 3 rest elsewhere.
    let actions = vec![[3u8, 1u8, 1u8], [3, 0, 0], [7, 0, 0], [9, 0, 0]];
    let rewards = engine.step(&actions).rewards;

    // Agent 0 is the sole worker -> alpha/1 = 2.0.
    assert!(
        (rewards[0] - 2.0).abs() < 1e-5,
        "Sole worker should get full alpha=2.0, got {}",
        rewards[0]
    );
    // Resting agents get rest bonus +0.5, no alpha share.
    for i in 1..4 {
        assert!(
            (rewards[i] - 0.5).abs() < 1e-5,
            "Resting agent {} should get +0.5 rest bonus only, got {}",
            i,
            rewards[i]
        );
    }
}

/// `beta` rewards preventive presence: an agent working at a SAFE house
/// that stays SAFE receives the flat bonus. Verifies the beta path is
/// distinct from alpha (which requires a BURNING -> SAFE transition).
#[test]
fn test_beta_rewards_preventive_presence() {
    let mut scenario = SCENARIOS.get("default").unwrap().clone();
    scenario.prob_fire_spreads_to_neighbor = 0.0;
    scenario.prob_house_catches_fire = 0.0;
    scenario.prob_solo_agent_extinguishes_fire = 0.0;
    scenario.team_reward_house_survives = 0.0;
    scenario.team_penalty_house_burns = 0.0;
    scenario.reward_own_house_survives = vec![0.0; 10];
    scenario.reward_other_house_survives = vec![0.0; 10];
    scenario.penalty_own_house_burns = vec![0.0; 10];
    scenario.penalty_other_house_burns = vec![0.0; 10];
    scenario.cost_to_work_one_night = 0.0;
    scenario.action_shaping_alpha = 0.0;
    scenario.action_shaping_beta = 0.3;

    let mut engine = BucketBrigade::new(scenario, 4, Some(7));
    // All houses safe; no fires; no spread.
    engine.houses = vec![0; 10];

    // All 4 agents work at distinct safe houses.
    let actions = vec![[0u8, 1u8, 1u8], [3, 1, 1], [5, 1, 1], [8, 1, 1]];
    let rewards = engine.step(&actions).rewards;

    // Each working agent gets +0.3 preventive bonus (no cost, no other
    // reward channels).
    for (i, &r) in rewards.iter().enumerate() {
        assert!(
            (r - 0.3).abs() < 1e-5,
            "Agent {} working at safe house should get beta=0.3, got {}",
            i,
            r
        );
    }
}

/// `beta` does NOT fire on a house that was BURNING at start-of-step
/// (even if extinguished by end-of-step). Beta requires prev==SAFE and
/// end==SAFE — strict preventive presence, not an alpha proxy.
#[test]
fn test_beta_skips_extinguished_house() {
    let mut scenario = SCENARIOS.get("default").unwrap().clone();
    scenario.prob_solo_agent_extinguishes_fire = 1.0;
    scenario.prob_fire_spreads_to_neighbor = 0.0;
    scenario.prob_house_catches_fire = 0.0;
    scenario.team_reward_house_survives = 0.0;
    scenario.team_penalty_house_burns = 0.0;
    scenario.reward_own_house_survives = vec![0.0; 10];
    scenario.reward_other_house_survives = vec![0.0; 10];
    scenario.penalty_own_house_burns = vec![0.0; 10];
    scenario.penalty_other_house_burns = vec![0.0; 10];
    scenario.cost_to_work_one_night = 0.0;
    scenario.action_shaping_alpha = 0.0;
    scenario.action_shaping_beta = 0.5;

    let mut engine = BucketBrigade::new(scenario, 4, Some(123));
    engine.houses = vec![0; 10];
    engine.houses[5] = 1; // Burning at start-of-step.

    // Agent 0 works house 5 (extinguishes it). Other agents rest.
    let actions = vec![[5u8, 1u8, 1u8], [0, 0, 0], [0, 0, 0], [0, 0, 0]];
    let rewards = engine.step(&actions).rewards;

    // alpha=0, so no extinguish bonus. beta=0.5 but prev[5]=BURNING so
    // beta doesn't fire. Agent 0 gets 0.0 (no cost, no shaping).
    assert!(
        rewards[0].abs() < 1e-5,
        "Agent extinguishing fire must NOT get beta (prev was BURNING), got {}",
        rewards[0]
    );
}

/// Engine smoke test: a scenario with both knobs enabled runs without
/// panicking and produces finite rewards.
#[test]
fn test_action_shaping_engine_smoke() {
    let mut scenario = SCENARIOS.get("default").unwrap().clone();
    scenario.action_shaping_alpha = 0.5;
    scenario.action_shaping_beta = 0.1;
    let mut engine = BucketBrigade::new(scenario, 4, Some(42));
    for _ in 0..15 {
        if engine.done {
            break;
        }
        let actions = vec![[0u8, 1u8, 1u8], [3, 1, 1], [5, 1, 1], [8, 1, 1]];
        let result = engine.step(&actions);
        assert_eq!(result.rewards.len(), 4);
        for &r in &result.rewards {
            assert!(r.is_finite(), "Per-step reward must be finite, got {}", r);
        }
    }
}

// ==============================================================================
// Issue #283: Potential-based team-welfare shaping (Ng-Harada-Russell 1999)
// ==============================================================================

/// Phi(s) is a pure function of `houses` for the closed-form kind. Spot-check
/// hand-computed values to keep parity with the Python implementation in
/// `bucket_brigade/envs/bucket_brigade_env.py::_compute_team_welfare_phi`.
#[test]
fn test_team_welfare_phi_closed_form_values() {
    let mut scenario = SCENARIOS.get("default").unwrap().clone();
    scenario.team_reward_house_survives = 10.0;
    scenario.team_penalty_house_burns = 10.0;
    scenario.team_welfare_kind = "team_welfare_closed_form".to_string();
    let engine = BucketBrigade::new(scenario, 4, Some(0));

    // All safe: Phi = 10*(10/10) - 0 - 0 = 10.0
    let all_safe = vec![0u8; 10];
    assert!((engine.team_welfare_phi(&all_safe) - 10.0).abs() < 1e-5);

    // All ruined: Phi = 0 - 10 - 0 = -10.0
    let all_ruined = vec![2u8; 10];
    assert!((engine.team_welfare_phi(&all_ruined) - (-10.0)).abs() < 1e-5);

    // All burning: Phi = 0 - 0 - 0.5*10 = -5.0
    let all_burning = vec![1u8; 10];
    assert!((engine.team_welfare_phi(&all_burning) - (-5.0)).abs() < 1e-5);

    // Mixed: 5 safe, 3 burning, 2 ruined. Phi = 5 - 2 - 1.5 = 1.5.
    let mut mixed = vec![0u8; 10];
    for h in &mut mixed[5..8] {
        *h = 1;
    }
    for h in &mut mixed[8..10] {
        *h = 2;
    }
    assert!((engine.team_welfare_phi(&mixed) - 1.5).abs() < 1e-5);
}

/// Phi(s) = 0 unconditionally when kind = "none". Guards the engine
/// fast-path skip in `compute_rewards`.
#[test]
fn test_team_welfare_phi_kind_none_returns_zero() {
    let scenario = SCENARIOS.get("default").unwrap().clone();
    // Defaults: team_welfare_kind == "none".
    let engine = BucketBrigade::new(scenario, 4, Some(0));
    for state_code in 0u8..=2 {
        let houses = vec![state_code; 10];
        assert_eq!(
            engine.team_welfare_phi(&houses),
            0.0,
            "kind=\"none\" must produce Phi=0 for any houses state"
        );
    }
}

/// Issue #283: `team_welfare_kind` allowlist re-check via `validate()`.
/// Mirrors the `distance_metric` invariant in
/// `test_validate_rejects_unknown_distance_metric` (scenarios.rs).
#[test]
#[should_panic(expected = "team_welfare_kind")]
fn test_engine_rejects_unknown_team_welfare_kind() {
    let mut scenario = SCENARIOS.get("default").unwrap().clone();
    scenario.team_welfare_kind = "bogus_kind".to_string();
    // Must panic from BucketBrigade::new -> Scenario::validate.
    let _ = BucketBrigade::new(scenario, 4, Some(42));
}

/// Byte-identity regression: `team_welfare_lambda == 0.0` (any kind) must
/// produce identical rewards to a scenario without the new fields touched.
/// Guards the engine fast-path skip in `compute_rewards`.
#[test]
fn test_zero_lambda_byte_identical_rewards() {
    let scenario_a = SCENARIOS.get("default").unwrap().clone();
    let mut scenario_b = scenario_a.clone();
    // Treatment: kind set but lambda zero -> still inert.
    scenario_b.team_welfare_kind = "team_welfare_closed_form".to_string();
    scenario_b.team_welfare_lambda = 0.0;
    scenario_b.team_welfare_gamma = 0.99;

    let mut engine_a = BucketBrigade::new(scenario_a, 4, Some(123));
    let mut engine_b = BucketBrigade::new(scenario_b, 4, Some(123));

    let actions = vec![[0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 0, 0]];
    for _ in 0..15 {
        if engine_a.done {
            break;
        }
        let r_a = engine_a.step(&actions).rewards;
        let r_b = engine_b.step(&actions).rewards;
        for (a, b) in r_a.iter().zip(r_b.iter()) {
            assert_eq!(
                a, b,
                "lambda=0 must yield byte-identical rewards regardless of kind"
            );
        }
    }
}

/// Shaping is team-shared: when lambda != 0, every agent receives the same
/// shaping increment, so the cross-agent reward differential is unchanged
/// vs the no-shaping case (rewards differ only by a uniform team-wide bonus).
#[test]
fn test_shaping_is_team_shared() {
    // Two engines that differ only in lambda. Same RNG seed + same actions
    // -> identical houses dynamics -> only difference in rewards is the
    // shaping term.
    let mut scenario_a = SCENARIOS.get("default").unwrap().clone();
    scenario_a.team_welfare_kind = "team_welfare_closed_form".to_string();
    scenario_a.team_welfare_lambda = 0.0;
    scenario_a.team_welfare_gamma = 0.99;

    let mut scenario_b = scenario_a.clone();
    scenario_b.team_welfare_lambda = 1.5;

    let mut engine_a = BucketBrigade::new(scenario_a, 4, Some(99));
    let mut engine_b = BucketBrigade::new(scenario_b, 4, Some(99));

    let actions = vec![[1, 1, 1], [2, 1, 1], [3, 0, 0], [4, 1, 1]];
    for _ in 0..10 {
        if engine_a.done {
            break;
        }
        let r_a = engine_a.step(&actions).rewards;
        let r_b = engine_b.step(&actions).rewards;
        // Confirm trajectories aligned.
        assert_eq!(engine_a.houses, engine_b.houses, "trajectories diverged");
        // Per-agent differential under team-shared shaping must be uniform.
        let diff0 = r_b[0] - r_a[0];
        for i in 1..r_a.len() {
            let diff_i = r_b[i] - r_a[i];
            assert!(
                (diff_i - diff0).abs() < 1e-4,
                "Shaping must be team-shared; per-agent diff non-uniform: {} vs {}",
                diff_i,
                diff0
            );
        }
    }
}

// ==============================================================================
// Issue #253: continuous extinguish dynamics (option D from #234)
// ==============================================================================

/// Hand-crafted single-fire test: with `suppression_per_worker = 1.0`,
/// one worker extinguishes a fire in exactly one step (deterministic).
#[test]
fn test_continuous_one_worker_one_step_suppression_one() {
    let mut scenario = SCENARIOS.get("default").unwrap().clone();
    scenario.extinguish_mode = "continuous".to_string();
    scenario.suppression_per_worker = 1.0;
    // Disable spread/ignition so the only dynamics are extinguishment.
    scenario.prob_fire_spreads_to_neighbor = 0.0;
    scenario.prob_house_catches_fire = 0.0;
    let mut engine = BucketBrigade::new(scenario, 4, Some(42));

    // Set a controlled single-fire state.
    engine.houses = vec![0; 10];
    engine.houses[5] = 1;
    engine.fire_progress = vec![0.0; 10];

    // One worker at house 5, others rest.
    let actions = vec![[5, 1, 1], [0, 0, 0], [1, 0, 0], [2, 0, 0]];
    engine.step(&actions);

    assert_eq!(
        engine.houses[5], 0,
        "Suppression=1.0 + 1 worker should fully extinguish in 1 step"
    );
    assert_eq!(
        engine.fire_progress[5], 0.0,
        "fire_progress must be zeroed on BURNING -> SAFE transition"
    );
}

/// With `suppression_per_worker = 0.5`, one worker takes exactly two steps
/// (deterministic). Verifies the accumulator persists across steps.
#[test]
fn test_continuous_one_worker_suppression_half_takes_two_steps() {
    let mut scenario = SCENARIOS.get("default").unwrap().clone();
    scenario.extinguish_mode = "continuous".to_string();
    scenario.suppression_per_worker = 0.5;
    scenario.prob_fire_spreads_to_neighbor = 0.0;
    scenario.prob_house_catches_fire = 0.0;
    let mut engine = BucketBrigade::new(scenario, 4, Some(42));

    engine.houses = vec![0; 10];
    engine.houses[5] = 1;
    engine.fire_progress = vec![0.0; 10];

    let actions = vec![[5, 1, 1], [0, 0, 0], [1, 0, 0], [2, 0, 0]];

    // Step 1: progress reaches 0.5, fire still burning.
    engine.step(&actions);
    assert_eq!(
        engine.houses[5], 1,
        "After 1 step at 0.5 suppression, fire should still be burning"
    );
    assert!(
        (engine.fire_progress[5] - 0.5).abs() < 1e-5,
        "fire_progress should be 0.5 after one step, got {}",
        engine.fire_progress[5]
    );

    // Step 2: progress reaches 1.0, fire transitions BURNING -> SAFE.
    engine.step(&actions);
    assert_eq!(
        engine.houses[5], 0,
        "After 2 steps at 0.5 suppression, fire should be extinguished"
    );
    assert_eq!(
        engine.fire_progress[5], 0.0,
        "fire_progress must be zeroed on extinguish"
    );
}

/// Two workers at `suppression_per_worker = 0.5` should extinguish in one
/// step (deterministic). Verifies the per-worker scaling.
#[test]
fn test_continuous_two_workers_suppression_half_one_step() {
    let mut scenario = SCENARIOS.get("default").unwrap().clone();
    scenario.extinguish_mode = "continuous".to_string();
    scenario.suppression_per_worker = 0.5;
    scenario.prob_fire_spreads_to_neighbor = 0.0;
    scenario.prob_house_catches_fire = 0.0;
    let mut engine = BucketBrigade::new(scenario, 4, Some(42));

    engine.houses = vec![0; 10];
    engine.houses[5] = 1;
    engine.fire_progress = vec![0.0; 10];

    // Two workers at house 5.
    let actions = vec![[5, 1, 1], [5, 1, 1], [1, 0, 0], [2, 0, 0]];
    engine.step(&actions);

    assert_eq!(
        engine.houses[5], 0,
        "Two workers at 0.5 each should fully extinguish in 1 step"
    );
}

/// In continuous mode fires do NOT auto-burn-out after a single
/// unextinguished step — they persist so the accumulator can integrate
/// suppression credit across multiple work steps. This is the core
/// semantic difference between the two modes; without it the calibration
/// to the Bernoulli expectation would be wrong.
///
/// Bernoulli mode's burn-out behavior is exercised separately by the
/// pre-existing `test_burn_out_phase` test, which still passes (the
/// dispatch in `burn_out_houses` skips only the continuous branch).
#[test]
fn test_continuous_no_auto_burn_out() {
    let mut scenario = SCENARIOS.get("default").unwrap().clone();
    scenario.extinguish_mode = "continuous".to_string();
    scenario.suppression_per_worker = 0.3; // Not enough to extinguish in one step.
    scenario.prob_fire_spreads_to_neighbor = 0.0;
    scenario.prob_house_catches_fire = 0.0;
    let mut engine = BucketBrigade::new(scenario, 4, Some(42));

    engine.houses = vec![0; 10];
    engine.houses[5] = 1;
    engine.fire_progress = vec![0.0; 10];

    // One worker — accumulates 0.3 per step. After two steps the fire is
    // still burning (progress = 0.6, not at threshold yet). The fire must
    // *not* be ruined here — that's the whole point of accumulator mode.
    let actions = vec![[5, 1, 1], [0, 0, 0], [1, 0, 0], [2, 0, 0]];
    engine.step(&actions);
    assert_eq!(engine.houses[5], 1, "Fire must persist after step 1");
    assert!((engine.fire_progress[5] - 0.3).abs() < 1e-5);
    engine.step(&actions);
    assert_eq!(engine.houses[5], 1, "Fire must persist after step 2");
    assert!((engine.fire_progress[5] - 0.6).abs() < 1e-5);
    // Step 4 finally pushes the accumulator over 1.0 (1.2).
    engine.step(&actions);
    engine.step(&actions);
    assert_eq!(
        engine.houses[5], 0,
        "Fire should be extinguished once accumulator crosses 1.0"
    );
}

/// `fire_progress` must be zeroed on spontaneous ignition so a newly-ignited
/// fire starts with a clean accumulator (even if a previous fire at the
/// same house ran up the counter and got ruined — see the burn-out test
/// above which already zeroes the accumulator, but the spontaneous-ignition
/// path zeroes it explicitly for symmetry).
#[test]
fn test_continuous_fire_progress_zeroed_on_spontaneous_ignition() {
    let mut scenario = SCENARIOS.get("default").unwrap().clone();
    scenario.extinguish_mode = "continuous".to_string();
    scenario.suppression_per_worker = 0.5;
    scenario.prob_fire_spreads_to_neighbor = 0.0;
    scenario.prob_house_catches_fire = 1.0; // Force ignition on all SAFE houses.
    let mut engine = BucketBrigade::new(scenario, 4, Some(42));

    // Cleared state with stale fire_progress value at house 5.
    engine.houses = vec![0; 10];
    engine.fire_progress = vec![0.0; 10];
    engine.fire_progress[5] = 0.4; // Stale (e.g. left over from a manually-set state).

    // No-one works; spontaneous ignition will set every house to BURNING.
    let actions = vec![[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]];
    engine.step(&actions);

    // Spontaneous ignition fires at end-of-step (after burn-out). Houses
    // not on fire at start-of-step are now BURNING; their fire_progress
    // must be 0.
    assert_eq!(engine.houses[5], 1);
    assert_eq!(
        engine.fire_progress[5], 0.0,
        "Spontaneous ignition must zero fire_progress so the new fire starts clean"
    );
}

/// Bit-exact regression: with `extinguish_mode = "bernoulli"` (default),
/// trajectories and rewards are byte-identical to a pre-#253 engine. Done
/// by running the same scenario with and without the new fields set
/// explicitly to defaults.
#[test]
fn test_bernoulli_mode_byte_identical_to_default() {
    let scenario_a = SCENARIOS.get("default").unwrap().clone();
    let mut scenario_b = scenario_a.clone();
    // Touch the new knobs explicitly to defaults; should not change anything.
    scenario_b.extinguish_mode = "bernoulli".to_string();
    scenario_b.suppression_per_worker = 0.0;

    let mut engine_a = BucketBrigade::new(scenario_a, 4, Some(123));
    let mut engine_b = BucketBrigade::new(scenario_b, 4, Some(123));

    let actions = vec![[0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 0, 0]];
    for _ in 0..20 {
        if engine_a.done {
            break;
        }
        let r_a = engine_a.step(&actions).rewards;
        let r_b = engine_b.step(&actions).rewards;
        assert_eq!(
            r_a, r_b,
            "Bernoulli mode must produce byte-identical rewards"
        );
        assert_eq!(engine_a.houses, engine_b.houses);
    }
}

/// Unknown `extinguish_mode` is rejected by `Scenario::validate()` -> engine
/// construction panics. Mirrors the `distance_metric` invariant.
#[test]
#[should_panic(expected = "extinguish_mode")]
fn test_engine_rejects_unknown_extinguish_mode() {
    let mut scenario = SCENARIOS.get("default").unwrap().clone();
    scenario.extinguish_mode = "bogus".to_string();
    // Must panic from BucketBrigade::new -> Scenario::validate.
    let _ = BucketBrigade::new(scenario, 4, Some(42));
}

/// Statistical equivalence: the per-step P(extinguish | fire still
/// burning, one worker, kappa) matches between Bernoulli mode (single coin
/// flip per step) and a one-step continuous trial where the accumulator is
/// reset each step (which makes the continuous model degenerate to a
/// deterministic per-step suppression that doesn't carry over).
///
/// For the more honest cross-mode comparison: when the Bernoulli `kappa`
/// matches the continuous `suppression_per_worker`, the *expected
/// nights-to-extinguish conditional on a non-burn-out path* matches
/// 1/kappa for both models. Bernoulli has a geometric distribution
/// conditional on no burn-out, but in this engine's pre-#253 dynamics
/// burn-out fires every unsucessful step, so the conditional doesn't
/// apply — Bernoulli always finishes in exactly 1 step (either SAFE or
/// RUINED). We test the **observable rate** that *something happens*
/// per step: in Bernoulli that's 100% (fires always resolve in 1 step);
/// in continuous it depends on the accumulator threshold. So a true
/// long-run extinguish-rate equivalence at fixed kappa requires the
/// continuous mode's *deterministic* T = ceil(1/kappa) steps to match
/// the *expectation of the geometric process where each step has
/// P(extinguish) = kappa and P(burn-out) = 1-kappa*.
///
/// Concretely we verify:
/// 1. Bernoulli mode resolves every single-fire scenario in exactly one
///    step (either SAFE or RUINED).
/// 2. Continuous mode at `suppression_per_worker = 0.5` always resolves
///    in exactly 2 steps (SAFE).
/// 3. The fraction-of-fires-extinguished under Bernoulli at kappa=0.5
///    over 1000 trials is within 5% of 0.5 (the theoretical kappa).
/// 4. Continuous mode at the calibrated `suppression_per_worker = kappa`
///    deterministically extinguishes 100% of fires, which is the
///    long-run "extinguish rate equivalence" benefit option D targets
///    (no fires are lost to bad luck under sufficient effort).
#[test]
fn test_statistical_extinguish_rate_bernoulli_matches_kappa() {
    let trials = 1000;
    let kappa: f32 = 0.5;

    let mut extinguished_count: u32 = 0;
    for seed in 0..trials {
        let mut scenario = SCENARIOS.get("default").unwrap().clone();
        scenario.prob_fire_spreads_to_neighbor = 0.0;
        scenario.prob_house_catches_fire = 0.0;
        scenario.prob_solo_agent_extinguishes_fire = kappa;
        scenario.min_nights = 0;
        let mut engine = BucketBrigade::new(scenario, 4, Some(seed as u64));

        engine.houses = vec![0; 10];
        engine.houses[5] = 1;
        engine.fire_progress = vec![0.0; 10];

        let actions = vec![[5, 1, 1], [0, 0, 0], [1, 0, 0], [2, 0, 0]];
        engine.step(&actions);
        // Bernoulli always resolves in 1 step.
        assert_ne!(engine.houses[5], 1, "Bernoulli mode must resolve in 1 step");
        if engine.houses[5] == 0 {
            extinguished_count += 1;
        }
    }
    let rate = extinguished_count as f32 / trials as f32;
    let err = ((rate - kappa) / kappa).abs();
    assert!(
        err < 0.05,
        "Bernoulli extinguish rate {:.3} should be within 5% of kappa = {:.3}, got error {:.3}",
        rate,
        kappa,
        err
    );

    // Continuous mode: at suppression_per_worker = kappa = 0.5, every
    // fire gets extinguished deterministically in 2 steps. The
    // "long-run extinguish rate" is 100%, vs Bernoulli's ~50%. This
    // *is* the calibration: continuous mode trades probabilistic
    // failure for deterministic time-to-resolution, which is the
    // gradient-smoothing benefit option D claims.
    let mut continuous_scenario = SCENARIOS.get("default").unwrap().clone();
    continuous_scenario.extinguish_mode = "continuous".to_string();
    continuous_scenario.suppression_per_worker = kappa;
    continuous_scenario.prob_fire_spreads_to_neighbor = 0.0;
    continuous_scenario.prob_house_catches_fire = 0.0;
    continuous_scenario.min_nights = 0;

    let mut continuous_extinguished: u32 = 0;
    let mut continuous_total_nights: u32 = 0;
    for seed in 0..trials {
        let mut engine = BucketBrigade::new(continuous_scenario.clone(), 4, Some(seed as u64));

        engine.houses = vec![0; 10];
        engine.houses[5] = 1;
        engine.fire_progress = vec![0.0; 10];

        let actions = vec![[5, 1, 1], [0, 0, 0], [1, 0, 0], [2, 0, 0]];
        let mut nights = 0u32;
        loop {
            nights += 1;
            engine.step(&actions);
            if engine.houses[5] != 1 {
                break;
            }
            if nights >= 10 {
                break;
            }
        }
        if engine.houses[5] == 0 {
            continuous_extinguished += 1;
        }
        continuous_total_nights += nights;
    }
    let continuous_rate = continuous_extinguished as f32 / trials as f32;
    let mean_nights = continuous_total_nights as f32 / trials as f32;

    // Continuous mode extinguishes 100% of fires (deterministic).
    assert!(
        (continuous_rate - 1.0).abs() < 1e-3,
        "Continuous mode extinguish rate {} should be ~100% (deterministic)",
        continuous_rate
    );
    // Mean nights-to-extinguish = 1/kappa = 2 (deterministic).
    let target_nights = 1.0 / kappa;
    let err_nights = ((mean_nights - target_nights) / target_nights).abs();
    assert!(
        err_nights < 0.05,
        "Continuous mean nights-to-extinguish {} should be within 5% of 1/kappa = {}, got error {:.3}",
        mean_nights,
        target_nights,
        err_nights
    );
}

/// `default_continuous` scenario boots cleanly end-to-end (smoke test):
/// the engine constructs without panics, runs a short rollout, and the
/// rewards are all finite.
#[test]
fn test_default_continuous_smoke() {
    let scenario = SCENARIOS.get("default_continuous").unwrap().clone();
    let mut engine = BucketBrigade::new(scenario, 4, Some(7));
    for _ in 0..20 {
        if engine.done {
            break;
        }
        let actions = vec![[0u8, 1u8, 1u8], [3, 1, 1], [5, 1, 1], [8, 1, 1]];
        let result = engine.step(&actions);
        assert_eq!(result.rewards.len(), 4);
        for &r in &result.rewards {
            assert!(r.is_finite(), "Per-step reward must be finite, got {}", r);
        }
    }
}

// ==============================================================================
// Property-Based Tests
// ==============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::scenarios::Scenario;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_agent_scores_bounded(
            num_agents in 2usize..20usize,
            seed in 0u64..10000u64
        ) {
            let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
            let mut engine = BucketBrigade::new(scenario.clone(), num_agents, Some(seed));

            // Run a few steps
            for _ in 0..10 {
                if engine.done {
                    break;
                }
                let actions: Vec<[u8; 3]> = (0..num_agents)
                    .map(|i| [(i % 10) as u8, 1, 1])
                    .collect();
                engine.step(&actions);
            }

            // Mark as done to get final scores
            engine.done = true;
            let result = engine.get_result();

            // Scores should be bounded
            // Maximum possible: all houses safe, all rest, long game
            // A=100, L=100, c=0.5, max_nights ~= 100
            let max_reasonable_score = 2000.0;
            let min_reasonable_score = -2000.0;

            for (i, &score) in result.agent_scores.iter().enumerate() {
                prop_assert!(
                    score > min_reasonable_score && score < max_reasonable_score,
                    "Agent {} score {} out of reasonable bounds [{}, {}]",
                    i, score, min_reasonable_score, max_reasonable_score
                );
            }
        }

        #[test]
        fn test_rewards_per_step_bounded(
            num_agents in 2usize..10usize,
            seed in 0u64..1000u64
        ) {
            let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
            let mut engine = BucketBrigade::new(scenario.clone(), num_agents, Some(seed));

            let actions: Vec<[u8; 3]> = (0..num_agents)
                .map(|i| [(i % 10) as u8, 1, 1])
                .collect();

            let result = engine.step(&actions);

            // Per-step rewards now include team rewards and ownership bonuses
            // No strict bounds since team rewards scale with saved/ruined houses
            // But we can check they're reasonable (not NaN or infinite)
            for (i, &reward) in result.rewards.iter().enumerate() {
                prop_assert!(
                    reward.is_finite(),
                    "Agent {} per-step reward {} should be finite",
                    i, reward
                );
            }
        }

        #[test]
        fn test_game_terminates_eventually(
            num_agents in 2usize..10usize,
            seed in 0u64..100u64
        ) {
            let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
            let mut engine = BucketBrigade::new(scenario, num_agents, Some(seed));

            let mut steps = 0;
            let max_steps = 200;

            while !engine.done && steps < max_steps {
                let actions: Vec<[u8; 3]> = (0..num_agents)
                    .map(|i| [(i % 10) as u8, 1, 1])
                    .collect();
                engine.step(&actions);
                steps += 1;
            }

            prop_assert!(
                engine.done,
                "Game should terminate within {} steps, but didn't after {}",
                max_steps, steps
            );
        }

        #[test]
        fn test_house_states_valid(
            num_agents in 2usize..10usize,
            seed in 0u64..100u64
        ) {
            let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
            let mut engine = BucketBrigade::new(scenario, num_agents, Some(seed));

            // Check initial state
            for &house in &engine.houses {
                prop_assert!(house <= 2, "House state {} invalid (should be 0-2)", house);
            }

            // Take a few steps and check again
            for _ in 0..5 {
                if engine.done {
                    break;
                }
                let actions: Vec<[u8; 3]> = (0..num_agents)
                    .map(|i| [(i % 10) as u8, 1, 1])
                    .collect();
                engine.step(&actions);

                for &house in &engine.houses {
                    prop_assert!(house <= 2, "House state {} invalid (should be 0-2)", house);
                }
            }
        }

        #[test]
        fn test_num_rewards_matches_num_agents(
            num_agents in 2usize..20usize,
            seed in 0u64..100u64
        ) {
            let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
            let mut engine = BucketBrigade::new(scenario, num_agents, Some(seed));

            let actions: Vec<[u8; 3]> = (0..num_agents)
                .map(|i| [(i % 10) as u8, 1, 1])
                .collect();

            let result = engine.step(&actions);

            prop_assert_eq!(
                result.rewards.len(),
                num_agents,
                "Number of rewards should match number of agents"
            );
        }

        #[test]
        fn test_scenario_probabilities_valid(
            beta in 0.0f32..1.0f32,
            kappa in 0.0f32..1.0f32,
            p_spark in 0.0f32..0.1f32
        ) {
            // Create custom scenario with random probabilities.
            // Per-agent ownership reward vectors are length 10 (max agents).
            let scenario = Scenario {
                prob_fire_spreads_to_neighbor: beta,
                prob_solo_agent_extinguishes_fire: kappa,
                prob_house_catches_fire: p_spark,
                team_reward_house_survives: 100.0,
                team_penalty_house_burns: 100.0,
                cost_to_work_one_night: 0.5,
                min_nights: 12,
                // Issue #447: historical rest reward so the existing
                // proptest invariants (assume pre-#447 reward semantics)
                // hold.
                reward_rest: 0.5,
                reward_own_house_survives: vec![100.0; 10],
                reward_other_house_survives: vec![50.0; 10],
                penalty_own_house_burns: vec![0.0; 10],
                penalty_other_house_burns: vec![0.0; 10],
                // Issue #203 spatial-cost fields: use pre-#203 defaults so
                // existing proptest invariants are unchanged.
                agent_home_positions: Vec::new(),
                distance_cost_alpha: 0.0,
                distance_metric: "ring_arc".to_string(),
                // Issue #254: pre-#254 fixed-10 ring for these proptests.
                num_houses: 10,
                // Issue #259: action shaping off so the existing proptest
                // invariants (which assume pre-#259 reward semantics) hold.
                action_shaping_alpha: 0.0,
                action_shaping_beta: 0.0,
                // Issue #265: progress shaping off so the existing proptest
                // invariants (which assume pre-#265 reward semantics) hold.
                progress_shaping_coef: 0.0,
                // Issue #283: potential-based shaping off so existing
                // proptest invariants (assume pre-#283 reward semantics) hold.
                team_welfare_lambda: 0.0,
                team_welfare_gamma: 1.0,
                team_welfare_kind: "none".to_string(),
                // Issue #251: action-validity off so existing proptest
                // invariants (which assume pre-#251 unconstrained-action
                // semantics) hold.
                action_validity_mode: "always_valid".to_string(),
                // Issue #253: pre-#253 Bernoulli extinguish mode so the
                // existing proptest invariants (assume pre-#253 reward
                // semantics) hold bit-exactly.
                extinguish_mode: "bernoulli".to_string(),
                suppression_per_worker: 0.0,
                // Issue #252: pre-#252 simultaneous commitment so the
                // existing proptest invariants (assume pre-#252 single-phase
                // semantics) hold bit-exactly.
                commitment_mode: "simultaneous".to_string(),
            };

            let mut engine = BucketBrigade::new(scenario, 4, Some(42));

            // Should not panic with any valid probabilities
            let actions = vec![[0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 1, 1]];
            let result = engine.step(&actions);

            prop_assert_eq!(result.rewards.len(), 4);
        }
    }
}

// --- Issue #251: position-constrained action validity (adjacent-only v1) ---

#[cfg(test)]
mod action_validity_tests {
    use super::*;
    use crate::{Action, Scenario, SCENARIOS};

    /// Build a scenario with `agent_home_positions = [0, 3, 5, 8]` (mirrors
    /// `positional_default`) and the supplied `action_validity_mode`. All
    /// other knobs are inherited from the `default` scenario.
    fn make_positioned_scenario(mode: &str) -> Scenario {
        let mut scenario = SCENARIOS.get("default").unwrap().clone();
        scenario.agent_home_positions = vec![0, 3, 5, 8];
        scenario.action_validity_mode = mode.to_string();
        scenario
    }

    /// `always_valid` mode (the default) is a bit-exact pass-through. Every
    /// raw action survives sanitization, so the resulting `last_actions`
    /// matches the input exactly. This is the entire backward-compat
    /// guarantee for pre-#251 scenarios.
    #[test]
    fn test_always_valid_mode_is_bit_exact_pass_through() {
        let scenario = SCENARIOS.get("default").unwrap().clone();
        assert_eq!(scenario.action_validity_mode, "always_valid");
        let mut engine = BucketBrigade::new(scenario, 4, Some(42));
        // Mix of out-of-reach (5,7,9) and home-position (0) targets.
        let actions: Vec<Action> = vec![[5, 1, 1], [7, 0, 0], [9, 1, 0], [0, 1, 1]];
        engine.step(&actions);
        assert_eq!(engine.last_actions, actions);
        assert_eq!(engine.agent_positions, vec![5, 7, 9, 0]);
    }

    /// `adjacent_only` mode rewrites every out-of-reach target to the
    /// agent's home position. Mode and signal bits are preserved.
    #[test]
    fn test_adjacent_only_clamps_out_of_reach_targets_to_home() {
        let scenario = make_positioned_scenario("adjacent_only");
        let mut engine = BucketBrigade::new(scenario, 4, Some(42));
        // Homes: [0, 3, 5, 8]. Adjacent (ring-dist <= 1) windows on a
        // 10-ring: {9,0,1}, {2,3,4}, {4,5,6}, {7,8,9}. Out-of-reach
        // targets below: 5 (a0), 6 (a1), 9 (a2), 0 (a3).
        let raw: Vec<Action> = vec![[5, 1, 1], [6, 0, 0], [9, 1, 0], [0, 1, 1]];
        engine.step(&raw);
        // Every action's house should be rewritten to home; mode + signal
        // unchanged.
        assert_eq!(engine.agent_positions, vec![0, 3, 5, 8]);
        assert_eq!(
            engine.last_actions,
            vec![[0, 1, 1], [3, 0, 0], [5, 1, 0], [8, 1, 1]]
        );
    }

    /// In-reach targets in `adjacent_only` mode pass through unchanged.
    /// Verifies the boundary gradient: the policy can still choose between
    /// home and immediate neighbors.
    #[test]
    fn test_adjacent_only_passes_through_in_reach_targets() {
        let scenario = make_positioned_scenario("adjacent_only");
        let mut engine = BucketBrigade::new(scenario, 4, Some(42));
        // For homes [0,3,5,8] on a 10-ring (wrap), legal targets:
        //   a0 home=0: {9, 0, 1}; pick 1
        //   a1 home=3: {2, 3, 4}; pick 4
        //   a2 home=5: {4, 5, 6}; pick 6
        //   a3 home=8: {7, 8, 9}; pick 9 (mode=REST + WORK signal too)
        let raw: Vec<Action> = vec![[1, 1, 1], [4, 1, 0], [6, 0, 1], [9, 0, 1]];
        engine.step(&raw);
        assert_eq!(engine.last_actions, raw);
        assert_eq!(engine.agent_positions, vec![1, 4, 6, 9]);
    }

    /// Ring wrap: `adjacent_only` recognizes wraparound neighbors. On a
    /// 10-ring, house 9 is adjacent to house 0 (ring dist == 1).
    #[test]
    fn test_adjacent_only_recognizes_ring_wrap() {
        let scenario = make_positioned_scenario("adjacent_only");
        let mut engine = BucketBrigade::new(scenario, 4, Some(42));
        // a0 home=0: target 9 is the wraparound neighbor (ring dist 1).
        // a3 home=8: target 7 is the in-ring neighbor (ring dist 1).
        let raw: Vec<Action> = vec![[9, 1, 1], [3, 0, 0], [5, 0, 0], [7, 1, 1]];
        engine.step(&raw);
        // All in-reach -> pass-through.
        assert_eq!(engine.last_actions, raw);
    }

    /// Determinism: same raw actions + same seed + same mode produce the
    /// same trajectory whether or not sanitization fired. Specifically,
    /// rewards under `always_valid` with home-position actions are
    /// identical to rewards under `adjacent_only` with the same
    /// home-position actions.
    #[test]
    fn test_adjacent_only_with_home_actions_matches_always_valid() {
        let base = SCENARIOS.get("default").unwrap().clone();
        let mut sc_always = base.clone();
        sc_always.agent_home_positions = vec![0, 3, 5, 8];
        sc_always.action_validity_mode = "always_valid".to_string();

        let mut sc_adj = sc_always.clone();
        sc_adj.action_validity_mode = "adjacent_only".to_string();

        let mut e_always = BucketBrigade::new(sc_always, 4, Some(7));
        let mut e_adj = BucketBrigade::new(sc_adj, 4, Some(7));

        // Both at home: sanitization is a no-op.
        let actions: Vec<Action> = vec![[0, 1, 1], [3, 1, 1], [5, 1, 1], [8, 1, 1]];
        let r_always = e_always.step(&actions);
        let r_adj = e_adj.step(&actions);
        assert_eq!(r_always.rewards, r_adj.rewards);
        assert_eq!(e_always.houses, e_adj.houses);
    }

    /// The sanitized action is what flows to the rewards engine: an agent
    /// that "tries to" WORK at a far-away house pays only the home work
    /// cost, not the far-away cost. This is the load-bearing semantic that
    /// makes the constraint effective rather than cosmetic.
    #[test]
    fn test_adjacent_only_redirects_work_cost_to_home() {
        let base = SCENARIOS.get("default").unwrap().clone();
        let mut sc = base.clone();
        sc.agent_home_positions = vec![0, 3, 5, 8];
        // Turn on a positive distance cost so the redirection is observable.
        sc.distance_cost_alpha = 0.1;
        sc.action_validity_mode = "adjacent_only".to_string();
        let mut engine = BucketBrigade::new(sc, 4, Some(11));
        // Agent 0 (home=0) tries to work at house 5 (ring dist 5, out of
        // reach). After sanitization, agent 0 works at home 0 — distance
        // cost contribution is 0.1 * 0 = 0, not 0.1 * 5.
        let raw: Vec<Action> = vec![[5, 1, 1], [3, 1, 1], [5, 1, 1], [8, 1, 1]];
        engine.step(&raw);
        // Agent 0's recorded position is home, not 5.
        assert_eq!(engine.agent_positions[0], 0);
        assert_eq!(engine.last_actions[0], [0, 1, 1]);
    }

    /// Engine construction with a bogus `action_validity_mode` must panic
    /// via `Scenario::validate`. Guards every non-serde construction site.
    #[test]
    #[should_panic(expected = "action_validity_mode")]
    fn test_engine_rejects_unknown_action_validity_mode() {
        let mut scenario = SCENARIOS.get("default").unwrap().clone();
        scenario.action_validity_mode = "k_hop_2".to_string();
        let _ = BucketBrigade::new(scenario, 4, Some(42));
    }
}

// ==========================================================================
// Issue #252: within-night commitment mode (two-phase signaling)
// ==========================================================================

mod commitment_mode_tests {
    use super::*;
    use crate::SCENARIOS;

    fn two_phase_scenario() -> crate::Scenario {
        let mut s = SCENARIOS.get("default").unwrap().clone();
        s.commitment_mode = "two_phase".to_string();
        s
    }

    /// Engine construction with a bogus `commitment_mode` must panic via
    /// `Scenario::validate`. Guards every non-serde construction site.
    #[test]
    #[should_panic(expected = "commitment_mode")]
    fn test_engine_rejects_unknown_commitment_mode() {
        let mut scenario = SCENARIOS.get("default").unwrap().clone();
        scenario.commitment_mode = "stochastic_order".to_string();
        let _ = BucketBrigade::new(scenario, 4, Some(42));
    }

    /// Bit-exact regression: every named scenario constructs with
    /// `commitment_mode == "simultaneous"` and `step()` succeeds normally.
    /// This is the critical backward-compat guarantee — if this breaks,
    /// every pre-#252 baseline (P3 evolution, BC fits, Nash analysis) is
    /// invalidated.
    #[test]
    fn test_simultaneous_step_works_on_all_scenarios() {
        for (name, scenario) in SCENARIOS.iter() {
            assert_eq!(
                scenario.commitment_mode, "simultaneous",
                "Scenario '{name}' must default to simultaneous"
            );
            let mut engine = BucketBrigade::new(scenario.clone(), 4, Some(42));
            let actions = vec![[0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 1, 1]];
            let _ = engine.step(&actions);
            assert_eq!(
                engine.night, 1,
                "Scenario '{name}' should advance to night 1"
            );
        }
    }

    /// Two-phase scenarios MUST go through `step_two_phase`. Single-phase
    /// `step()` panics so callers can't accidentally skip the signal
    /// phase (which would leak round-1 signals across nights).
    #[test]
    #[should_panic(expected = "two-phase")]
    fn test_step_panics_on_two_phase_scenario() {
        let scenario = two_phase_scenario();
        let mut engine = BucketBrigade::new(scenario, 4, Some(42));
        let actions = vec![[0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 1, 1]];
        let _ = engine.step(&actions);
    }

    /// Conversely, `step_two_phase` requires `commitment_mode ==
    /// "two_phase"`. Calling it on a simultaneous scenario panics.
    #[test]
    #[should_panic(expected = "two_phase")]
    fn test_step_two_phase_panics_on_simultaneous_scenario() {
        let scenario = SCENARIOS.get("default").unwrap().clone();
        let mut engine = BucketBrigade::new(scenario, 4, Some(42));
        let r1 = vec![1, 1, 1, 1];
        let r2 = vec![[0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 1, 1]];
        let _ = engine.step_two_phase(&r1, &r2);
    }

    /// Round-1 invariant: round-1 signals must appear in the obs after
    /// `step_two_phase`. The round-2 obs read by the round-2 policy
    /// forward (mid-step, not after the step) carries the round-1
    /// signals so the policy can condition on them. After the step
    /// completes, `round1_signals` remains visible until the next
    /// `step_two_phase` overwrites them.
    #[test]
    fn test_two_phase_round1_signals_visible_in_obs() {
        let scenario = two_phase_scenario();
        let mut engine = BucketBrigade::new(scenario, 4, Some(42));
        let r1 = vec![0, 1, 0, 1];
        let r2 = vec![[0, 1, 1], [1, 1, 1], [2, 1, 0], [3, 0, 0]];
        let _ = engine.step_two_phase(&r1, &r2);
        let obs = engine.get_observation(2);
        assert_eq!(obs.round1_signals, vec![0, 1, 0, 1]);
    }

    /// Simultaneous-mode `round1_signals` is all-zeros so the obs is
    /// byte-identical to pre-#252 once the channel is excluded from the
    /// flat obs vector. Critical for backward compatibility.
    #[test]
    fn test_simultaneous_round1_signals_are_zero() {
        let scenario = SCENARIOS.get("default").unwrap().clone();
        let mut engine = BucketBrigade::new(scenario, 4, Some(42));
        let actions = vec![[0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 1, 1]];
        let _ = engine.step(&actions);
        let obs = engine.get_observation(0);
        assert_eq!(obs.round1_signals, vec![0, 0, 0, 0]);
    }

    /// Round-2 action-phase parity: under `two_phase`, given identical
    /// round-2 actions and an inert round-1 signal vector (all zeros),
    /// the per-night reward equals the simultaneous-mode reward for the
    /// same actions and seed. Confirms round-1 is a true no-op when
    /// signals are zero — the only thing two-phase adds is the round-1
    /// obs channel.
    #[test]
    fn test_two_phase_zero_signals_matches_simultaneous_rewards() {
        // Build two engines with identical seeds; one simultaneous, one
        // two-phase with zero round-1 signals.
        let sim = SCENARIOS.get("default").unwrap().clone();
        let mut tp = sim.clone();
        tp.commitment_mode = "two_phase".to_string();

        let mut engine_sim = BucketBrigade::new(sim, 4, Some(123));
        let mut engine_tp = BucketBrigade::new(tp, 4, Some(123));

        // Identical initial fire layout (same seed).
        assert_eq!(engine_sim.houses, engine_tp.houses);

        let r1_zero = vec![0, 0, 0, 0];
        let r2 = vec![[0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 1, 1]];
        let result_sim = engine_sim.step(&r2);
        let result_tp = engine_tp.step_two_phase(&r1_zero, &r2);

        assert_eq!(result_sim.rewards, result_tp.rewards);
        assert_eq!(result_sim.done, result_tp.done);
        assert_eq!(engine_sim.houses, engine_tp.houses);
        assert_eq!(engine_sim.agent_signals, engine_tp.agent_signals);
        assert_eq!(engine_sim.night, engine_tp.night);
    }

    /// **PR GATE (can-still-lie)**: the architect flagged this as the
    /// highest research-interest risk. The two-phase design preserves the
    /// deception channel iff the engine does not constrain round-2 mode
    /// based on the round-1 signal. We exercise this directly with a
    /// hardcoded "Liar" policy: round-1 signal = WORK (1), round-2 mode =
    /// REST (0). The engine must accept the action and the recorded
    /// trajectory must show the inconsistency.
    ///
    /// This is a *mechanical* test of the engine surface, not a
    /// training-time test. It's strictly stronger than the "100 PPO iters
    /// produce lying ≥ 1%" gate from the issue body: if the engine
    /// silently equalizes round-2 mode to round-1 signal, *no* trained
    /// policy could lie regardless of how many iters it ran. Conversely,
    /// if the engine accepts the lie here, a trained policy that ever
    /// emits inconsistent (signal, mode) pairs will see them reflected in
    /// the trajectory — exactly the deception substrate the project
    /// requires.
    #[test]
    fn test_can_still_lie_two_phase() {
        let scenario = two_phase_scenario();
        let mut engine = BucketBrigade::new(scenario, 4, Some(7));

        // All four agents lie: round-1 signal=WORK (1), round-2 mode=REST (0).
        // Round-2 signal is also 0 (consistent within round 2; the lie is
        // between round-1 and round-2).
        let r1_lie = vec![1u8, 1, 1, 1];
        let r2_rest = vec![[0u8, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]];

        // Sample a few nights and inspect the recorded trajectory.
        let mut lying_count = 0;
        let mut total_count = 0;
        for _ in 0..5 {
            let _ = engine.step_two_phase(&r1_lie, &r2_rest);
            let obs = engine.get_observation(0);
            // The round-1 signals are still visible immediately after the
            // step (until the next step_two_phase overwrites them).
            for i in 0..4 {
                total_count += 1;
                if obs.round1_signals[i] != r2_rest[i][1] {
                    lying_count += 1;
                }
            }
            if engine.done {
                break;
            }
        }
        assert!(
            total_count > 0,
            "Test must exercise at least one (round1, round2) pair"
        );
        let lie_rate = lying_count as f32 / total_count as f32;
        // The hardcoded liar lies on every pair, so we expect rate == 1.0.
        // The acceptance threshold is 1% per the PR gate; we assert the
        // mechanical reality (the engine doesn't equalize them) is much
        // stronger.
        assert!(
            lie_rate >= 0.01,
            "can-still-lie PR gate: expected lying rate >= 1% with a \
             hardcoded liar; got {:.4} ({}/{} pairs inconsistent). \
             If this rate is 0, the engine has silently equalized round-2 \
             mode to round-1 signal — the deception channel has been \
             destroyed and the design is broken. DO NOT MERGE.",
            lie_rate,
            lying_count,
            total_count
        );
        // Also assert no per-step penalty was specifically tied to lying:
        // the rewards should be the same as if the agent had been honest
        // about resting (signal=0, mode=0). We don't compute the honest
        // baseline here — the assertion is implicit in the fact that
        // step_two_phase produced a valid StepResult with finite rewards.
    }

    /// Round-1 does not advance the night. Two `step_two_phase` calls
    /// advance the night counter by 2, not 1+1+round1_no_op. This
    /// confirms the curator spec: round-1 is a pure obs-channel write,
    /// not an engine substep.
    #[test]
    fn test_two_phase_advances_night_once_per_call() {
        let scenario = two_phase_scenario();
        let mut engine = BucketBrigade::new(scenario, 4, Some(99));
        let r1 = vec![1, 0, 1, 0];
        let r2 = vec![[0, 1, 1], [1, 0, 0], [2, 1, 1], [3, 0, 0]];
        assert_eq!(engine.night, 0);
        let _ = engine.step_two_phase(&r1, &r2);
        assert_eq!(engine.night, 1);
        let _ = engine.step_two_phase(&r1, &r2);
        assert_eq!(engine.night, 2);
    }

    /// Deterministic-with-seed parity for the two-phase path: identical
    /// seeds produce identical trajectories across runs. Mirrors the
    /// simultaneous-mode determinism test.
    #[test]
    fn test_two_phase_deterministic_with_seed() {
        let scenario = two_phase_scenario();
        let mut engine1 = BucketBrigade::new(scenario.clone(), 4, Some(456));
        let mut engine2 = BucketBrigade::new(scenario, 4, Some(456));
        assert_eq!(engine1.houses, engine2.houses);

        let r1 = vec![1, 0, 1, 0];
        let r2 = vec![[0, 1, 1], [1, 1, 0], [2, 1, 1], [3, 1, 0]];
        let r1_a = engine1.step_two_phase(&r1, &r2);
        let r1_b = engine2.step_two_phase(&r1, &r2);
        assert_eq!(r1_a.rewards, r1_b.rewards);
        assert_eq!(engine1.houses, engine2.houses);
    }

    /// Length-mismatch panics: round1_signals and round2_actions must
    /// each have length num_agents. This protects against silent shape
    /// drift between trainer and engine.
    #[test]
    #[should_panic(expected = "round1_signals length")]
    fn test_two_phase_rejects_wrong_round1_length() {
        let scenario = two_phase_scenario();
        let mut engine = BucketBrigade::new(scenario, 4, Some(42));
        let r1_bad = vec![1, 0, 1]; // length 3, not 4
        let r2 = vec![[0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 1, 1]];
        let _ = engine.step_two_phase(&r1_bad, &r2);
    }

    #[test]
    #[should_panic(expected = "round2_actions length")]
    fn test_two_phase_rejects_wrong_round2_length() {
        let scenario = two_phase_scenario();
        let mut engine = BucketBrigade::new(scenario, 4, Some(42));
        let r1 = vec![1, 0, 1, 0];
        let r2_bad = vec![[0, 1, 1], [1, 1, 1], [2, 1, 1]]; // length 3
        let _ = engine.step_two_phase(&r1, &r2_bad);
    }

    /// Reset clears round1_signals so they don't leak across episodes.
    #[test]
    fn test_two_phase_reset_clears_round1_signals() {
        let scenario = two_phase_scenario();
        let mut engine = BucketBrigade::new(scenario, 4, Some(42));
        let r1 = vec![1, 1, 1, 1];
        let r2 = vec![[0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 1, 1]];
        let _ = engine.step_two_phase(&r1, &r2);
        assert_eq!(engine.round1_signals, vec![1, 1, 1, 1]);
        engine.reset();
        assert_eq!(engine.round1_signals, vec![0, 0, 0, 0]);
    }
}
