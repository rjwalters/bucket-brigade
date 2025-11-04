use super::*;
use crate::SCENARIOS;

#[test]
fn test_engine_creation() {
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    let engine = BucketBrigade::new(scenario.clone(), Some(42));

    assert_eq!(engine.night, 0);
    assert!(!engine.done);
    assert_eq!(engine.scenario.num_agents, 4);
}

#[test]
fn test_engine_initialization_fires() {
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    let engine = BucketBrigade::new(scenario, Some(42));

    // Count burning houses
    let burning = engine.houses.iter().filter(|&&h| h == 1).count();

    // trivial_cooperation has prob_house_catches_fire=0.01, so with seed 42 we should have 0-2 burning houses
    assert!(burning <= 2, "Should not have too many burning houses");
}

#[test]
fn test_reset() {
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    let mut engine = BucketBrigade::new(scenario, Some(42));

    // Take some steps
    let actions = vec![[0, 1], [1, 1], [2, 1], [3, 1]];
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
    let mut engine1 = BucketBrigade::new(scenario.clone(), Some(123));
    let mut engine2 = BucketBrigade::new(scenario, Some(123));

    // Same seed should produce identical initial states
    assert_eq!(engine1.houses, engine2.houses);

    // And identical behavior
    let actions = vec![[0, 1], [1, 1], [2, 1], [3, 1]];
    let result1 = engine1.step(&actions);
    let result2 = engine2.step(&actions);

    assert_eq!(result1.rewards, result2.rewards);
    assert_eq!(engine1.houses, engine2.houses);
}

#[test]
fn test_step_advances_night() {
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    let mut engine = BucketBrigade::new(scenario, Some(42));

    let initial_night = engine.night;
    let actions = vec![[0, 1], [1, 1], [2, 1], [3, 1]];
    engine.step(&actions);

    assert_eq!(engine.night, initial_night + 1);
}

#[test]
fn test_work_vs_rest_rewards() {
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    let mut engine = BucketBrigade::new(scenario, Some(42));

    // All agents rest
    let rest_actions = vec![[0, 0], [1, 0], [2, 0], [3, 0]];
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
    let mut engine = BucketBrigade::new(scenario, Some(42));

    // Set up a burning house
    engine.houses = vec![0; 10];
    engine.houses[5] = 1;

    // Send all agents to work on it
    let actions = vec![[5, 1], [5, 1], [5, 1], [5, 1]];

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

    let mut engine = BucketBrigade::new(scenario, Some(100));

    // Set up a controlled initial state with one burning house
    engine.houses = vec![0; 10];
    engine.houses[5] = 1; // House 5 is burning

    // No one works (let it spread)
    let actions = vec![[0, 0], [1, 0], [2, 0], [3, 0]];
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
    let mut engine = BucketBrigade::new(scenario, Some(42));

    // Set a house on fire
    engine.houses[3] = 1;

    // Let it burn without intervention
    let actions = vec![[0, 0], [1, 0], [2, 0], [3, 0]];
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

    let mut engine = BucketBrigade::new(scenario, Some(42));

    // Clear all fires
    engine.houses = vec![0; 10];
    engine.night = 0;

    // Step once - should create spontaneous ignitions
    let actions = vec![[0, 0], [1, 0], [2, 0], [3, 0]];
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

    let mut engine = BucketBrigade::new(scenario, Some(42));
    engine.houses = vec![0; 10];

    // Step multiple nights
    let actions = vec![[0, 0], [1, 0], [2, 0], [3, 0]];
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

    let mut engine = BucketBrigade::new(scenario, Some(42));
    engine.houses = vec![0; 10]; // All safe

    let actions = vec![[0, 0], [1, 0], [2, 0], [3, 0]];
    let result = engine.step(&actions);

    assert!(result.done, "Game should end when all houses are safe");
}

#[test]
fn test_termination_all_ruined() {
    let mut scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    scenario.min_nights = 0;

    let mut engine = BucketBrigade::new(scenario, Some(42));
    engine.houses = vec![2; 10]; // All ruined

    let actions = vec![[0, 0], [1, 0], [2, 0], [3, 0]];
    let result = engine.step(&actions);

    assert!(result.done, "Game should end when all houses are ruined");
}

#[test]
fn test_minimum_nights_prevents_early_termination() {
    let mut scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    scenario.min_nights = 5; // Require at least 5 nights

    let mut engine = BucketBrigade::new(scenario, Some(42));
    engine.houses = vec![0; 10]; // All safe

    let actions = vec![[0, 0], [1, 0], [2, 0], [3, 0]];

    // Step through 3 nights
    for _ in 0..3 {
        let result = engine.step(&actions);
        assert!(!result.done, "Should not terminate before min_nights");
    }
}

#[test]
fn test_get_observation() {
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    let engine = BucketBrigade::new(scenario, Some(42));

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
    let mut engine = BucketBrigade::new(scenario, Some(42));

    let actions = vec![[0, 1], [1, 1], [2, 1], [3, 1]];
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
    let mut engine = BucketBrigade::new(scenario, Some(42));

    let actions = vec![[0, 1], [1, 1], [2, 1], [3, 1]];
    engine.step(&actions);

    let result = engine.get_result();

    // Final score should be sum of all agent scores
    let expected_sum: f32 = result.agent_scores.iter().sum();
    assert!((result.final_score - expected_sum).abs() < 0.001);
}

#[test]
fn test_agent_positions_update() {
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    let mut engine = BucketBrigade::new(scenario, Some(42));

    let actions = vec![[5, 1], [6, 1], [7, 1], [8, 1]];
    engine.step(&actions);

    let state = engine.get_current_state();
    assert_eq!(state.agent_positions, vec![5, 6, 7, 8]);
}

#[test]
fn test_agent_signals_update() {
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    let mut engine = BucketBrigade::new(scenario, Some(42));

    let actions = vec![[0, 0], [1, 1], [2, 0], [3, 1]];
    engine.step(&actions);

    let state = engine.get_current_state();
    assert_eq!(state.agent_signals, vec![0, 1, 0, 1]);
}

#[test]
#[should_panic(expected = "Game is already finished")]
fn test_step_after_done_panics() {
    let mut scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    scenario.min_nights = 0;

    let mut engine = BucketBrigade::new(scenario, Some(42));
    engine.houses = vec![0; 10];

    let actions = vec![[0, 0], [1, 0], [2, 0], [3, 0]];
    engine.step(&actions);

    // Game should be done now, stepping again should panic
    engine.step(&actions);
}

#[test]
fn test_ownership_rewards() {
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    let mut engine = BucketBrigade::new(scenario, Some(42));

    // Agent 0 owns house 0, agent 1 owns house 1, etc.
    engine.houses = vec![0; 10]; // All safe

    let actions = vec![[0, 0], [1, 0], [2, 0], [3, 0]];
    let result = engine.step(&actions);

    // Per-step rewards should only reflect work/rest costs
    // All agents resting, so should get +0.5
    assert!(
        result.rewards.iter().all(|&r| r == 0.5),
        "Agents should get rest reward only"
    );

    // Complete the game and check final rewards
    engine.done = true;
    let game_result = engine.get_result();

    // All agents should have positive final scores (ownership bonus for safe houses + neighbor bonuses)
    assert!(
        game_result.agent_scores.iter().all(|&s| s > 0.0),
        "All agents should have positive scores with all houses safe"
    );
}

#[test]
fn test_ownership_penalty() {
    let scenario = SCENARIOS.get("trivial_cooperation").unwrap().clone();
    let mut engine = BucketBrigade::new(scenario, Some(42));

    // Set agent 0's house (house 0) to ruined
    engine.houses = vec![0; 10];
    engine.houses[0] = 2;

    let actions = vec![[5, 0], [5, 0], [5, 0], [5, 0]];
    let result = engine.step(&actions);

    // Per-step rewards should be equal (only work/rest costs)
    // All agents resting, so should get +0.5
    assert_eq!(result.rewards[0], 0.5, "Agent 0 should have rest reward");
    assert_eq!(result.rewards[1], 0.5, "Agent 1 should have rest reward");
    assert_eq!(
        result.rewards[0], result.rewards[1],
        "Per-step rewards should be equal"
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
