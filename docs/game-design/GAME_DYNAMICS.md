BUCKET BRIGADE — GAME DYNAMICS
--------------------------------

OVERVIEW
--------
Bucket Brigade is a cooperative multi-agent environment played on a ring
of ten houses. Each night, agents may signal their intent (to work or rest)
and then act by choosing where to go and whether to work or rest.
Fires spread probabilistically between neighboring houses, and agents
must cooperate to prevent the town from burning down. The game ends
when all fires are extinguished or all houses are ruined.

ENVIRONMENT
-----------
The town consists of ten houses arranged in a circular ring numbered 0-9.
Each house has two neighbors: (i-1) mod 10 and (i+1) mod 10.
Each house can be in one of three states:
  0 = Safe
  1 = Burning
  2 = Ruined

Each house has an owner (agents are assigned ownership in a round-robin
pattern if there are fewer than ten agents).

AGENTS
------
There are between 4 and 10 agents.
Each agent occupies a house each night and takes two decisions:

  1. Signal: broadcast publicly whether they plan to WORK or REST.
  2. Action: choose (house, mode)
       house ∈ {0..9}
       mode ∈ {WORK, REST}

A "lie" is recorded when the signal differs from the actual mode.

OBSERVATIONS
------------
At the beginning of each night, every agent receives an observation
containing:
  - All agents' signals for this night
  - All agents' current locations
  - The state of all ten houses
  - The actions taken by all agents on the previous night

All of these are public. Agents do not observe hidden intent before
actions are revealed.

NIGHTLY SEQUENCE
----------------
1. Observation phase:
   Each agent observes signals, locations, house states, and last actions.
   (Fires from previous night's spread/spark phase are visible)

2. Signal phase:
   All agents simultaneously broadcast their signal (WORK or REST).

3. Action phase:
   After observing all signals, each agent chooses an action (house, mode).

4. Extinguish phase:
   Agents respond to fires visible at start of turn.
   For each burning house with k agents working there,
   extinguish with probability (independent probabilities):
       P(extinguish) = 1 - (1 - prob_solo_agent_extinguishes_fire)^k
   Extinguished houses become Safe.

5. Burn-out phase:
   Unextinguished burning houses become Ruined.

6. Spread phase:
   Each remaining burning house ignites each neighbor (if Safe)
   with probability prob_fire_spreads_to_neighbor.
   **New fires are visible NEXT turn.**

7. Spontaneous ignition phase:
   On every night, each Safe house can catch fire spontaneously
   with probability prob_house_catches_fire.
   **New fires are visible NEXT turn.**

8. Reward and logging phase:
   Compute team and individual rewards, record all actions,
   signals, and house state updates.

9. Termination:
   The game runs for at least min_nights nights.
   After that, it ends when either:
      (a) all fires are extinguished, or
      (b) all houses are ruined, or
      (c) 100 nights have elapsed (safety limit).
   Spontaneous ignition continues throughout the game.

KEY DESIGN DECISION: Fires spread and spark at the END of each turn,
making them visible for the NEXT turn. This allows agents to observe
fire locations and coordinate strategic responses, rewarding teamwork
over luck.

SCENARIO PARAMETERS
-------------------
Each game scenario is defined by a parameter vector:

  prob_fire_spreads_to_neighbor      Fire spread probability per neighbor
  prob_solo_agent_extinguishes_fire  Probability one agent extinguishes fire
  prob_house_catches_fire            Probability house catches fire (any night)
  team_reward_house_survives         Team reward per saved house
  team_penalty_house_burns           Team penalty per ruined house
  cost_to_work_one_night             Cost per worker per night
  min_nights                         Minimum nights before termination
  num_agents                         Number of agents (4-10)
  reward_own_house_survives          Individual reward when own house survives
  reward_other_house_survives        Individual reward when neighbor house survives
  penalty_own_house_burns            Individual penalty when own house burns
  penalty_other_house_burns          Individual penalty when other house burns

Typical defaults:
  prob_fire_spreads_to_neighbor = 0.25
  prob_solo_agent_extinguishes_fire = 0.45
  team_reward_house_survives = 100
  team_penalty_house_burns = 100
  cost_to_work_one_night = 0.5
  prob_house_catches_fire = 0.01
  min_nights = 12
  num_agents = 4

REWARDS
-------
TEAM REWARD
  R_team = A * SavedFraction
            - L * BurnedFraction
            - c * total number of workers over all nights

INDIVIDUAL REWARD
  R_i = sum_t r_i(t)
        + gamma * (A * SavedFraction - L * BurnedFraction)
        - lambda_own * OwnedRuined_i(T)

  where
    r_i(t) =
       -c_i if working,
       +r_rest if resting,
       +alpha_own * change in owned Safe houses during the night.

All rewards are computed within the environment so that episodes
produce complete numerical results for ranking or reinforcement learning.

TERMINATION GUARANTEE
---------------------
The process is finite. Fires either extinguish or burn out,
and sparks cease after N_spark nights. Therefore, all houses
eventually reach a non-burning state and the game ends.

IMPLEMENTATION STATUS
--------------------

This game specification has been fully implemented in `bucket_brigade/envs/bucket_brigade_env.py` with:

- Complete environment dynamics in `BucketBrigadeEnv` class
- Scenario generation in `scenarios.py`
- Heuristic agents with 10 behavioral parameters
- JSON replay export functionality
- Batch orchestration for ranking experiments

SUMMARY
-------
The Bucket Brigade game provides:
  - Public signaling with potential deception
  - Cooperative firefighting dynamics
  - Probabilistic outcomes tied to team coordination
  - Stable termination conditions
  - Structured numeric rewards for ranking and analysis

This specification defines the deterministic update logic of the
environment, independent of any agent decision model or learning system.
