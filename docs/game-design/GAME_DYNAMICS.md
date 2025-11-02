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

2. Signal phase:
   All agents simultaneously broadcast their signal (WORK or REST).

3. Action phase:
   After observing all signals, each agent chooses an action (house, mode).

4. Extinguish phase:
   For each burning house with k agents working there,
   extinguish with probability:
       P(extinguish) = 1 - exp(-kappa * k)
   Extinguished houses become Safe.

5. Spread phase:
   Each remaining burning house ignites each neighbor (if Safe)
   with probability beta.

6. Burn-out phase:
   Burning houses that neither extinguished nor spread become Ruined.

7. Spark phase:
   For nights t <= N_spark, each Safe house ignites spontaneously
   with probability p_spark.

8. Reward and logging phase:
   Compute team and individual rewards, record all actions,
   signals, and house state updates.

9. Termination:
   The game runs for at least N_min nights.
   After that, it ends when either:
      (a) all fires are extinguished, or
      (b) all houses are ruined.
   Sparks stop after N_spark nights, so termination is guaranteed.

SCENARIO PARAMETERS
-------------------
Each game scenario is defined by a parameter vector:

  beta          Fire spread probability per neighbor
  kappa         Extinguish efficiency
  A             Reward per saved house
  L             Penalty per ruined house
  c             Cost per worker per night
  rho_ignite    Initial fraction of houses burning
  N_min         Minimum nights before termination
  p_spark       Probability of spontaneous ignition
  N_spark       Number of nights with sparks active

Typical defaults:
  beta = 0.25
  kappa = 0.5
  A = 100
  L = 100
  c = 0.5
  rho_ignite = 0.2
  N_min = 12
  p_spark = 0.02
  N_spark = 12

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
