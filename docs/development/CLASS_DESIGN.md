BUCKET BRIGADE — CLASS DESIGN AND IO SPECIFICATION
---------------------------------------------------

PURPOSE
--------
This document describes the class structure for the Bucket Brigade simulator,
including all data inputs and outputs. It is designed for forward compatibility
with the PufferLib reinforcement-learning framework while retaining full
support for heuristic scripted agents and ranking experiments.

CLASS OVERVIEW
--------------
bucket_brigade.envs.bucket_brigade_env
    A multi-agent environment implementing the standard Gym / PufferLib API.

bucket_brigade.envs.scenarios
    Defines parameter distributions and generators for randomized game setups.

bucket_brigade.agents.AgentBase
    Common base class for all agents. Provides a unified interface compatible
    with both scripted heuristics and learned PufferLib policies.

bucket_brigade.orchestration.orchestrator
    Runs batches of games, stores results, and feeds data into the ranking model.

CLASS DESIGN
-------------
1. BucketBrigadeEnv
   ----------------
   Responsibilities:
     - Manage environment state (houses, agents, fires, nights).
     - Implement reset() and step() following Gym/PufferLib conventions.
     - Generate initial scenario and randomize parameters.
     - Record full trajectories for replay.

   Initialization:
     BucketBrigadeEnv(scenario: Scenario)

   Methods:
     reset() -> ObservationDict
       Resets the environment to a new game instance.
       Generates a scenario, initializes houses and agents, and returns the
       initial observation shared by all agents.

     step(actions: np.ndarray) -> (ObservationDict, np.ndarray, np.ndarray, dict)
       Executes one night of play. Takes per-agent actions (house, mode),
       updates environment state, computes rewards, and returns:
         observations   dict of arrays describing new state
         rewards        per-agent float array
         dones          per-agent boolean array
         info           optional debug and logging information

     render() -> None
       Optional; provides visualization hooks for local debugging.

     save_replay(path: str) -> None
       Writes JSON file containing complete episode trajectory.

   Key attributes:
     self.houses        np.ndarray[int8], shape (10,)
     self.locations     np.ndarray[int8], shape (N,)
     self.signals       np.ndarray[int8], shape (N,)
     self.last_actions  np.ndarray[int8], shape (N,2)
     self.scenario      Scenario
     self.rewards       np.ndarray[float32], shape (N,)
     self.done          bool
     self.trajectory    list[dict], for replay export

2. Scenario
   --------
   Represents the stochastic configuration of a game.

   Parameters:
     beta          fire spread probability per neighbor
     kappa         extinguish efficiency
     A             reward per saved house
     L             penalty per ruined house
     c             cost per worker per night
     rho_ignite    initial burning fraction
     N_min         minimum nights before termination
     p_spark       spontaneous ignition probability
     N_spark       spark duration (nights)
     num_agents    number of agents participating

   Scenario sampling:
     - Scenarios are drawn from distributions defined in scenarios.py.
     - Example: beta ~ Uniform(0.15, 0.35)
                kappa ~ Uniform(0.4, 0.6)
                rho_ignite ~ Uniform(0.1, 0.3)
                p_spark ~ Bernoulli(0.5) * Uniform(0.01, 0.05)
     - Each run of env.reset() can accept a Scenario or generate one randomly
       from these distributions.

   Agent access:
     - Each agent receives the complete scenario parameters once, at game start.
     - Scenario features are included in the initial observation (or as a
       separate "scenario_info" dictionary) so that agents can adapt to the
       specific risk level and difficulty of each run.

3. AgentBase
   ----------
   Provides a uniform interface compatible with both heuristic and learned agents.

   Methods:
     reset() -> None
       Clears agent internal state between games.

     act(obs: dict) -> np.ndarray[2]
       Given the current observation, returns an action:
       (house_index, mode_flag)
         house_index ∈ [0..9]
         mode_flag ∈ {0=REST, 1=WORK}

   Derived classes:
     - HeuristicAgent: scripted decision logic for initial experiments.
     - RandomAgent: baseline random actor.
     - PufferPolicyAdapter: wraps PufferLib Actor-Critic models.

4. Observation and Action Structures
   ---------------------------------
   ObservationDict (shared across agents):
     {
       "signals":       np.ndarray[int8], shape (N,), current signals,
       "locations":     np.ndarray[int8], shape (N,), current agent positions,
       "houses":        np.ndarray[int8], shape (10,), house states,
       "last_actions":  np.ndarray[int8], shape (N,2), previous-night actions,
       "scenario_info": np.ndarray[float32], shape (len(phi_c),), scenario vector
     }

   Action format (per agent):
     np.ndarray[int8], shape (2,)
       [house_index, mode_flag]

   Return signature of env.step():
     observations   dict
     rewards        np.ndarray[float32], shape (N,)
     dones          np.ndarray[bool], shape (N,)
     info           dict (metadata, debug, logging)

IO FLOW
-------
1. Inputs:
   - Scenario parameters (random or fixed).
   - Agent policies (heuristic or learned).
   - Random seed for reproducibility.

2. Outputs:
   - Trajectory list for each episode (written as JSON):
       [ { "night": int,
           "houses": [int]*10,
           "signals": [int]*N,
           "actions": [[int,int]]*N,
           "rewards": [float]*N } ... ]
   - Summary CSV or SQLite entry for ranking:
       episode_id, scenario_id, team, team_reward, agent_rewards, replay_path

PUFFERLIB COMPATIBILITY
-----------------------
The environment is fully compliant with the PufferLib multi-agent interface:

  - reset() and step() signatures match pe.MultiAgentEnv expectations.
  - obs_space and act_space defined as pe.spaces.Dict and pe.spaces.MultiDiscrete.
  - vectorized per-agent rewards and dones.
  - supports parallelization via pufferlib.vector wrappers.
  - deterministic RNG seeding for reproducible batches.

GAME SCENARIO GENERATION
------------------------
Each new episode begins by sampling a scenario from a parameter generator.

Example in scenarios.py:

  def random_scenario(num_agents: int) -> Scenario:
      return Scenario(
          beta=np.random.uniform(0.15,0.35),
          kappa=np.random.uniform(0.4,0.6),
          A=100, L=100, c=0.5,
          rho_ignite=np.random.uniform(0.1,0.3),
          N_min=np.random.randint(10,20),
          p_spark=np.random.choice([0, np.random.uniform(0.01,0.05)]),
          N_spark=N_min,
          num_agents=num_agents
      )

The sampled scenario is stored internally and distributed to all agents
through the initial observation as a numerical feature vector (phi_c).
Agents are free to condition their behavior on these values but do not
see any randomness beyond what is encoded in the scenario.

TERMINATION
-----------
Termination occurs when:
  - The minimum required nights have passed, and
  - Either all houses are safe, or all houses are ruined.

No hard cap is needed because spark events stop after N_spark nights
and burning houses always transition to a terminal state.

IMPLEMENTATION STATUS
--------------------

This class design has been fully implemented:

- `BucketBrigadeEnv` class in `bucket_brigade/envs/bucket_brigade_env.py`
- `Scenario` class and generation functions in `bucket_brigade/envs/scenarios.py`
- `AgentBase` and `HeuristicAgent` classes in `bucket_brigade/agents/`
- PufferLib-compatible API with observation/action spaces
- JSON replay export functionality
- Batch orchestration system

SUMMARY
-------
This class design provides:
  - Full reproducibility and structured state updates.
  - A clean API compatible with PufferLib and Gym.
  - Probabilistic scenario generation with agent-visible parameters.
  - Simple extensibility for heuristic or learned agents.
  - Clear input/output data structures for downstream ranking and visualization.
