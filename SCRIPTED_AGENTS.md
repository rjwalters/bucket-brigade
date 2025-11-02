BUCKET BRIGADE — SCRIPTED HEURISTIC AGENT DESIGN
------------------------------------------------

PURPOSE
--------
Scripted heuristic agents form the initial population for simulation and ranking
experiments. They provide interpretable baselines and allow generation of large
agent sets from parameter vectors. Each agent’s decision policy is a function of
its internal parameters and current observations. The same interface will later
be used by learned agents so that both can be evaluated in identical conditions.

GOALS
-----
1. Define a compact, parameterized behavior model that can be instantiated from
   random vectors or hand-crafted archetypes.
2. Allow continuous variation between strategies to simulate realistic diversity.
3. Demonstrate that the ranking engine can recover each agent’s latent quality
   efficiently using only team-level outcomes.

AGENT CLASS STRUCTURE
---------------------
All scripted agents subclass AgentBase and implement:

  class HeuristicAgent(AgentBase):
      def __init__(self, params: np.ndarray, agent_id: int)
      def reset(self)
      def act(self, obs: dict) -> np.ndarray[2]

The parameter vector defines behavioral tendencies that influence signaling,
house targeting, and risk preferences.

PARAMETER VECTOR STRUCTURE
--------------------------
Each agent is defined by a fixed-length parameter vector θ (length ~8–12).

Example parameterization:

  θ = [
      honesty_bias,         # probability of truthful signaling
      work_tendency,        # base probability to choose mode=WORK
      neighbor_help_bias,   # preference for helping burning neighbor houses
      own_house_priority,   # preference for defending own house
      risk_aversion,        # sensitivity to number of burning houses
      coordination_weight,  # trust in other agents' signals
      exploration_rate,     # randomness in action selection
      fatigue_memory,       # (optional) inertia to repeat last action
      rest_reward_bias,     # preference for rest if fires are low
      altruism_factor       # willingness to work even if personal cost high
  ]

All values are real numbers normalized to [0,1]. The agent’s decision rules
use these values as weights or thresholds in deterministic or probabilistic
policies.

AGENT DECISION LOGIC
--------------------
At each night t, the agent receives observation obs containing:
  signals[N], locations[N], houses[10], last_actions[N,2], scenario_info[k].

The decision proceeds in two stages.

1. Signal selection:
   - Compute a working-intent probability:
         p_work_signal = work_tendency * (1 - rest_reward_bias)
   - Draw signal = WORK with probability p_work_signal.
   - If honesty_bias < 1, actual action may differ (lie).

2. Action selection:
   - Determine candidate houses:
        • own house
        • burning houses
        • neighbors of burning houses
   - Score each candidate by:
        score(h) = own_house_priority * I(owned)
                 + neighbor_help_bias * I(neighbor_of_burning)
                 + coordination_weight * fraction_of_agents_signaling_WORK_on_h
                 - risk_aversion * global_burning_fraction
   - Select highest-scoring house (or sample proportionally).
   - Choose mode:
        • WORK if signal==WORK with probability honesty_bias,
          else flip to REST (lie).
        • If signal==REST, may still WORK with small probability (false modesty).

The resulting action is (house_index, mode_flag).

RANDOM GENERATION
-----------------
To create a population of random heuristic agents:

  1. Sample parameter vectors θ_i ~ Uniform(0,1)^d.
  2. Optionally constrain correlations (e.g., honesty vs. altruism).
  3. Normalize or clip to ensure plausible behaviors.

These random agents create a diverse but continuous behavior space suitable
for ranking experiments.

ARCHETYPAL AGENTS
-----------------
Predefined parameter sets will represent interpretable behavioral archetypes.

Examples:

  The Firefighter
    honesty_bias = 1.0
    work_tendency = 0.9
    own_house_priority = 0.8
    altruism_factor = 0.8
    exploration_rate = 0.1
    neighbor_help_bias = 0.5
    coordination_weight = 0.7
    risk_aversion = 0.5

  The Liar
    honesty_bias = 0.1
    work_tendency = 0.7
    coordination_weight = 0.8
    rest_reward_bias = 0.4
    altruism_factor = 0.2
    exploration_rate = 0.3

  The Free Rider
    honesty_bias = 0.7
    work_tendency = 0.2
    own_house_priority = 0.9
    altruism_factor = 0.0
    rest_reward_bias = 0.9

  The Heroic Martyr
    honesty_bias = 1.0
    work_tendency = 1.0
    altruism_factor = 1.0
    risk_aversion = 0.1
    fatigue_memory = 0.9

  The Cautious Coordinator
    honesty_bias = 0.9
    work_tendency = 0.6
    coordination_weight = 1.0
    risk_aversion = 0.8
    exploration_rate = 0.05

These archetypes can be stored as a dictionary of named presets for reproducible
experiments. Mixing random and archetypal agents ensures a heterogeneous
population with both structured and stochastic behaviors.

SIGNALING AND MEMORY
--------------------
Each agent may maintain short-term memory of:
  - who lied last night (inferred from signal vs. action),
  - which houses were burning and who responded,
  - its own last action (for inertia).

Memory can be stored as small state vectors (e.g., honesty scores per peer)
and updated after each step. For early experiments, this can be disabled.

EVALUATION AND RANKING
----------------------
A ranking tournament uses a mix of random and archetypal agents.

Procedure:
  1. Generate a pool of agents with both random and fixed parameters.
  2. Randomly sample teams and scenarios; run many episodes.
  3. Collect team-level rewards and fit the surrogate model.
  4. Compare recovered skill estimates (theta_i_hat) to known ground truth
     metrics such as average contribution or expected team reward.

This demonstrates the ranking engine’s ability to efficiently estimate
individual performance from aggregate outcomes.

IMPLEMENTATION STATUS
--------------------

Heuristic agents have been fully implemented in `bucket_brigade/agents/heuristic_agent.py` with:

- `HeuristicAgent` class with 10 behavioral parameters
- Archetypal agent presets (firefighter, free_rider, coordinator, etc.)
- Random agent generation for diverse populations
- Integration with environment observation/action interface

EXTENSIONS
----------
- Parameter evolution: sample new agents near top performers.
- Bayesian inference over parameter vectors from observed behavior.
- Visualization of population distributions and emergent cooperation.

SUMMARY
-------
Heuristic agents are parameterized by continuous behavioral traits that control
how they signal, lie, and act in the environment. Archetypal configurations
anchor interpretable roles, while random draws fill the surrounding space.
This population provides a controlled yet diverse testbed for validating the
ranking system’s ability to recover latent contributions from team-based games.
