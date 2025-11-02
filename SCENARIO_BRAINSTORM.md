BUCKET BRIGADE — SCENARIO BRAINSTORM
-------------------------------------

PURPOSE
--------
To design and catalog a set of game scenarios where the optimal strategy or
team behavior is clearly defined and explainable. These scenarios provide
ground truth test cases for evaluating both agent decision quality and the
ranking system's ability to recover true skill orderings.

We seek parameter regimes and environmental configurations that lead to
distinct coordination challenges, clear equilibria, or trade-offs between
selfish and cooperative behavior.

SCENARIO CATEGORIES
-------------------
1. Trivial Cooperation
2. Early Containment
3. Greedy Neighbor
4. Sparse Heroics
5. Rest Trap
6. Chain Reaction
7. Deceptive Calm
8. Overcrowding
9. Honest vs Liar
10. Mixed Motivation

Each of these can be defined by a characteristic parameter vector (phi_c) and
expected optimal strategies.

---

1. TRIVIAL COOPERATION
----------------------
Description:
  Fires are rare and extinguish easily. Everyone working together guarantees
  full success.

Parameters:
  beta = 0.15     (low spread)
  kappa = 0.9     (high extinguish rate)
  rho_ignite = 0.1
  p_spark = 0.0
  c = 0.5

Optimal strategy:
  - All agents signal WORK honestly.
  - Divide evenly across burning houses.
  - Resting is suboptimal because the cost of work is small relative to A/L.

Purpose:
  - Baseline for full cooperation.
  - Verifies agents can learn to avoid redundant work on same house.

---

2. EARLY CONTAINMENT
--------------------
Description:
  Fires start aggressively but can be stopped with early coordination.
  Delay causes cascading failure.

Parameters:
  beta = 0.35
  kappa = 0.6
  rho_ignite = 0.3
  N_min = 12

Optimal strategy:
  - Everyone works in early nights, focus on burning clusters.
  - Once no fires remain, rest collectively.
  - Early miscoordination leads to widespread ruin.

Purpose:
  - Tests timing: work early, rest later.
  - Distinguishes planners from short-sighted agents.

---

3. GREEDY NEIGHBOR
------------------
Description:
  Fires spread slowly, but helping neighbors costs valuable rest time.
  Selfish agents can rely on others to work.

Parameters:
  beta = 0.15
  kappa = 0.4
  rho_ignite = 0.2
  c = 1.0
  A/L = 100/100

Optimal strategy:
  - Defend your own house first.
  - Cooperation only matters when two adjacent houses burn.
  - Free-riding is rewarded if enough others help.

Purpose:
  - Creates social dilemma.
  - Ideal for testing ranking under mixed cooperation and self-interest.

---

4. SPARSE HEROICS
-----------------
Description:
  Few workers can make the difference. Fires spread slowly but do not stop
  spontaneously. The team must dispatch minimal firefighters.

Parameters:
  beta = 0.1
  kappa = 0.5
  rho_ignite = 0.15
  N_min = 20
  c = 0.8

Optimal strategy:
  - One or two agents work per fire each night.
  - Others rest to save cost.
  - Overworking wastes points; underworking allows spread.

Purpose:
  - Tests allocation efficiency and implicit coordination.
  - Good for ranking: heroes vs freeloaders visible in outcome.

---

5. REST TRAP
------------
Description:
  Fires almost always extinguish by themselves, so rational agents rest.
  But rare long fires can destroy the town if everyone rests too long.

Parameters:
  beta = 0.05
  kappa = 0.95
  p_spark = 0.02
  rho_ignite = 0.1
  c = 0.2

Optimal strategy:
  - Rest until one persistent fire survives >2 nights.
  - Then mobilize minimal workers.
  - Overreaction wastes cost.

Purpose:
  - Tests situational awareness and adaptability.
  - Encourages observation-based triggers.

---

6. CHAIN REACTION
-----------------
Description:
  High spread and moderate extinguish rate.
  Fire containment requires distributed teams at multiple fronts.

Parameters:
  beta = 0.45
  kappa = 0.6
  rho_ignite = 0.3
  N_min = 15
  c = 0.7

Optimal strategy:
  - Split into two or three work groups covering separate clusters.
  - Avoid overconcentration.
  - Communication (signals) critical for coordination.

Purpose:
  - Tests spatial planning and teamwork under pressure.
  - Good for visual demonstrations.

---

7. DECEPTIVE CALM
-----------------
Description:
  Low ignition rate but occasional flare-ups. Honest signaling provides
  advantage; lying or ignoring signals causes inefficiency.

Parameters:
  beta = 0.25
  kappa = 0.6
  rho_ignite = 0.1
  p_spark = 0.05
  N_min = 20
  c = 0.4

Optimal strategy:
  - Signal truthfully.
  - React quickly to new fires.
  - Maintain partial coverage rather than clustering.

Purpose:
  - Evaluates honesty and responsiveness.
  - Good for comparing honest vs deceptive agents.

---

8. OVERCROWDING
---------------
Description:
  Fires are few but attract too many workers, reducing team efficiency.
  The marginal return per worker drops quickly.

Parameters:
  beta = 0.2
  kappa = 0.3
  rho_ignite = 0.1
  c = 0.6
  A/L = 50/100

Optimal strategy:
  - Assign at most two agents per burning house.
  - Resting or spreading out yields better reward.

Purpose:
  - Tests agents' ability to avoid redundant actions.
  - Highlights coordination benefits.

---

9. HONEST VS LIAR
-----------------
Description:
  Mixed population of honest and dishonest agents. The scenario
  amplifies the advantage or disadvantage of deception.

Parameters:
  beta = 0.25
  kappa = 0.5
  rho_ignite = 0.2
  c = 0.6
  N_min = 15
  honesty distribution bimodal (0.1 and 0.9)

Optimal strategy:
  - Honesty dominates in repeated play because coordination fails otherwise.
  - Short-term: liars exploit others' trust.

Purpose:
  - Evaluates how ranking separates honest cooperators from manipulators.

---

10. MIXED MOTIVATION
--------------------
Description:
  Each agent owns a house and gets extra reward for saving it.
  Fires spread moderately; selfishness conflicts with team optimum.

Parameters:
  beta = 0.3
  kappa = 0.5
  rho_ignite = 0.2
  lambda_own = 50
  c = 0.6

Optimal strategy:
  - Help others only when adjacent fires threaten your property.
  - Pure cooperation still yields higher team reward, but less individual gain.

Purpose:
  - Classic tension between self-interest and global benefit.
  - Useful for studying stable mixed equilibria.

---

SCENARIO SAMPLING PLAN
----------------------
To generalize beyond fixed cases, define named distributions:

  EasyCoop   -> beta~U(0.1,0.2), kappa~U(0.7,0.9)
  Crisis     -> beta~U(0.3,0.5), kappa~U(0.4,0.6)
  SparseWork -> beta~U(0.1,0.2), kappa~U(0.4,0.6), c~U(0.6,0.9)
  Deception  -> honesty_bias bimodal, beta~0.25, p_spark~0.04

Scenario IDs and ranges can be stored in scenarios.py for random sampling
or explicit selection in experiments.

---

SUMMARY
-------
These scenarios create varied conditions with known optimal or dominant
strategies. They will serve as benchmark environments for verifying that
the ranking engine correctly identifies high-contributing agents and that
visualizations produce recognizable patterns of success or failure.
