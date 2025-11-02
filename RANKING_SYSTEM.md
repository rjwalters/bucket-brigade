BUCKET BRIGADE — RANKING SYSTEM
--------------------------------

PURPOSE
-------
Estimate each agent’s marginal contribution to team performance from batches
of mixed-team games, and choose future teams/scenarios to reduce uncertainty
efficiently. The environment produces numeric outcomes; the ranking system
fits a surrogate model, computes per-agent values, and iterates with
batch design.

DATA MODEL
----------
Each completed game (episode) yields one row:

  episode_id     unique id
  scenario_id    id of scenario parameter vector (phi_c)
  team           list of agent ids who played
  team_reward    scalar team outcome (R_team)
  agent_rewards  list of per-agent returns (optional for modeling)
  nights_played  integer
  replay_path    file path to JSON replay (for visualization/debug)

Optionally store the exact scenario vector (phi_c) alongside scenario_id.

SURROGATE MODELS
----------------
We model expected team reward as a function of team composition and scenario.
Use log-reward (y = log(R_team)) if rewards are multiplicative; else use
the raw reward.

(1) Additive model (fast, interpretable)
  y_g = alpha + mu_c(g) + sum_{i in team(g)} theta_i + eps_g
  eps_g ~ Normal(0, sigma^2)

  - alpha: intercept
  - mu_c: scenario offsets (one per scenario or linear in scenario features)
  - theta_i: agent skill parameters
  - y_g: observed outcome for game g

(2) Additive + low-rank interactions (scales to many agents)
  y_g = alpha + mu_c(g) + sum_i theta_i
        + < sum_{i in team(g)} u_i , sum_{i in team(g)} v_i > + eps_g
  with u_i, v_i in R^k (k small, e.g., 4–16), strong regularization.

PRIORS / REGULARIZATION
-----------------------
Use Gaussian priors (ridge) for all parameters:
  theta_i ~ Normal(0, tau^2)
  mu_c    ~ Normal(0, tau_c^2)
  u_i, v_i ~ Normal(0, tau_uv^2)

For the additive model, this is Bayesian linear regression with a
closed-form Gaussian posterior:

  beta_post ~ Normal(beta_hat, Sigma)

  beta_hat = (X^T X + lambda I)^(-1) X^T y
  Sigma    = sigma^2 (X^T X + lambda I)^(-1)

Here X is the design matrix with columns for agents (and scenarios/features).

MARGINAL VALUE (TARGET TO RANK)
-------------------------------
Define each agent’s deployment-relevant value as expected marginal contribution:

  v_i = E_{(S,c) ~ deployment} [ f_hat(S ∪ {i}, c) - f_hat(S, c) ]

- For the additive model, v_i = theta_i directly.
- With interactions, estimate v_i by Monte Carlo over (S, c):
  sample teams S (without i) and scenarios c from your deployment distribution,
  evaluate the surrogate difference, average.

Retain uncertainty by sampling beta ~ Normal(beta_hat, Sigma) and recomputing v_i.

BATCH DESIGN (NEXT TEAMS TO PLAY)
---------------------------------
Given a candidate set of designs d = (team S, scenario c), each has a design row x(d).

(A) A-optimal (reduce average posterior variance)
  Choose a batch B that minimizes trace(Sigma_new), where
    Sigma_new = ( (X^T X + lambda I) + (1/sigma^2) * sum_{d in B} x(d)^T x(d) )^(-1)
  Greedy selection: add designs one at a time by largest trace reduction using
  Sherman–Morrison rank-1 updates.

(B) D-optimal (shrink posterior volume)
  Maximize log det( (X^T X + lambda I) + (1/sigma^2) * sum x(d)^T x(d) ).
  Greedy, with tie-breaking for diversity.

(C) Thompson Sampling (robust default)
  Sample beta_t ~ Normal(beta_hat, Sigma).
  Score candidates by disagreement or predictive variance, e.g.:
    score(d) = ( y_hat(d) - y_t(d) )^2  +  kappa * Var_hat[y(d)]
  Pick the top m feasible designs; promote diversity (see below).

DIVERSITY & FEASIBILITY
-----------------------
Enforce practical constraints during greedy selection:
  - Min/max appearances per agent in a batch.
  - Balanced scenario coverage.
  - Avoid near-duplicate rosters (Hamming distance thresholds).
  - Team size distribution matches deployment.
  - Roster graph connectivity (do not isolate agents).

Use simple feasibility checks plus a diversity mechanism:
  - Pivoted Cholesky on candidate Gram matrix, or
  - Lightweight Determinantal Point Process (DPP), or
  - Explicit spacing rules on agent overlap.

ESTIMATION LOOP
---------------
(a) Run a batch of games using selected teams/scenarios.
(b) Append rows to the results table; (optionally) write replay JSONs.
(c) Refit/Update posterior (closed-form for additive; MAP/variational for low-rank).
(d) Compute v_i and credible intervals.
(e) Check stopping criteria:
    - ranking stability (e.g., Kendall tau > threshold across two iterations);
    - max SE(v_i) below tolerance;
    - budget reached.
(f) If not stopped, design next batch and repeat.

DESIGN MATRIX CONSTRUCTION
--------------------------
For G games and N agents:

  - Agent block: one column per agent (binary: 1 if in team, else 0).
  - Scenario block:
      • One-hot per scenario id; or
      • Linear features from the scenario vector phi_c;
      • Center columns to improve conditioning.

Rows have about (team_size + scenario_cols) nonzeros; store X in CSR format.

SCALING TRICKS
--------------
- Sparse linear algebra for X and incremental rank-1 updates of Sigma.
- Woodbury identity: treat agent and scenario blocks separately if needed.
- For interactions, keep k small and fit with stochastic optimizers; for acquisition,
  linearize at current params or use TS with MC predictive variance.

DIAGNOSTICS
-----------
- Roster graph connectivity: ensure most agents belong to one giant component.
- Posterior predictive checks: compare predicted vs observed rewards by scenario and team size.
- Leverage/influence by agent: detect agents used in narrow contexts only.
- Uncertainty monotonicity: trace(Sigma) and mean SE(v_i) should decrease per iteration.

FILE FORMATS
------------
Results table (CSV/SQLite):
  episode_id, scenario_id, team (JSON list), team_reward, nights_played, replay_path

Replay JSON (for visualizer):
  scenario: dict of phi_c
  nights: list of {night, houses[10], signals[N], actions[N][2], rewards[N]}

MINIMAL CONFIG TO RUN
---------------------
- Number of agents (N) and agent ids.
- Team-size distribution (e.g., uniform over {4..10}).
- Scenario set or sampler over phi_c.
- Batch size per iteration (e.g., 32–256 games).
- Regularization lambda and noise sigma^2 (or estimate sigma^2 from residuals).

IMPLEMENTATION STATUS
--------------------

Basic ranking infrastructure has been implemented:

- Batch orchestration system in `scripts/run_batch.py`
- Data collection and CSV export functionality
- Agent parameter storage and tracking
- Basic result aggregation and summary statistics

Full Bayesian ranking system (ridge regression, uncertainty quantification, adaptive batch design) is planned for the next development stage.

OUTPUTS
-------
- Agent scores: theta_i (additive) or v_i (interaction-aware).
- Uncertainties: SE(theta_i) or CI(v_i) via posterior sampling.
- Leaderboard: sorted agents with confidence intervals.
- Suggested next batch: list of (team, scenario) designs meeting constraints.

SUMMARY
-------
The ranking system is a batch Bayesian design loop over combinatorial team
choices. It maintains a simple, interpretable surrogate (additive, with an
optional low-rank synergy term), computes deployment-relevant marginal values,
and adaptively selects future teams and scenarios to reduce uncertainty. It is
data-format compatible with the Bucket Brigade environment and the replay
visualizer, and scales with sparse linear algebra and greedy information-based
selection.
