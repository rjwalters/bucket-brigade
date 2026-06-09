# Source: formal environment specification

The notation $(\beta, \kappa, \rho, c, H, N, T_{\min}, W)$, the seven-phase night ordering, and the independent-workers extinguish formula $1 - (1-\kappa)^w$ that the analytical derivation in `ne_structure.md` uses are all imported verbatim from:

- `paper/anvil_memo.env_spec.1/env_spec.md` — Bucket Brigade formal environment specification (issue #362, version 1).

Symbol pin-cites from `env_spec.md` used in this memo:

- **§1.3 (Parameter family)** — the canonical parameter vector and its admissible domain. The default scenario uses $H{=}10, N{=}4, \beta{=}0.25, \kappa{=}0.5, \rho{=}0.02, c{=}0.5, T_{\min}{=}12$; the `minimal_specialization` scenario in this memo uses the same $(H, N, T_{\min}, \rho)$ but the per-agent-ownership-dominant reward tuple $W = (R_{\text{team}}, P_{\text{team}}, r_{\text{own}}, r_{\text{other}}, p_{\text{own}}, p_{\text{other}}) = (10, 10, 50, 0, 100, 0)$.
- **§3.2 (Extinguish phase)** — $\Pr[\text{extinguish} \mid w \text{ workers}] = 1 - (1-\kappa)^w$. The single-deviation marginals that drive the §3 boundary inequalities of this memo are direct algebraic consequences of this formula.
- **§3.5 (Spontaneous ignition phase)** — $\Pr[\text{re-ignite SAFE house}] = \rho$, applied independently to every SAFE house each night. The steady-state SAFE-fraction recursion in §2 of this memo uses this verbatim.
- **§4 (Reward structure)** — the per-agent reward decomposition into (work cost, per-house ownership term, team term). The single-house mean-field reduction uses the ownership term ($r_{\text{own}}, p_{\text{own}}, p_{\text{other}}$) directly and the team term as the survival-coefficient multiplier.
- **§5.1 (Horizon and termination)** — $T_{\min}{=}12$ floor; termination via all-SAFE or all-RUINED. The episode-length proxy used to define the survival coefficient $A$ is the $T_{\min}$ horizon plus a continuation tail that absorbs into $A$.
- **§6 (Relationship to canonical templates)** — the volunteer's-dilemma reduction at $H{=}1$ is the textbook anchor for the asymmetric-NE inequality. The Stag Hunt limit at $\kappa \to 1$ is the anchor for the asymmetric-dominant regime.

Implementation cross-reference (for readers who want to sanity-check the algebra against running code):

- `bucket_brigade/envs/bucket_brigade_env.py:step` — phase dispatch.
- `bucket_brigade/envs/scenarios_generated.py:minimal_specialization_scenario` (line 570) — $W$ values used in this memo's numerical predictions.
