# Source: heterogeneous Double-Oracle strategy space caveat

The empirical phase diagram (`refs/empirical-phase-diagram.md`) is computed by the heterogeneous Double-Oracle solver, **not** the binary Work/REST stage-game solver implied by the analytical derivation in `ne_structure.md`. This file documents the gap and the bridging interpretation the memo's §1.2 ("Mapping the analytical primitives to the empirical strategy space") relies on.

## Empirical strategy space

- Source: `bucket_brigade/equilibrium/double_oracle_heterogeneous.py` (especially lines ~455–520 where the archetype menu is initialized).
- Each pure strategy is a **10-dimensional continuous parameter vector** of the form:

  ```
  [honesty_bias, work_tendency, neighbor_help_bias, own_house_priority,
   risk_aversion, coordination_weight, exploration_rate, fatigue_memory,
   rest_reward_bias, altruism_factor]
  ```

  Parameters are floats in $[0, 1]$ that modulate a fixed heuristic policy (`bucket_brigade/agents/heuristic.py`); see `bucket_brigade/agents/archetypes.py` for the canonical 5-archetype menu.
- The DO algorithm seeds with 5 archetypes (`FF`, `FR`, `Hero`, `Coord`, `Liar`; `archetype_names` at `double_oracle_heterogeneous.py` line 472) and grows the strategy menu by adding best responses to the current Nash mixture.
- The reported per-cell payoff and verdict are the equilibrium of the matrix game over the **discovered, finite** strategy menu — not the full continuous strategy space.

## Analytical strategy space

The derivation in `ne_structure.md` collapses each agent's strategy to a **single binary choice per night**: Work (`a^mode = 1`) or REST (`a^mode = 0`), with the house target fixed to a single house in the mean-field reduction. This is the textbook volunteer-dilemma framing.

## Mapping (the "bridge" the memo's §1.2 names)

| Empirical archetype | Work tendency | Mapped to analytical primitive |
|---|---|---|
| Hero (`HERO_PARAMS`) | $\texttt{work\_tendency} = 1.0$ | **all-Work** (always plays $a^{\text{mode}}=1$) |
| Firefighter (`FIREFIGHTER_PARAMS`) | $\texttt{work\_tendency} = 0.9$ | **near all-Work**; treated as all-Work for the analytical boundary |
| Free Rider (`FREE_RIDER_PARAMS`) | $\texttt{work\_tendency} = 0.2$, $\texttt{rest\_reward\_bias} = 0.9$ | **all-REST** (mostly plays $a^{\text{mode}}=0$) |
| Coordinator (`COORDINATOR_PARAMS`) | $\texttt{work\_tendency} = 0.6$ | mixed; not used in the canonical analytical NE candidates |
| Liar (`LIAR_PARAMS`) | $\texttt{work\_tendency} = 0.7$ | mixed; cheap-talk channel is out of the analytical reduction's scope |

The empirical verdicts then translate to the analytical regime as follows:

- `symmetric_only` cells where the best symmetric profile is Hero(d=0)|Hero(d=0)|Hero(d=0)|Hero(d=0) or Firefighter|Firefighter|Firefighter|Firefighter → **symmetric all-Work NE** in the analytical reduction.
- `asymmetric_only` cells where the best asymmetric profile is one Hero plus three Firefighters (the canonical pattern observed in the preview at $\kappa{=}0.9, c{=}0.5$, $\beta \in \{0.5, 0.9\}$) → **asymmetric "one Worker, rest free-ride" NE** in the analytical reduction. The empirical "rest" is Firefighter ($\texttt{work\_tendency}{=}0.9$), not literal REST ($\texttt{work\_tendency}{=}0.2$), because the per-agent ownership reward (50 own, 0 other) still rewards each agent for defending its own house regardless of the team-level NE structure. The analytical reduction must read "free-ride relative to the central fire" rather than "absolute REST" — see §2.3 of the memo body.
- `no_convergence` cells → **regime where neither symmetric nor asymmetric NE produces a payoff that survives the burn-out trajectory**, i.e., the per-night extinguish probability $1-(1-\kappa)^w$ is too low to defend houses against the spontaneous-ignition rate $\rho$ over $T_{\min}$ nights. Algebraic boundary inequality derived in §3.3.

## Why the gap matters for the comparison

Two failure modes the predicted-vs-empirical comparison must guard against:

1. **The analytical "all-REST is not a NE" claim** (§3.1 of the memo) is exactly true in the binary-choice framing but is **misleadingly stated** in the empirical comparison, because the empirical solver never finds an all-REST profile — every solved cell shows at least one Worker. The cells with `verdict = no_convergence` are *not* "all-REST collapses": they're "no profile in the discovered menu is approximate-NE within tolerance." The memo's §3.3 reframes this correctly.
2. **The asymmetric NE boundary inequality** (§3.2 of the memo) compares the marginal payoff of a single Worker vs. three free-riders. The empirical analogue is one Hero plus three Firefighters; Firefighters at $\texttt{work\_tendency}{=}0.9$ contribute non-zero extinguish probability, so the empirical "1 Worker, 3 free-riders" cell technically has $w \approx 1 + 3 \cdot 0.9 \cdot E[\text{at-fire}] \approx 1.3$ to 2 effective workers per fire, not the analytical $w{=}1$. This biases the empirical $\kappa$ at which the asymmetric NE first appears **upward** relative to the analytical prediction; the memo's §5 perturbation analysis names this.
