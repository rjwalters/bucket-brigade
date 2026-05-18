# Thesis: PPO is locally trapped in the random-play basin on social-dilemma games

**Date:** 2026-05-17 (revised after #271 verdict — was "active anti-cooperation gradient")
**Status:** Working hypothesis. #271 K=200 verdict (PR #277) ruled out the strongest form (active anti-attractor). #270 PPO-from-BC-init is now the load-bearing remaining experiment.

## Thesis statement (revised)

In multi-agent cooperative games structured as social dilemmas, **vanilla PPO converges to a basin near random play and cannot escape via local gradient updates, even when cooperative basins are reachable in principle**. The 5 standard PPO-side interventions (MAPPO, obs differentiation, spatial reward, curriculum, action-shaping) do not move the plateau. Cooperative behavior is recoverable only via training procedures that bypass the local-search dynamics (BC-init, population-based escape, hierarchical decomposition).

Three load-bearing empirical claims (revised):

1. **Cooperation is representable** ✅: BC from a hand-tuned cooperative specialist reaches `gap_closed = 0.934` on the default `PolicyNetwork(hidden_size=64)` (PR #278 cross-finding). The architecture has the capacity.
2. **Cooperation requires non-local initialization** (pending #270): PPO from random init plateaus at gap_closed ≈ 0.18. The discriminating experiment — whether PPO continuation from a BC-warm-started policy *holds* near specialist — answers whether the specialist basin is locally stable under PPO updates.
3. ~~Random-init policies are closer to cooperation than PPO converges to~~ ❌ **FALSIFIED by PR #277**: best-of-1000 random nets, with K=200 stability re-eval, top-10 phase-2 mean = gap_closed ≈ 0.0 (regressed to random play). Phase-1 → phase-2 drift = -46 points. PPO is **not** doing worse than naive random search. Random nets ≈ uniform random play; PPO finds the same locale.

The previous strong form of the thesis ("active anti-cooperation gradient") **is falsified.** The pivot is to:

> **The per-step advantage estimator on bucket-brigade is too local to escape the random-play basin via gradient updates from random init.** The gradient direction is not actively misleading; the gradient is correct locally but the cooperative basin is unreachable from random init's basin via small-step descent. This is a Goodhart-shaped failure at the level of *gradient locality*, not gradient direction.

## What we know empirically (post-#277)

- BC ✅ reaches near-specialist behavior at default architecture (#278)
- Best-of-N random search ❌ does NOT find policies above PPO's plateau (#277)
- 5 PPO-side interventions plateau at gap_closed ∈ [0.07, 0.18] across MAPPO / obs-diff / spatial / curriculum / action-shaping (PRs #232, #233, #228, #266, #268)

The combination implies: **cooperation is in a different basin than random init, and PPO cannot get from one to the other.** Whether the cooperative basin is *stable* (basin trap) or *unstable* under PPO updates (deeper Goodhart failure) is what #270 will answer.

## Mechanism — why this happens

Bucket-brigade is a multi-agent fire-fighting game where each agent owns one house in a ring of 10. Per night each agent chooses `(house, work, signal)`. Working pays a small immediate cost; resting is free. Houses on fire might be saved if enough agents work at them this turn (Bernoulli). If no one works, fires spread and ruin houses, distributing a large team-negative reward over the next ~13 nights.

GAE-Adam-PPO computes `A(s,a) = r + γV(s') - V(s)`. Per-step structure on this game:

| Action | Immediate reward |
|---|---|
| REST (always) | small + (saved work cost), occasional - (own house burns) |
| WORK at burning house | small - (work cost) + rare large + (shared across coordinators if fire extinguished) |
| WORK at safe house | small - (work cost) |

The locally-greedy gradient sees REST > WORK in expectation. Everyone defecting → houses burn → large team penalty distributed thinly across ~52 (agent, step) pairs → swamped by per-step variance.

This is the canonical defection equilibrium of a social dilemma. The game is *designed* to require cooperation; PPO is designed to find per-step advantage maxima; these objectives misalign by construction.

## Connection to AI alignment (the bigger thesis)

This is a microcosm of **Goodhart's law in RL**: PPO optimizes the measure (per-step advantage estimate) faithfully, but the measure isn't faithful to the long-horizon cooperative objective. The local gradient *correctly* points toward defection because defection genuinely maximizes immediate per-step reward.

If this generalizes — i.e., the same misalignment shows up in any RL setting where the natural reward decomposition incentivizes immediate per-step gain at the cost of long-horizon collective outcomes — then:

- "Careful training" (BC-init, curriculum, RLHF, reward shaping, multi-objective auxiliary terms) is not a hyperparameter detail; it is the alignment intervention
- Standard RL benchmarks may systematically *underestimate* the prevalence of this failure mode because they are typically single-agent or non-dilemma
- This frames classical alignment concerns (mesa-optimization, specification gaming, deceptive alignment) at the level of RL gradient dynamics rather than only at the level of high-level objective design

## Relevant published work (working bibliography — verify before citing)

These are starting points for the lit review; need to confirm exact framing and authors before paper. Marked `[verify]` where my memory is uncertain.

### Cooperative MARL & credit assignment
- **COMA — Counterfactual Multi-Agent Policy Gradients** (Foerster et al., 2018). Explicit per-agent counterfactual baselines. Addresses credit assignment in cooperative settings. We have not tried this. **Worth investigating as next intervention.**
- **MADDPG / MAPPO** — centralized critic, decentralized actor. We tried MAPPO (#225), tier 3.
- **QMIX / VDN / MAVEN** — value decomposition. Untried.
- **Sequential Social Dilemmas** (Leibo et al. 2017, DeepMind) — the framing this work is in conversation with. They studied cooperation/defection emergence under different reward structures. `[verify]`

### Misalignment & local gradient failure
- **Goodhart's Law in RL** — Krakovna et al. specification gaming compendium. `[verify]` — there's an Anthropic / DeepMind document collecting specification gaming examples.
- **Mesa-optimization** (Hubinger et al. 2019) — the trained policy has its own (different) objective than the training objective. Conceptually adjacent.
- **Influence-based intrinsic motivation** (Jaques et al. 2019, "Social Influence as Intrinsic Motivation"). Adds an intrinsic reward for behavior that influences other agents. Could be a direct intervention for our problem.

### Long-horizon credit / coarse-grained gradients
- **Hierarchical RL** — Sutton's options framework, FeUdal Networks (Vezhnevets et al.), HAC, h-DQN. Operate at multiple timescales. Direct fit for the user's "coarse-grained gradient" intuition.
- **GAE** (Schulman et al. 2016) — `λ` interpolates between high-bias bootstrap (`λ=0`) and high-variance Monte Carlo (`λ=1`). Pushing `λ → 1` gives more Monte Carlo credit.
- **Reward shaping with potential functions** (Ng, Harada, Russell 1999). Optimal policy invariance under `Φ(s) - γΦ(s')` shaping. We tried action-shaping (#262) — flat. Worth trying *potential-based* state-value shaping next.
- **Hindsight Credit Assignment** (Harutyunyan et al. 2019, DeepMind). `[verify]` Reweights returns toward actions that retrospectively caused observed rewards.
- **DAgger / BC + RL hybrids** (Ross & Bagnell 2011). Existing BC-init work (#270) is in this family.

### Multi-agent algorithms that might escape the gradient
- **LOLA — Learning with Opponent Awareness** (Foerster et al. 2018). Each agent's gradient includes a term predicting opponents' learning step. Designed for exactly the iterated-prisoner's-dilemma style problem we have. Direct fit.
- **MASAC / Off-policy MARL** — using off-policy data may sidestep the on-policy gradient bias.
- **Population-Based Training** (Jaderberg et al. 2017) — explicit population diversity + mutation can escape local basins.

## Brainstorm: "mix in coarse-grained gradients" — concrete interventions

User's intuition: the per-step gradient is biased toward defection; correct it by mixing in long-horizon signal. This is a research-direction sketch, not a commitment.

### Closest to "literal mix-in"

1. **GAE-λ extension** — push `λ → 0.99 or 1.0` in our PPO config. Cheap; possibly already covered in #145/#174 ablations but worth confirming. If high-λ shifts the verdict, our thesis is "GAE bootstrap bias is the misalignment source."
2. **N-step return augmentation** — add an n-step (n large, e.g. 13 = episode length) advantage as a separate gradient term. Combine with standard PPO advantage. Direct test of the coarse-grained intuition.
3. **Episode-return baseline as an auxiliary loss** — give the value head a second target = full-episode return (not just bootstrapped 1-step). Penalize deviation. The full-episode return is per-episode, so its gradient is unbiased Monte Carlo but high-variance.

### Hierarchical / option-based

4. **Macro-actions** — augment action space with "patrol-and-defend N nights" or "rest-until-fire" options that commit the agent for multiple turns. Reduces decision frequency from per-step to per-N-steps. The gradient at the macro-action level might be aligned with cooperation in ways the per-step gradient isn't.
5. **FeUdal Networks** — explicit manager/worker decomposition with manager seeing coarsened state.

### Auxiliary objectives / reward shaping

6. **Potential-based shaping with V(s) = team welfare** — fully aligned per Ng-Harada-Russell. Doesn't change optimal policy but accelerates learning. Could pre-compute potential function from heuristic specialist's value estimates.
7. **Intrinsic motivation for influence** (Jaques et al.) — agents rewarded for actions whose causal influence on others is high. Encodes "your actions matter to others" directly into the gradient.
8. **Multi-objective scalarization with episode return** — combine per-step advantage with episode return signal. Tunable mix.

### Algorithmic alternatives

9. **COMA** — counterfactual baseline; directly addresses credit assignment in cooperative settings. Probably the single biggest item not yet tried.
10. **LOLA** — opponent-aware gradient; specifically designed for iterated cooperation games. Niche but very on-point for our thesis.
11. **Off-policy MARL with replay buffer** — possibly less subject to the on-policy gradient bias.
12. **PBT with mutation** — escape the basin via exploration of policy diversity rather than gradient correction.

## Suggested next experiments (prioritized by leverage × cost)

Once #270/#271 land their verdicts:

| # | Experiment | Tests | Cost |
|---|---|---|---|
| 1 | High-λ GAE PPO smoke (`λ=0.99`) | Is the GAE bootstrap bias the misalignment source? | ~30 min |
| 2 | Potential-based team-welfare shaping | Does aligned shaping accelerate learning where non-aligned shaping (action #262) failed flat? | ~1 day |
| 3 | COMA implementation | Does counterfactual credit assignment do what MAPPO couldn't? | ~3 days |
| 4 | BC-init + PPO with high-λ | Combination: warm-start away from defection + correct local gradient | ~1 day |
| 5 | Macro-action wrapper | Does coarsening the decision frequency change which gradient direction wins? | ~3 days |
| 6 | LOLA | Direct test of opponent-aware gradient on iterated social dilemma | ~1 week |
| 7 | PBT with mutation | Population escape from defection basin | ~3 days |

## Sequencing

Don't run these until #270 PPO continuation result lands. The verdict on whether the BC-init basin holds determines which of the above is most relevant:

- **BC-init basin holds (basin trap)**: experiments #4 (BC-init + high-λ), #7 (PBT) become highest priority. The training procedure problem is mostly about *getting* to the cooperative basin.
- **BC-init basin collapses (anti-attractor)**: experiments #1, #2, #3, #6 become priority. The training procedure problem is about *staying* in cooperative basins; need to fix the gradient signal itself.

## Open questions for the paper

1. Does this misalignment story require the *multi-agent* framing, or does it appear in single-agent long-horizon cooperative tasks (e.g., MazeRunner-style)? If the latter, the result is much bigger.
2. Is the bucket-brigade Nash equilibrium of converged-PPO policies near random? (#275, blocked.) If yes, the result reframes from "PPO has a bug" to "PPO is rationally finding the Nash."
3. What's the precise relationship between the "misaligned gradient" claim and the existing specification-gaming literature? Is it a special case, a generalization, or a distinct phenomenon?
4. Could we construct a *minimal* example — a 2-agent, 2-action repeated game — that shows the same anti-attractor signature? Would make the result more legible.

## References to update later

- Add to project memory `project_ppo_failure_mode.md`
- Cross-link from `docs/ppo-help-brief.md` (the writeup we drafted for your friend) — the brief is now a working summary of the *empirical* case; this doc is the *theoretical* frame
- File the next-experiment issues once Phase 1 verdicts land
