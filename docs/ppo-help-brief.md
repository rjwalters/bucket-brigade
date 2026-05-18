# Help wanted: PPO won't learn this multi-agent game, and I've exhausted the obvious fixes

I'm running into a stubborn empirical plateau on a custom cooperative MARL game. **Five independent plausible PPO-side interventions have all returned tier-3 "insufficient" verdicts (<25% of the random→specialist gap closed)** against a hand-tuned specialist baseline. The diagnostic signals are inconsistent with the standard explanations. I'd love a second pair of eyes before I commit to a more invasive direction.

## The environment

**Bucket Brigade** — a cooperative multi-agent fire-fighting game I designed for studying cooperation/deception/coordination.

- **World:** 10 houses arranged in a circle. Each house is `SAFE | BURNING | RUINED`. Fires spread to neighbors probabilistically; new fires ignite randomly; burning houses become ruined if unextinguished.
- **Agents:** 4 agents, each owning one house. Episode length 12-13 nights.
- **Per-night action** (post-fix, MultiDiscrete `[10, 2, 2]`):
  - `house ∈ {0..9}` — which house to visit
  - `mode ∈ {work, rest}` — what to do there
  - `signal ∈ {work, rest}` — what to broadcast (can lie)
- **Rewards:**
  - Team term: +/- for houses saved/ruined
  - Per-agent ownership term: penalty if your own house burns, small reward if it survives
  - Per-agent work cost: penalty for working
- **Hand-tuned heuristic agents** as a specialist baseline (∼-22.07 per-step team reward on `minimal_specialization`; random play is ∼-87.72 to -96.07 depending on sampler/n).
- **Code:** Rust core via PyO3 + Python env wrapper. PufferLib for vectorized PPO. ~250-700ms per training iteration. We use the standard joint-trainer pattern (per-agent policy heads, can be combined with a centralized critic).

## The plateau

Across **every** non-trivial intervention we've tried, PPO converges to roughly the random baseline and stays there. The gap-closed metric — `(trained_reward - random) / (specialist - random)` — sits in `[0.07, 0.18]` across 5 substantive interventions × 3 seeds each.

| Intervention | gap_closed | Verdict |
|---|---:|---|
| Baseline IPPO | 0.075 - 0.19 | (the floor) |
| MAPPO / centralized critic | -0.083 (vs IPPO) | INSUFFICIENT |
| Per-agent observation differentiation (identity one-hot tail) | 0.182 | INSUFFICIENT |
| Spatial reward asymmetry (`distance_cost_alpha=0.1`) | 0.096 - 0.161 | INSUFFICIENT |
| Episode-length curriculum (5 → 8 → 12 nights) | 0.072 | INSUFFICIENT |
| Action-conditioned per-step reward shaping (α×β grid sweep) | 0.164 best | INSUFFICIENT |

Plus the pre-existing PPO hyperparameter sweeps (3 separate ablation studies on learning rate, batch size, value coef, entropy coef, normalize-returns, GAE λ, clip ε, hidden size, longer 500-iter horizon) which all came back negative before I started touching the env.

## The strange parts

These are the diagnostic readings that don't match the standard hypotheses:

### 1. Per-agent obs differentiation worked structurally but didn't lift reward

Pre-fix, all 4 agents received identical observations (they only differed via fixed positional encoding once we re-checked). I added a per-agent identity one-hot to the observation. **Pairwise action-distribution KL between agents grew to 2.7-3.7** (strongly differentiated policies). But the agents differentiated into *different patterns that are all about as good as random*. Reward stayed at the floor.

### 2. MAPPO collapses entropy 1874× without lifting reward

Standard MAPPO (centralized critic, decentralized actors) implementation. Default-off bit-identical to IPPO; with the centralized critic enabled, **per-agent action entropy collapses by 1874× on `default` and `positional_default` scenarios**, but the team reward stays at random. The shared critic apparently pulls policies toward each other, and the basin it pulls them into is *near* the random baseline but not at the specialist. MAPPO actually *marginally underperforms* IPPO on all 3 tested scenarios.

### 3. Action-conditioned per-step shaping was a flat grid

I added two per-step shaping knobs:
- α: credit-shared bonus for extinguishing a fire (`α / num_workers_at_house`)
- β: flat bonus for being at a SAFE→SAFE house while working (preventive presence)

Swept `α ∈ {0, 0.1, 0.5, 2.0}` × `β ∈ {0, 0.1, 0.5}` × 3 seeds (36 cells, 50 iters each). The H2 audit had flagged that **35% of steps have zero per-agent gradient** — action-shaping was supposed to fill exactly that gap.

Result: **the grid is flat. All 12 (α, β) cells land in `[0.090, 0.164]`.** Increasing the shaping intensity doesn't move things directionally. Crucially, **no over-shaping signature either** — max entropy collapse was only 3.4× (well under MAPPO's 1874× and our 100× flag). Shaping at these magnitudes isn't *too strong*; it's *not strong enough to matter*.

### 4. The signaling channel was a documented-but-not-implemented mechanic

This one's a "gotcha." The game's docs claim agents can lie (broadcast Work while planning to Rest, etc.). I discovered the engine was deterministically copying the work/rest action bit into the signal — agents could not lie at all. I fixed this (made signal a real 3rd action dimension), verified the Liar archetype with `honesty_bias=0.1` now lies ~90% of actions. **PPO learning is unchanged.** Obs-fix gap_closed went from 0.059 → 0.182 (3× improvement, statistical not behavioral), still below the 25% bar.

### 5. Behavioral signature

Trained policies on `minimal_specialization`:
- Mean trailing-5 team reward: -87 to -98 per step
- Random reference: -92.92 (n=50 sampler) or -87.72 (n=1000 sampler)
- Specialist reference: -22.07
- Per-agent action entropies stable, not collapsed (in IPPO; collapsed in MAPPO)

So PPO settles near random and stays there, with healthy entropy (in IPPO at least). It's not over-fitting, not collapsing to a degenerate single-action policy, not diverging. It just doesn't move toward the specialist's region of policy space.

## What I've ruled out

- **PPO hyperparameter regime** — three separate ablation studies, all negative
- **Random baseline being wrong** — re-derived at n=1000 × 5 seeds on the current substrate; specialist re-derived for invariance under the signal-channel fix
- **Code correctness** — all 5 interventions have regression tests, bit-identity preservation when disabled, and the H1/H2/H3 diagnostic suite verifies invariants
- **Pre-#236 substrate quirks** — the signal channel bug fix, the redundant-obs bug, the heuristic action mask bug — all found and fixed; PPO verdict is unchanged

## What I'm wondering

A few hypotheses I haven't tested:

1. **Behavioral cloning from the specialist** as PPO initialization. If the basin of attraction around random is large and the specialist basin is far, PPO's local-gradient updates may never find it from random init.
2. **Population-based training** to escape via mutation (could be combined with neuroevolution starting from heuristic archetypes).
3. **QMIX / VDN-style value decomposition** as opposed to MAPPO's centralized critic. The "averaging across agents" failure mode of MAPPO suggests value-decomposition might behave differently.
4. **A dense progress signal** (intervention #4 on my list, untried): reward each agent every step proportional to ΔHouses-still-safe. This is *team-shaping* not *action-shaping* — different attack on credit assignment.
5. **Reframing the research question.** The game is my own invention. The current rules may simply not be PPO-learnable from random init. I'm willing to redesign mechanics (action validity constraints, sequential within-night commitment, continuous extinguish probabilities) if that's what it takes — but I'd rather understand *why* PPO fails before redesigning.

## Specific questions

1. **Have you seen this exact pattern before** — every plausible intervention returning tier-3, no over-shaping signatures, healthy entropy in IPPO, structurally-correct differentiation that doesn't translate to reward — in cooperative MARL literature? Is there a name for this failure mode?

2. **Is there a known diagnostic that distinguishes** "PPO is locally trapped in a bad basin" from "PPO is sample-inefficient on this game" from "the game has no PPO-reachable specialist equivalent from random init"? I have an H1/H2/H3 diagnostic suite but it's measuring the *symptoms*, not the underlying cause.

3. **Would a BC-init experiment** (clone the specialist for 1k steps, then PPO from there) be informative? My intuition is "if BC-init then PPO holds, we've confirmed the basin is unreachable from random; if BC-init then PPO collapses back to random, we've confirmed something is actively pulling policies toward random."

4. **Is the MAPPO entropy collapse signature** (1874× while reward stays at random) consistent with any specific pathology you'd recognize? My read is "shared critic finds a value-function minimum where all policies agree but the agreement isn't valuable" — does that map to known failure modes?

5. **Any reason to prefer COMA or QMIX** over what I've already tried? I want to spend the next attempt wisely.

Happy to share code, run further diagnostics, or share the full notebook of failed attempts.

---

*Context for the reader: this is a research log on a custom env, not a benchmark. PPO/PufferLib/Rust core all are mainstream stacks; the env design is the part I'd be most willing to mutate if you think the game itself is the problem.*
