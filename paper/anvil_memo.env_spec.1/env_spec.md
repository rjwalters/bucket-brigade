---
title: "Bucket Brigade — formal environment specification"
recipient: "Bucket Brigade workshop paper readership"
artifact_type: "descriptive-thesis"
version: 1
date: 2026-06-05
issue: rjwalters/bucket-brigade#362
tracker: rjwalters/bucket-brigade#357
---

# Executive summary

This document fixes the formal specification of the **Bucket Brigade** environment for §2 of the workshop paper. Bucket Brigade is a finite, discrete-time, partially observable stochastic game on a ring of houses played by a fixed population of agents. Each night every agent simultaneously chooses a house to attend, a work/rest mode, and a one-bit broadcast signal; fires then have a chance to be extinguished, burn out, spread to neighbors, and ignite spontaneously, in that fixed order; the night ends with reward distribution and a termination check. The parameter family $(\beta, \kappa, \rho, c, H, N, T_{\min})$ together with the reward weight tuple fully determines the game.

Two design decisions distinguish Bucket Brigade from the canonical cooperative-MARL benchmarks (Overcooked, Hanabi, SMAC, Melting Pot, MAgent, PettingZoo MPE; see the comparison report at `paper/anvil_report.benchmark_comparison.1/`). First, the state and joint-action spaces are **small enough to enumerate** at minimal parameterizations (2304 states and $8^N$ joint actions at $H{=}2,N{=}4$), so Nash equilibria of the one-shot stage game can be computed exactly rather than approximated. Second, the dynamics are parameterized by three load-bearing scalars $(\beta,\kappa,c)$ that change equilibrium structure continuously, so the game family interpolates smoothly between cooperative-dominant and free-rider-dominant regimes. The specification below is **self-contained**: a reader unfamiliar with the codebase should be able to reimplement the environment from this document alone.

The notation matches the rest of the paper. $\beta$ is the per-night, per-neighbor probability that fire spreads from a burning house to an adjacent SAFE house (load-bearing only in the contagion-active phase-order variant; see §3.4). $\kappa$ is the probability that a single agent extinguishes a fire in one night. $\rho$ is the per-night, per-SAFE-house probability of spontaneous ignition. $c$ is the per-night work cost. $H$ is the number of houses on the ring; $N$ is the number of agents. The relationship to canonical game-theory templates — Volunteer's Dilemma at $N{=}1$, $N$-player Public Goods over time, Stag Hunt in the limit $\kappa \to 1$ — is made precise in §6.

# 1. Players and parameters

## 1.1 Player set

A finite set of agents $\mathcal{I} = \{1, 2, \ldots, N\}$, with $N \geq 1$ fixed across an episode. Agents are **positionally symmetric**: the environment dynamics are invariant under any permutation $\sigma : \mathcal{I} \to \mathcal{I}$ applied jointly to every per-agent quantity (locations, signals, actions, rewards). The asymmetric reward variants of §5.4 (per-agent home-position cost, per-agent ownership weights) break this symmetry only when their dedicated parameters are nonzero; the symmetric default game preserves exchangeability.

## 1.2 Topology

The world is a **cycle graph** $\mathcal{H} = \mathbb{Z}/H\mathbb{Z}$ with $H \geq 2$ vertices ("houses") and edge set $\{(h, h{+}1 \bmod H) : h \in \mathcal{H}\}$. House indices are taken mod $H$ throughout. The two neighbors of house $h$ are $h{-}1 \bmod H$ and $h{+}1 \bmod H$.

## 1.3 Parameter family

The full parameter family is $\theta = (\beta, \kappa, \rho, c, H, N, T_{\min}, W)$ where:

| Symbol | Domain | Meaning |
|---|---|---|
| $\beta$ | $[0, 1]$ | Per-night, per-neighbor fire-spread probability |
| $\kappa$ | $[0, 1]$ | Per-agent solo extinguish probability |
| $\rho$ | $[0, 1]$ | Per-night, per-SAFE-house spontaneous ignition probability |
| $c$ | $\mathbb{R}_{\geq 0}$ | Per-night work cost |
| $H$ | $\mathbb{Z}_{\geq 2}$ | Number of houses on the ring |
| $N$ | $\mathbb{Z}_{\geq 1}$ | Number of agents |
| $T_{\min}$ | $\mathbb{Z}_{\geq 0}$ | Minimum nights before termination is allowed |
| $W$ | reward tuple (§4) | Reward weights (team and per-agent) |

The default scenario referenced as the "10-house default" in the paper uses $H{=}10, N{=}4, \beta{=}0.25, \kappa{=}0.5, \rho{=}0.02, c{=}0.5, T_{\min}{=}12$ and reward weights $W = (R_{\text{team}}, P_{\text{team}}, r_{\text{own}}, r_{\text{other}}, p_{\text{own}}, p_{\text{other}}) = (100, 100, 20, 0, 40, 0)$. The minimal-parameterization scenario sets $H{=}2, N{=}4$.

# 2. State, action, observation

## 2.1 State

The environment state at the start of night $t$ is

$$s_t = (h_t, \ell_t, \zeta_t, t) \in \{S, B, R\}^H \times \mathcal{H}^N \times \{0, 1\}^N \times \mathbb{N}$$

where $h_t \in \{S, B, R\}^H$ is the vector of house statuses (S = SAFE, B = BURNING, R = RUINED), $\ell_t \in \mathcal{H}^N$ is the vector of agent locations, $\zeta_t \in \{0, 1\}^N$ is the vector of one-bit broadcast signals emitted on the previous night (REST = 0, WORK = 1), and $t \in \mathbb{N}$ is the night counter.

The RUINED status is **absorbing**: once $h_{t,k} = R$ the house remains $R$ for the remainder of the episode. SAFE and BURNING are mutually reachable in either direction.

**Initial state.** At $t{=}0$ each house is independently BURNING with probability $\rho$ and SAFE otherwise; agent locations are $\ell_{0,i} = 0$ and signals $\zeta_{0,i} = 0$ for all $i$.

**State cardinality.** $|S^H \times \mathcal{H}^N \times \{0,1\}^N| = 3^H \cdot H^N \cdot 2^N$. For $H{=}2,N{=}4$: $9 \cdot 16 \cdot 16 = 2304$ states. For $H{=}10,N{=}4$: $\approx 9.44 \times 10^{9}$ — large, but polynomial in $(H, N)$ rather than exponential in observation pixels.

## 2.2 Action

Each agent $i$ chooses an action $a_{t,i} = (a_{t,i}^{\text{house}}, a_{t,i}^{\text{mode}}, a_{t,i}^{\text{sig}})$ from the product set

$$\mathcal{A} = \mathcal{H} \times \{0, 1\} \times \{0, 1\}$$

with $a^{\text{house}} \in \mathcal{H}$ the target house index, $a^{\text{mode}} \in \{0, 1\}$ the WORK bit ($1$ = WORK, $0$ = REST), and $a^{\text{sig}} \in \{0, 1\}$ the broadcast bit emitted *this* night and observable to all agents on the *next* night. Per-agent cardinality is $|\mathcal{A}| = 2H \cdot 2 = 4H$ (so $|\mathcal{A}| = 8$ at $H{=}2$ and $|\mathcal{A}| = 40$ at $H{=}10$).

Joint action $a_t = (a_{t,1}, \ldots, a_{t,N}) \in \mathcal{A}^N$. Joint action cardinality $|\mathcal{A}^N| = (4H)^N$: $4096$ at $H{=}2,N{=}4$, $\approx 2.56 \times 10^{6}$ at $H{=}10,N{=}4$.

The signal bit is **strictly informational** and incurs no cost — it does not constrain the agent's mode or location and other agents cannot reject it. Signals support cheap-talk equilibria; their interpretation is not part of the game specification.

## 2.3 Observation

The environment is **perfect-monitoring with one-step delay on signals**: at the start of night $t$, every agent observes the same tuple

$$o_t = (h_t, \ell_{t-1}, \zeta_{t-1}, \alpha_{t-1}, t)$$

where $\alpha_{t-1} \in \mathcal{H}^N \times \{0,1\}^N$ is the previous night's joint $(\text{house}, \text{mode})$ slice. There is **no partial observability of houses or agent positions**. The observation is identical for every agent (no private observations).

This is the simultaneous-move variant. The optional two-phase variant (§5.5) inserts a second observation channel for round-1 signals before round-2 actions.

# 3. Transition dynamics

A single night consists of seven sequential phases applied to the state, using a sequence of independent uniform random draws $U_1, U_2, \ldots$. The phase order is **load-bearing** — swapping any two adjacent phases changes the transition kernel. The order is: (1) signal write, (2) location and mode write, (3) extinguish, (4) burn-out, (5) spread, (6) spontaneous ignition, (7) reward and termination check.

For notational economy, write $\mathbb{1}[\,\cdot\,]$ for the indicator and $w_h(a) := |\{i : a_i^{\text{house}} = h \land a_i^{\text{mode}} = 1\}|$ for the number of agents who WORK at house $h$ under joint action $a$.

## 3.1 Signal and bookkeeping (phases 1–2)

Update bookkeeping: $\zeta_t \leftarrow (a_{t,i}^{\text{sig}})_{i \in \mathcal{I}}$, $\ell_t \leftarrow (a_{t,i}^{\text{house}})_{i \in \mathcal{I}}$. These updates do not alter house states.

## 3.2 Extinguish phase (phase 3)

For each house $h$ with $h_{t,h} = B$, draw a single uniform $U \sim \text{Uniform}[0,1]$ and update:

$$h_{t,h}^{(3)} = \begin{cases} S & \text{if } U < 1 - (1 - \kappa)^{w_h(a_t)} \\ B & \text{otherwise} \end{cases}$$

The probability that $w$ workers extinguish a fire jointly is the **independent-workers** model: each worker independently succeeds with probability $\kappa$, so the probability that at least one succeeds is $1 - (1-\kappa)^w$. SAFE and RUINED houses are unaffected.

## 3.3 Burn-out phase (phase 4)

Every house that survived the extinguish phase as BURNING transitions to RUINED:

$$h_{t,h}^{(4)} = \begin{cases} R & \text{if } h_{t,h}^{(3)} = B \\ h_{t,h}^{(3)} & \text{otherwise} \end{cases}$$

This is **deterministic** conditional on the extinguish phase: any fire not extinguished in the same night ruins the house. The burn-out phase encodes the cost of inattention.

## 3.4 Spread phase (phase 5)

For each pair $(h, h')$ such that $h_{t,h}^{(4)} = B$ and $h' \in \{h{-}1, h{+}1\} \bmod H$ and $h_{t,h'}^{(4)} = S$, draw independent $U \sim \text{Uniform}[0,1]$ and set $h_{t,h'}^{(5)} = B$ if $U < \beta$.

**Observation on phase composition.** Because the burn-out phase 4 transitions every unextinguished BURNING house to RUINED, the spread phase as written above finds no BURNING source houses under the canonical (binary-extinguish) dynamics — phase 5 is effectively inert in the canonical sequence, and the only path from SAFE to BURNING during a night is the spontaneous-ignition phase 6. This is the literal behavior of the specification as written. A reader who wants neighbor-driven contagion to *fire* under the canonical mode should swap phases 4 and 5 (perform spread before burn-out so that unextinguished fires can ignite neighbors before themselves ruining), at which point:

$$\Pr[h_{t,h'}^{(5)} = B \mid h_{t,h'}^{(3)} = S, h_{t,h}^{(3)} = B, h' \sim h] = \beta$$

with draws mutually independent across $(h, h')$ pairs and a SAFE house with two BURNING neighbors having combined ignition probability $1 - (1-\beta)^2$. The canonical paper-§2 specification follows the implementation order (extinguish → burn-out → spread → ignition); the contagion-active variant is the natural alternative cited by readers familiar with epidemic-process MARL benchmarks. The phase-diagram analysis at issue #358 is conducted against the canonical order.

## 3.5 Spontaneous ignition phase (phase 6)

For each house $h$ with $h_{t,h}^{(5)} = S$, draw independent $U \sim \text{Uniform}[0,1]$ and set $h_{t,h}^{(6)} = B$ if $U < \rho$. RUINED houses are unaffected; freshly-ruined houses from phase 4 cannot re-ignite. The resulting state is $h_{t+1} := h_t^{(6)}$.

# 4. Reward structure

Rewards are paid at the end of each night, after all transition phases. The per-agent reward $r_{t,i}$ decomposes into three terms — a private work/rest cost, a per-house ownership term, and a team-welfare term — each parameterized independently.

## 4.1 Work/rest term

For each agent $i$,

$$r_{t,i}^{\text{work}} = \begin{cases} -c & \text{if } a_{t,i}^{\text{mode}} = 1 \text{ (WORK)} \\ +c_{\text{rest}} & \text{if } a_{t,i}^{\text{mode}} = 0 \text{ (REST)} \end{cases}$$

with $c_{\text{rest}} = 0.5$ as the implementation default. The asymmetry between $c$ and $c_{\text{rest}}$ is intentional: REST is a positive payoff that opt-in WORK pays to forgo. The free-rider gradient comes from this asymmetry.

## 4.2 Per-house ownership term

Each house $h$ has a designated owner $\text{owner}(h) \in \mathcal{I}$ assigned at episode start by round-robin: $\text{owner}(h) = h \bmod N$. The per-house contribution to agent $i$'s reward is

$$r_{t,i,h}^{\text{own}} = \mathbb{1}[h_{t-1,h} \neq S \land h_{t+1,h} = S] \cdot w^{\text{save}}_i(h) - \mathbb{1}[h_{t+1,h} = R] \cdot w^{\text{ruin}}_i(h)$$

with weights

$$w^{\text{save}}_i(h) = \begin{cases} r_{\text{own},i} & \text{if } i = \text{owner}(h) \\ r_{\text{other},i} & \text{otherwise} \end{cases}, \qquad w^{\text{ruin}}_i(h) = \begin{cases} p_{\text{own},i} & \text{if } i = \text{owner}(h) \\ p_{\text{other},i} & \text{otherwise} \end{cases}$$

The save event fires once on the night the house transitions from non-SAFE to SAFE. The ruin penalty fires **every night** the house remains in state $R$, so a ruined house imposes a recurring cost until the episode terminates.

## 4.3 Team-welfare term

Let $n_S(t) := |\{h : h_{t+1,h} = S\}|$ and $n_R(t) := |\{h : h_{t+1,h} = R\}|$. The team term, paid to every agent equally, is

$$r_{t}^{\text{team}} = R_{\text{team}} \cdot \frac{n_S(t)}{H} - P_{\text{team}} \cdot \frac{n_R(t)}{H}$$

## 4.4 Total reward

$$r_{t,i} = r_{t,i}^{\text{work}} + \sum_{h \in \mathcal{H}} r_{t,i,h}^{\text{own}} + r_t^{\text{team}}$$

The reward is bounded: $|r_{t,i}| \leq c + \max(c_{\text{rest}}, 0) + H \cdot \max(|w^{\text{save}}|, |w^{\text{ruin}}|) + (R_{\text{team}} + P_{\text{team}})$.

# 5. Game length, information structure, symmetries

## 5.1 Horizon and termination

The episode terminates at night $T$ when $t \geq T_{\min}$ **and** at least one of the following holds:

- All houses are SAFE: $\forall h, h_{t+1,h} = S$.
- All houses are RUINED: $\forall h, h_{t+1,h} = R$.
- No house is BURNING: $\forall h, h_{t+1,h} \neq B$.

Under canonical parameters with $\rho > 0$ the "no-fires-burning" condition holds only transiently because spontaneous ignition reintroduces fires, so termination typically occurs via the absorbing all-RUINED condition or via a transient SAFE state that survives the spontaneous-ignition draw. There is no hard upper bound on $T$; the expected episode length is finite for $\rho > 0$ but the game is not strictly finite-horizon.

The minimum-nights clamp $T_{\min}$ prevents pathological zero-length episodes when the initial draw happens to leave all houses SAFE.

## 5.2 Information structure

The information structure is **symmetric perfect monitoring**: every agent observes the full state $h_t$, the previous joint location/mode/signal $(\ell_{t-1}, \alpha_{t-1}^{\text{mode}}, \zeta_{t-1})$, and the night counter $t$. No agent has private information about house states or other agents' actions. The only private information is each agent's own *policy* (and any internal randomness it employs).

The signal channel is non-binding cheap talk: $\zeta_{t,i}$ is observable to every agent at night $t{+}1$ but is functionally decoupled from $a_{t+1,i}^{\text{mode}}$ — an agent may signal WORK and play REST, or vice versa.

## 5.3 Symmetry properties

The default game enjoys two symmetries.

**Agent exchangeability.** Let $\sigma : \mathcal{I} \to \mathcal{I}$ be a permutation. Apply $\sigma$ jointly to the agent index of every per-agent quantity ($\ell_t, \zeta_t, a_t, r_t$). The transition kernel and reward function are invariant under $\sigma$. Consequence: symmetric Nash equilibria — strategy profiles where all agents play the same (possibly mixed) policy — exist for every parameter cell. Whether the symmetric NE is the *unique* equilibrium is a per-parameter-cell question (see the comparison report on the rest-trap cell, where the only NE is asymmetric).

**Ring rotation.** Let $\tau : \mathcal{H} \to \mathcal{H}$ be a rotation $\tau(h) = h + k \bmod H$. The transition kernel is invariant under $\tau$ applied jointly to $h_t, \ell_t, a_t^{\text{house}}$. Reward invariance under $\tau$ holds only when reward weights are not per-house-specific (i.e., when ownership weights are permuted consistently with $\tau$).

Asymmetric reward variants (§5.4) break agent exchangeability; the position-constrained variant (§5.5) breaks ring rotation by anchoring each agent to a home position. The symmetric default game preserves both.

## 5.4 Asymmetric reward variants

The implementation exposes per-agent reward weight vectors so that $r_{\text{own}}, r_{\text{other}}, p_{\text{own}}, p_{\text{other}}$ can each be set per-agent rather than uniformly. When per-agent vectors are nonuniform, agent exchangeability is broken; symmetric equilibrium analysis no longer applies and the game admits genuinely asymmetric NE.

## 5.5 Optional dynamic variants

The base specification of §3 is the canonical Bucket Brigade. Four optional variants are also part of the calibrated parameter family, each gated by a parameter that defaults to the no-op value:

- **Adjacent-only action validity**: a per-agent home position $h^{\text{home}}_i$ and a constraint that $a_i^{\text{house}}$ must satisfy ring-distance $\leq 1$ from $h^{\text{home}}_i$. Out-of-reach targets are remapped to $h^{\text{home}}_i$ before phase 1.
- **Continuous-extinguish variant**: a damage-accumulation alternative to §3.2. Each WORK at a BURNING house adds a fixed increment $\sigma$ to a per-house accumulator; the fire transitions to SAFE deterministically when the accumulator reaches 1. Fires do not burn out in this variant.
- **Two-phase signaling**: each night becomes two micro-rounds. Round-1 emits a signal only; round-2 observes everyone's round-1 signal in an added observation channel and emits a full action. Round-2 mode is unconstrained by the round-1 signal, preserving the deception channel.
- **Distance-cost asymmetry**: the work cost is augmented to $c + \alpha \cdot d(h^{\text{home}}_i, a_i^{\text{house}})$ where $d$ is ring-arc distance, breaking ring-rotation symmetry.

These variants are out of scope for the canonical paper but are part of the calibrated parameter family the broader research program sweeps over.

# 6. Relationship to canonical game-theory templates

Bucket Brigade is a parametric family that recovers, as limiting cases or restrictions, several canonical templates from the cooperative-game-theory literature. The relationships below are exact at the named limits and approximate elsewhere.

**Volunteer's Dilemma** (Diekmann 1985). At $N{=}1, H{=}1$ with $\rho > 0, \beta = 0$, each night the single agent faces a binary choice: WORK (pay $c$, save the house with probability $\kappa$) or REST (gain $c_{\text{rest}}$, lose the house with probability 1 in the next phase). This is the textbook single-shot volunteer's dilemma with payoff parameters $(c, c_{\text{rest}}, \kappa, P_{\text{team}})$. The multi-agent generalization at $N > 1, H = 1$ is the **$N$-player volunteer's dilemma**: the team needs at least one volunteer per fire, and the free-rider incentive is exactly the $c_{\text{rest}} - (-c) = c_{\text{rest}} + c$ asymmetry between REST and WORK.

**$N$-Player Public Goods Game** (Hardin 1968; Ostrom 1990). With $r_{\text{own}} = r_{\text{other}}$ and $p_{\text{own}} = p_{\text{other}}$ — i.e., ownership weights collapsed to uniform — the per-house ownership term reduces to a non-excludable common good, and the team term $r_t^{\text{team}}$ is paid equally regardless of contribution. The marginal private return to WORK is then $\kappa \cdot (R_{\text{team}} + P_{\text{team}})/H - c$, which may be strictly less than the social marginal return $\kappa \cdot (R_{\text{team}} + P_{\text{team}}) - c$, producing the textbook public-goods underinvestment gradient.

**Stag Hunt** (Skyrms 2004). In the limit $\kappa \to 1$ a single worker is sufficient to extinguish any fire, but the work cost $c$ remains. The game then has two natural NE: the "all-stag" cooperative NE where every agent works coordinated houses (high payoff, requires belief in coordination) and the "all-hare" lazy NE where everyone REST-defects (low payoff, no coordination required). The intermediate-$\kappa$ regime interpolates between Stag Hunt and the public-goods regime.

**Free-rider problem and the rest-trap cell.** At specific parameter cells (the "rest-trap" cell explored under issue #355) the unique NE is asymmetric: $N{-}1$ agents play REST and one agent plays full-force WORK. This is the textbook free-rider problem cast in the bucket-brigade vocabulary: every agent prefers another to bear the work cost, and the only equilibrium that avoids the all-REST collapse is the asymmetric one. The phase diagram across $(\beta, \kappa, c)$ shows the boundary between symmetric-NE-dominant and asymmetric-NE-dominant regimes.

**Stochastic Game** (Shapley 1953). Stripped to its bones, Bucket Brigade is a finite-state, finite-action, discounted stochastic game with $N$ players, symmetric perfect monitoring, and reward decomposition into private and team components. Every formal result for finite stochastic games — minimax, equilibrium existence in stationary mixed strategies, value iteration convergence — applies directly. The contribution of the Bucket Brigade design is not in the game-theoretic class; it is in the **size**. At the minimal parameterization the game is small enough that the standard tools actually compute, which is the property the paper exploits.

# 7. Notation summary

| Symbol | Meaning |
|---|---|
| $\mathcal{I} = \{1, \ldots, N\}$ | Agent set; $N$ = population size |
| $\mathcal{H} = \mathbb{Z}/H\mathbb{Z}$ | House ring; $H$ = number of houses |
| $S, B, R$ | House statuses: SAFE, BURNING, RUINED (RUINED is absorbing) |
| $\beta$ | Per-night, per-neighbor fire-spread probability |
| $\kappa$ | Solo-agent extinguish probability (independent-workers model) |
| $\rho$ | Per-night, per-SAFE-house spontaneous ignition probability |
| $c$ | Per-night work cost (paid when $a_i^{\text{mode}} = 1$) |
| $c_{\text{rest}}$ | Per-night rest payoff (implementation default 0.5) |
| $T_{\min}$ | Minimum nights before termination is allowed |
| $a_{t,i} = (a^{\text{house}}, a^{\text{mode}}, a^{\text{sig}})$ | Per-agent action: target house, work bit, signal bit |
| $w_h(a)$ | Number of WORKers at house $h$ under joint action $a$ |
| $R_{\text{team}}, P_{\text{team}}$ | Team reward / penalty per fraction-safe / fraction-ruined |
| $r_{\text{own}}, r_{\text{other}}$ | Per-agent save-reward weights (own / other-owned house) |
| $p_{\text{own}}, p_{\text{other}}$ | Per-agent ruin-penalty weights (own / other-owned house) |
| $n_S(t), n_R(t)$ | Counts of SAFE / RUINED houses at end of night $t$ |
| $\zeta_t \in \{0,1\}^N$ | Vector of broadcast signals (observable next night) |
| $\ell_t \in \mathcal{H}^N$ | Vector of agent locations after phase 2 |

The notation above is the canonical vocabulary for §2 of the paper; every downstream section ($§3$ equilibrium analysis, $§4$ experiments, $§5$ discussion) reuses it without redefinition.

# 8. Reproducibility note

This specification is the **mathematical contract** for Bucket Brigade. A reader who reimplements from this document alone — using any RNG and any programming language — should obtain matching equilibrium structure and policy-evaluation results at the documented parameter cells. (For pointers to the in-repo reference implementations and scenario declarations, see `refs/bucket-brigade-env.md`.)

The seven-phase ordering of §3 is load-bearing: implementers must apply phases in the exact order (signal write, location/mode write, extinguish, burn-out, spread, spontaneous ignition, reward). The independent-workers extinguish formula of §3.2 and the post-extinguish-pre-burn-out spread test of §3.4 are the two details most often misread on first encounter; both are explicit above to make reimplementation tractable.
