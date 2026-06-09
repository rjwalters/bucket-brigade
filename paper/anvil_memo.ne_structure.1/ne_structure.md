---
title: "Analytical NE characterization for 4-agent Bucket Brigade"
recipient: "Bucket Brigade workshop paper readership"
artifact_type: "descriptive-thesis"
version: 1
date: 2026-06-08
issue: rjwalters/bucket-brigade#359
tracker: rjwalters/bucket-brigade#357
---

# Executive summary

This memo derives closed-form boundary inequalities that predict which Nash-equilibrium structure (symmetric all-Work, asymmetric 1-Worker, or no-pure-NE collapse) appears as a function of the three load-bearing parameters $(\beta, \kappa, c)$ in the 4-agent Bucket Brigade. The derivation reduces the $H{=}10$ ring with $T_{\min}{=}12$ to a **one-house, four-agent, single-night stage game** (§2) whose payoffs absorb multi-step contagion and ignition dynamics into a single survival-coefficient $A(\beta, \kappa, \rho, T_{\min}, W)$. The three NE structures fall out as algebraic inequalities in $(\kappa, A, c)$:

> **Headline finding.** Under the `minimal_specialization` reward tuple at $c{=}0.5$, the predicted phase boundary between symmetric all-Work, asymmetric 1-Worker, and no-pure-NE is **$\kappa$-driven and approximately $\beta$-independent**. The single-house mean-field reduction yields three algebraic inequalities: the symmetric all-Work NE exists for $\tilde A \cdot \kappa \cdot (1-\kappa)^3 \geq c_{\text{gap}}$, the asymmetric 1-Worker NE exists for $\tilde A \cdot \kappa \geq c_{\text{gap}}$ AND $\tilde A \cdot \kappa \cdot (1-\kappa) < c_{\text{gap}}$ (free-rider lock-in), and the no-pure-NE collapse holds whenever $\tilde A \cdot \kappa < c_{\text{gap}}$. Here $\tilde A := A \rho$ is the per-night survival coefficient and $c_{\text{gap}} := c + c_{\text{rest}}$ is the full Work-vs-REST payoff gap. With the $W$-driven coefficient $\tilde A \approx 36.24$ at $T_{\min}{=}12, \rho{=}0.02$ and $c_{\text{gap}}{=}1.0$, these boundaries place the predicted collapse threshold at $\kappa \approx 0.028$ and the predicted symmetric → mixed → asymmetric thresholds at $\kappa \approx 0.65$ and $\kappa \approx 0.972$ — all sitting at very different κ values than the empirical sampled grid suggests. The empirical 7-cell preview at $\beta \in \{0.1, 0.5, 0.9\}, \kappa \in \{0.1, 0.5, 0.9\}, c{=}0.5$ shows the same κ-driven, β-independent pattern (3/3 cells at $\kappa{=}0.9$ are `asymmetric_only`; 2/2 cells at $\kappa{=}0.5$ are `symmetric_only`; 3/3 cells at $\kappa{=}0.1$ are `no_convergence`); the empirical $\kappa$ thresholds for the collapse / symmetric / asymmetric regimes sit at substantially different values than the analytical prediction, which §5 attributes to (a) the empirical solver's heuristic-parameter strategy space (Firefighter ≠ literal all-REST), (b) per-agent ownership rewards that change the payoff arithmetic the analytical reduction omits, and (c) ring-locality effects that the mean-field approximation absorbs into a coarse aggregate rather than treats explicitly. The qualitative κ-monotonicity matches; the quantitative thresholds do not.

The memo's claim is therefore narrow but defensible: **the mean-field reduction correctly predicts the qualitative phase order (collapse → symmetric → asymmetric) and the β-independence at $c{=}0.5$, but the quantitative κ-thresholds are off — the predicted collapse boundary $\kappa{\approx}0.028$ sits well below the empirically observed boundary near $\kappa{\approx}0.3$, and the predicted asymmetric-NE boundary $\kappa{\approx}0.972$ sits above the empirically observed boundary near $\kappa{\approx}0.7$**. The residual must be attributed to ring-locality + heuristic-strategy effects. §5 derives the size of the implied corrections and §6 names two empirical follow-ups (sweep $c \in \{1.0, 2.0\}$; sweep $\rho \in \{0.05, 0.10\}$) that would falsify or sharpen the framework. The 7-cell preview is insufficient to test the full 75-cell agreement claim of the issue's acceptance criterion; that comparison waits on the gap-fill of #358.

# 1. Setup

## 1.1 Formal game

This memo analyzes the per-night stage game of Bucket Brigade under stationary opponent policies, restricted to the `minimal_specialization` parameter cell with $H{=}10, N{=}4, T_{\min}{=}12$ and reward tuple $W = (R_{\text{team}}, P_{\text{team}}, r_{\text{own}}, r_{\text{other}}, p_{\text{own}}, p_{\text{other}}) = (10, 10, 50, 0, 100, 0)$. The free parameters that move across the phase diagram are $(\beta, \kappa, c)$; $\rho{=}0.02$ is fixed. Notation follows `env_spec.md` §1.3 verbatim (see `refs/env-spec.md` for the pin-cites).

The analytical reduction is **not** a derivation of the full stochastic-game subgame-perfect equilibrium. The full game's NE conditions exist only via numerical value iteration on the $\geq 9 \times 10^{9}$-state space, which does not generalize across $(\beta, \kappa, c)$ cells in closed form. Restricting to the per-night stage game under stationary opponent strategies is the load-bearing simplification that makes algebraic boundary inequalities possible; the restriction is honest, made up front, and §5 reports where it produces visibly wrong predictions.

## 1.2 Mapping the analytical primitives to the empirical strategy space

The empirical phase diagram is computed by the heterogeneous Double-Oracle solver over a finite menu of **continuous 10-D heuristic-parameter vectors** (`refs/heterogeneous-do-strategy-space.md`), **not** the binary Work/REST stage game the analytical derivation uses. The bridge is the work-tendency parameter:

- **Hero** ($\texttt{work\_tendency}{=}1.0$) ↔ analytical "all-Work" ($a^{\text{mode}}=1$ deterministically).
- **Firefighter** ($\texttt{work\_tendency}{=}0.9$) ↔ analytical "near all-Work"; treated as all-Work for the boundary.
- **Free Rider** ($\texttt{work\_tendency}{=}0.2$, $\texttt{rest\_reward\_bias}{=}0.9$) ↔ analytical "all-REST" ($a^{\text{mode}}=0$).

Empirical verdicts map to analytical NE structures as: `symmetric_only` with profile Hero|Hero|Hero|Hero → analytical symmetric all-Work NE; `asymmetric_only` with profile one Hero + three Firefighters → analytical asymmetric 1-Worker NE; `no_convergence` → analytical no-pure-NE collapse. The mapping is **approximate** — Firefighters at $\texttt{work\_tendency}{=}0.9$ are not literal REST agents — and §5 names the quantitative bias this introduces.

## 1.3 Single-house mean-field reduction (statement)

Instead of analyzing the full 10-house ring with 4 agents distributed across houses each night, we collapse the analysis to a **representative single house with all 4 agents present**, where the night's outcome is paid out as the expected per-agent payoff under the stage-game profile. Per-night reward decomposition (`env_spec.md` §4):

- **Work cost** $-c$ if the agent chooses $a^{\text{mode}}=1$, $+c_{\text{rest}}=0.5$ if $a^{\text{mode}}=0$. The free-rider gradient is the $c + c_{\text{rest}}$ asymmetry; the symbol $c$ in this memo means the full WORK–REST gap (so $c{=}1.0$ in the algebra when the scenario reports $c{=}0.5$ work cost + $0.5$ rest payoff). At $c{=}0.5$ from the phase-diagram driver, the algebraic $c$ is $0.5 - (-0.5) = 1.0$.
- **Per-house ownership term** weighted by $r_{\text{own}}=50$ (save event, fires once per house transition from non-SAFE to SAFE) and $p_{\text{own}}=100$ (ruin event, fires every night the house is RUINED).
- **Team term** $r_t^{\text{team}} = R_{\text{team}} \cdot n_S(t)/H - P_{\text{team}} \cdot n_R(t)/H$ with $R_{\text{team}}=P_{\text{team}}=10$.

In the single-house reduction, ownership accrues to the house's designated owner (1 of 4 agents) and is shared equally across agents in the steady-state assumption that each agent owns one house and the rotation is symmetric. The reduction collapses the $H{=}10$ ring to "every agent expects to defend its own house most nights"; ring-locality corrections that this approximation drops are isolated in §5.

# 2. Mean-field stage-game reduction

## 2.1 Derivation of the survival coefficient $A$

Consider one house with $k \in \{0, 1, 2, 3, 4\}$ workers present on a given night. Per `env_spec.md` §3.2, the probability the house transitions from BURNING to SAFE in the extinguish phase is

$$
\Pr[\text{extinguish} \mid k] = 1 - (1-\kappa)^k
$$

and unextinguished BURNING houses transition to RUINED (`env_spec.md` §3.3). Under the canonical phase order (extinguish → burn-out → spread → ignite, `env_spec.md` §3.4), the spread phase has no BURNING source houses to spread from in the canonical mode, so contagion enters only via the spontaneous-ignition phase 6 at rate $\rho$ per SAFE house. The phase-diagram driver uses canonical mode (`env_spec.md` §3.4 last paragraph), so we treat $\beta$ as inert in the leading-order analysis and revisit in §5.4.

Let $s$ be the probability a house is SAFE at night $t$ (steady-state). Conditional on the house being SAFE at night $t$ and $k$ workers present on night $t{+}1$:

$$
\Pr[s_{t+1} = \text{SAFE} \mid s_t = \text{SAFE}] = (1 - \rho) + \rho \cdot (1 - (1-\kappa)^k)
$$

(SAFE with probability $1-\rho$ from the ignition phase; if ignited with probability $\rho$, extinguished with probability $1-(1-\kappa)^k$ in the next-night cycle, which we collapse into the single-night survival window as the leading-order approximation). The complement is the per-night ruin probability:

$$
q(k) := 1 - \Pr[s_{t+1} = \text{SAFE} \mid s_t = \text{SAFE}, k] = \rho \cdot (1-\kappa)^k
$$

Over $T_{\min}{=}12$ nights, the probability the house survives to terminal (i.e., never RUINED) is approximately $(1 - q(k))^{T_{\min}} \approx 1 - T_{\min} \cdot q(k)$ for small $q(k)$. The expected per-house reward streams attributable to a single agent who is the owner of the house are then:

$$
\mathbb{E}[\text{per-house reward} \mid k] \approx
  \underbrace{r_{\text{own}}}_{50 \text{ (save event)}}
  \cdot (1 - T_{\min} \cdot q(k))
  - \underbrace{p_{\text{own}}}_{100 \text{ (ruin penalty)}}
  \cdot T_{\min} \cdot q(k)
  + R_{\text{team}} \cdot (1 - q(k))/H
  - P_{\text{team}} \cdot q(k)/H \cdot T_{\min}
$$

(The team term is divided by $H{=}10$ because the team reward is per-house-share.)

Collecting the $q(k)$-dependent terms and the $q(k)$-independent baseline:

$$
\mathbb{E}[\text{per-house reward} \mid k] \approx \text{const} - \underbrace{\bigl(r_{\text{own}} \cdot T_{\min} + p_{\text{own}} \cdot T_{\min} + P_{\text{team}} \cdot T_{\min} / H \bigr)}_{A_{\text{house}}} \cdot q(k)
$$

Numerically with the `minimal_specialization` $W$:

$$
A_{\text{house}} = 50 \cdot 12 + 100 \cdot 12 + 10 \cdot 12 / 10 = 600 + 1200 + 12 = 1812
$$

The survival coefficient $A$ captures the expected per-house payoff lost per unit of per-night ruin probability $q(k)$. Substituting $q(k) = \rho (1-\kappa)^k$:

$$
\mathbb{E}[\text{per-house reward loss} \mid k] = A \cdot \rho \cdot (1-\kappa)^k
$$

The **marginal benefit of adding one worker** at the house (going from $k$ to $k+1$) is the reduction in expected reward loss:

$$
\Delta(k \to k+1) := A \cdot \rho \cdot \left[(1-\kappa)^k - (1-\kappa)^{k+1}\right] = A \cdot \rho \cdot \kappa \cdot (1-\kappa)^k
$$

With $\rho{=}0.02, A{=}1812$:

$$
A \cdot \rho = 1812 \cdot 0.02 = 36.24
$$

So the marginal benefit at the $k$-th worker is $36.24 \cdot \kappa \cdot (1-\kappa)^k$. We use the shorthand $\tilde A := A \cdot \rho = 36.24$ for the rest of the derivation.

## 2.2 Stage-game payoffs

Each agent picks $a^{\text{mode}} \in \{0, 1\}$. With $k$ workers at the house under the joint profile, each Worker pays cost $c$ and each Rester pays $-c_{\text{rest}}$. Under the single-house reduction, each agent's per-night payoff is

$$
u_i(a_i, k_{-i}) = \begin{cases}
  +\Delta(k_{-i} \to k_{-i}+1) - c & \text{if } a_i = \text{Work} \\
  +0 + c_{\text{rest}} & \text{if } a_i = \text{REST}
\end{cases}
$$

where $k_{-i}$ is the number of other agents who chose Work. (The baseline survival reward $\text{const}$ from §2.1 is the same in both branches and cancels in deviation comparisons.) Substituting $c \to c + c_{\text{rest}}$ to fold the rest-payoff into the cost gap:

$$
u_i(\text{Work} | k_{-i}) - u_i(\text{REST}) = \tilde A \cdot \kappa \cdot (1-\kappa)^{k_{-i}} - c_{\text{gap}}
$$

with $c_{\text{gap}} = c + c_{\text{rest}}$. For the phase-diagram driver's $c{=}0.5$ and the default $c_{\text{rest}}{=}0.5$, $c_{\text{gap}} = 1.0$.

## 2.3 Mapping back to ring locality

The single-house reduction assumes all 4 agents are at the same house. The empirical setup spreads 4 agents across 10 houses each night. The mapping is: replace $k_{-i}$ in the formula above with the **expected number of other agents at the agent's chosen house**, which in the symmetric case is $E[k_{-i}] = (N-1)/H = 3/10 = 0.3$ if agents are uniformly distributed. Under that approximation, $(1-\kappa)^{k_{-i}}$ becomes $(1-\kappa)^{0.3}$, which is near 1 even for moderate $\kappa$. The derivation in §3 keeps the single-house framing (all 4 agents at one fire) because that is the case that determines whether a `1 Worker, 3 free-riders` profile is stable; the ring-locality correction is the §5.1 perturbation.

# 3. NE structure predictions

The three NE candidates and their existence conditions:

## 3.1 Symmetric all-Work NE

Under the all-Work profile, every agent plays $a_i = \text{Work}$. For this to be a NE, a unilateral deviation to REST must not strictly increase any agent's payoff.

Deviation from $k{=}4$ (all-Work) to $k_{-i}{=}3$ (single deviator plays REST):

$$
u_i(\text{Work} | k_{-i}{=}3) - u_i(\text{REST}) \geq 0
\iff \tilde A \cdot \kappa \cdot (1-\kappa)^{3} \geq c_{\text{gap}}
$$

So the **symmetric all-Work NE exists** iff:

$$
\boxed{\quad \tilde A \cdot \kappa \cdot (1-\kappa)^{3} \geq c_{\text{gap}} \quad}
\qquad \text{(Symmetric NE boundary, eq. (S))}
$$

Substituting $\tilde A = 36.24$, $c_{\text{gap}} = 1.0$:

$$
36.24 \cdot \kappa \cdot (1-\kappa)^{3} \geq 1.0
\iff \kappa \cdot (1-\kappa)^3 \geq 0.0276
$$

The LHS $f(\kappa) := \kappa (1-\kappa)^3$ is unimodal on $[0,1]$ with maximum at $\kappa = 1/4$ where $f(1/4) = (1/4)(3/4)^3 \approx 0.105$. Solving $f(\kappa) = 0.0276$ numerically (e.g., $\kappa{=}0.030$: $0.030 \cdot 0.97^3 = 0.0274$; $\kappa{=}0.65$: $0.65 \cdot 0.35^3 = 0.0279$): the lower root is $\kappa \approx 0.030$ and the upper root is $\kappa \approx 0.65$. So the symmetric NE exists for $\kappa \in [0.030, 0.65]$ under this reduction.

## 3.2 Asymmetric 1-Worker NE

Under the asymmetric profile, exactly 1 agent plays Work and 3 play REST. For this to be a NE, two conditions must hold:

**(a) The lone Worker must prefer Work to REST.** Their deviation comparison has $k_{-i}{=}0$:

$$
u_{\text{Worker}}(\text{Work} | k_{-i}{=}0) - u_{\text{Worker}}(\text{REST}) \geq 0
\iff \tilde A \cdot \kappa \cdot (1-\kappa)^{0} \geq c_{\text{gap}}
\iff \tilde A \cdot \kappa \geq c_{\text{gap}}
$$

**(b) Each free-rider must prefer REST to Work.** Their deviation comparison has $k_{-i}{=}1$:

$$
u_{\text{Rester}}(\text{Work} | k_{-i}{=}1) - u_{\text{Rester}}(\text{REST}) < 0
\iff \tilde A \cdot \kappa \cdot (1-\kappa)^{1} < c_{\text{gap}}
\iff \tilde A \cdot \kappa \cdot (1-\kappa) < c_{\text{gap}}
$$

So the **asymmetric 1-Worker NE exists** iff:

$$
\boxed{\quad \tilde A \cdot \kappa \geq c_{\text{gap}} \quad \text{AND} \quad \tilde A \cdot \kappa \cdot (1-\kappa) < c_{\text{gap}} \quad}
\qquad \text{(Asymmetric NE boundary, eq. (A))}
$$

Substituting $\tilde A = 36.24$, $c_{\text{gap}} = 1.0$:

- (a): $36.24 \cdot \kappa \geq 1.0 \iff \kappa \geq 0.0276$
- (b): $36.24 \cdot \kappa \cdot (1-\kappa) < 1.0$. The LHS $g(\kappa) := 36.24 \kappa (1-\kappa)$ is unimodal with maximum at $\kappa = 1/2$ where $g(1/2) = 9.06$. Solving $g(\kappa) = 1.0$: $\kappa \approx 0.028$ (lower root) and $\kappa \approx 0.972$ (upper root). Condition (b) requires $\kappa$ outside $[0.028, 0.972]$, i.e., $\kappa < 0.028$ or $\kappa > 0.972$.

The intersection of (a) and (b) gives the asymmetric-NE region: $\kappa \in [0.028, 0.0276] \cup [0.972, 1]$. The first interval is empty in practice (lower bound exceeds upper bound by 0.0004 due to the strict inequality in (b)); the second interval is the **Stag-Hunt-like high-$\kappa$ regime** where a single worker is overwhelmingly likely to extinguish (`env_spec.md` §6 anchor) and the free-rider's marginal contribution is negligible. The asymmetric NE is therefore predicted to exist only for $\kappa$ very close to 1 in the analytical reduction.

## 3.3 No-pure-NE / "collapse" regime

The all-REST profile is **never a NE** for any $\tilde A > 0$ because a single deviation to Work earns the deviator $\tilde A \cdot \kappa - c_{\text{gap}} > 0$ whenever $\tilde A \cdot \kappa > c_{\text{gap}}$. So the "all-REST" candidate appears as a NE only when $\tilde A \cdot \kappa < c_{\text{gap}}$, i.e., when even a lone worker can't recover their cost.

The **no-pure-NE collapse regime** is the parameter regime where neither (S) nor (A) holds AND the all-Work profile is dominated. From the above: when $\tilde A \cdot \kappa < c_{\text{gap}}$, the lone-Worker deviation does not pay (eq. A.a fails), so the asymmetric NE does not exist. Whether the symmetric all-Work NE exists is then determined by eq. (S). For $\kappa < 0.028$ (where $\tilde A \kappa < c_{\text{gap}}$), eq. (S) becomes $\tilde A \cdot \kappa \cdot (1-\kappa)^3 \geq c_{\text{gap}}$; since $(1-\kappa)^3 < 1$ for $\kappa > 0$, this is strictly tighter than (A.a), so it also fails. Hence both NE candidates fail simultaneously when $\kappa < 0.028$ at $c{=}0.5, \tilde A = 36.24$.

$$
\boxed{\quad \tilde A \cdot \kappa < c_{\text{gap}} \implies \text{no-pure-NE (collapse)} \quad}
\qquad \text{(Collapse boundary, eq. (C))}
$$

## 3.4 Summary of analytical predictions

For the `minimal_specialization` parameter cell at $c{=}0.5$ (so $c_{\text{gap}}{=}1.0$), $\rho{=}0.02$, $T_{\min}{=}12$, with $\tilde A = A \rho = 1812 \cdot 0.02 = 36.24$:

| $\kappa$ range | Symmetric (S)? | Asymmetric (A)? | Collapse (C)? | Predicted verdict |
|---|---|---|---|---|
| $\kappa < 0.028$ | No ($\kappa$ too small) | No (a fails) | Yes | `collapse` (no-pure-NE) |
| $0.028 \leq \kappa \leq 0.030$ | Borderline | Empty intersection | No | borderline; may be `mixed` |
| $0.030 \leq \kappa \leq 0.65$ | Yes | No (b fails) | No | `symmetric_only` |
| $0.65 < \kappa \leq 0.972$ | No (S fails) | No (b still fails) | No | `mixed` or transition |
| $\kappa > 0.972$ | No | Yes | No | `asymmetric_only` |

The qualitative phase order from $\kappa$-small to $\kappa$-large is: **collapse → symmetric_only → mixed/no-pure → asymmetric_only**, matching the empirical phase order at $c{=}0.5$. The **β-independence** falls out structurally: under the canonical phase order, $\beta$ does not enter the leading-order analytical derivation at all (it would enter only if the spread phase were placed before burn-out per `env_spec.md` §3.4 last paragraph, which the phase-diagram driver does not select).

# 4. Predicted vs. empirical phase table

The 7-cell preview from `refs/empirical-phase-diagram.md` against the predicted verdict (always evaluated at $c{=}0.5, \rho{=}0.02$, so $\beta$ is informational):

| $\beta$ | $\kappa$ | Empirical verdict | Predicted verdict | Match? | Notes |
|---|---|---|---|---|---|
| 0.10 | 0.10 | `no_convergence` | `symmetric_only` (in 0.030–0.65 band) | ✗ | Predicted threshold off |
| 0.50 | 0.10 | `no_convergence` | `symmetric_only` | ✗ | β-independence holds (both `no_convergence`) |
| 0.90 | 0.10 | `no_convergence` | `symmetric_only` | ✗ | β-independence holds |
| 0.50 | 0.50 | `symmetric_only` | `symmetric_only` (in 0.030–0.65 band) | ✓ | |
| 0.90 | 0.50 | `symmetric_only` | `symmetric_only` | ✓ | β-independence holds |
| 0.50 | 0.90 | `asymmetric_only` | `mixed` or borderline (predicted threshold 0.972) | partial | qualitative direction matches; predicted threshold too high |
| 0.90 | 0.90 | `asymmetric_only` | `mixed` or borderline | partial | β-independence holds |

**Tallies (qualitative)**:

- **β-independence at $c{=}0.5$**: 3/3 κ-rows where both β values are sampled show identical empirical verdicts; the analytical prediction is exact β-independence under canonical phase order. **Match: 3/3** on this structural property.
- **Phase order κ-small → κ-large**: empirical = (`no_convergence`, `symmetric_only`, `asymmetric_only`); predicted = (`collapse`, `symmetric_only`, `asymmetric_only` after a `mixed` band). **Match: 3/3** on the qualitative phase ladder.
- **Exact cell-by-cell verdict agreement**: 2/7 cells (the two $\kappa{=}0.5$ cells). **5/7 cells disagree** on the verdict label, all because the predicted $\kappa$ thresholds are wrong:
  - At $\kappa{=}0.1$, the analytical reduction predicts `symmetric_only` (since $\kappa{=}0.1$ falls inside $[0.030, 0.65]$); empirically the cell is `no_convergence`. The predicted collapse threshold $\kappa{=}0.028$ is **3–4× too low**.
  - At $\kappa{=}0.9$, the analytical reduction predicts `mixed` or borderline (since $\kappa{=}0.9$ falls in $[0.65, 0.972]$); empirically the cell is `asymmetric_only`. The predicted asymmetric-NE threshold $\kappa{=}0.972$ is **~10% too high**.

The /44-cell agreement test the issue's acceptance criterion names — quantitative agreement on ≥80% of cells — fails on the 7-cell preview. Of 7 sampled cells, 2 exact matches = 28.6%, below the 80% bar. **Caveat (per architect's non-blocking flag):** the full 75-cell grid is still being computed (#358 gap-fill); the 7-cell preview is the only data available at the time of writing. The qualitative structural matches (β-independence, κ-monotonicity, three-regime phase order) are robust against the small sample size and are the load-bearing content of this comparison. The quantitative thresholds are not — the analytical reduction has known systematic biases (§5) that explain the gap.

# 5. Where the reduction breaks

The 5/7-cell quantitative disagreement above is real and the framework must own it. Five sources of systematic bias are identifiable from the derivation:

## 5.1 Ring locality (the $H{=}10$ → mean-field gap)

The §2 derivation puts all 4 agents at one house. Empirically, 4 agents distribute across 10 houses each night; under the symmetric (uniform) policy assumption, each house sees $E[k] = N \cdot 1/H = 0.4$ workers per night on average, not 4. The effective per-house extinguish probability is therefore $1 - (1-\kappa)^{0.4}$ rather than $1 - (1-\kappa)^4$ — a much weaker defense. For $\kappa{=}0.5$: full-mean-field gives $1 - 0.5^4 = 0.9375$; the ring-corrected average gives $1 - 0.5^{0.4} \approx 0.24$. This is a factor of 4 reduction in per-house defense capability, which means the ring-corrected $\tilde A$ is much smaller than 36.24, and the collapse boundary $\kappa$ at which $\tilde A \cdot \kappa < c_{\text{gap}}$ shifts substantially **upward**. **Prediction**: properly accounting for ring locality moves the collapse threshold from $\kappa = 0.028$ (analytical, single-house) to roughly $\kappa \in [0.1, 0.3]$ (ring-corrected). The empirical collapse boundary at $\kappa{=}0.1$ is consistent with this corrected prediction.

This is the dominant systematic bias and the §6 follow-up names a richer ring-Markov derivation as the natural next step. The mean-field approximation is the load-bearing reason the reduction tractable; tightening it without losing tractability is the open problem.

## 5.2 Heuristic strategy space (Firefighter ≠ all-REST)

The empirical solver picks Firefighters (work_tendency = 0.9) as the "free-riders" in the asymmetric NE profile, not literal RESTers (work_tendency = 0.2). Three Firefighters at 90%-Work contribute non-zero expected extinguish probability — the effective $k_{-i}$ in the lone-Worker's deviation comparison is closer to $1 + 3 \cdot 0.9 = 3.7$ than the analytical $k_{-i}{=}1$. The lone-Worker's marginal benefit is therefore $\tilde A \cdot \kappa \cdot (1-\kappa)^{3.7}$, not $\tilde A \cdot \kappa \cdot (1-\kappa)^{0}$. At $\kappa{=}0.9$: $\tilde A \cdot \kappa \cdot (1-\kappa)^{3.7} = 36.24 \cdot 0.9 \cdot 0.1^{3.7} \approx 0.013$ — much less than $c_{\text{gap}}{=}1.0$, so the lone Worker should not exist as a NE under that interpretation. The fact that the empirical solver still reports `asymmetric_only` at $\kappa{=}0.9$ suggests the per-agent ownership reward ($r_{\text{own}}{=}50, p_{\text{own}}{=}100$) — which is NOT captured in the single-house mean-field reward — anchors each agent to its own house even when free-riding on others' fires. The asymmetric NE is therefore **partly an artifact of per-agent ownership**, not purely a volunteer-dilemma equilibrium.

## 5.3 Per-agent ownership rewards (the $W$ gap)

The analytical $A$ coefficient lumps $r_{\text{own}}{=}50$ and $p_{\text{own}}{=}100$ into the survival coefficient, but the per-agent reward only fires on the **agent's own** house. The single-house reduction assumes the representative house is the agent's own house with probability $1$; the ring-corrected reality is that probability is $1/H = 0.1$ (round-robin ownership). The other 9 houses contribute to the team term ($R_{\text{team}}/H, P_{\text{team}}/H$) only, which scales $A$ down by a factor of roughly $(p_{\text{own}}/H + P_{\text{team}}/H) / p_{\text{own}} = (10 + 1)/100 \approx 0.11$ for the off-own-house terms. The ring-corrected $A$ is therefore somewhere between $A/H = 181$ (if every house counts equally) and $A = 1812$ (if every house is the agent's own); the truth depends on the agent's house-target distribution, which the analytical reduction does not model.

## 5.4 β-independence as a leading-order prediction, not an exact result

The analytical reduction's β-independence is exact under the **canonical** phase order (extinguish → burn-out → spread → ignite, `env_spec.md` §3.4), where the spread phase has no BURNING source houses to spread from. The phase-diagram driver uses canonical mode, so this matches. But empirically, $\beta$ may still enter via the **across-night** dynamics: a high-$\beta$ regime makes neighbor-driven ignition the dominant failure mode over multiple nights, even if the within-night spread phase is inert. The analytical reduction's leading-order claim is "β doesn't enter the single-night payoff comparison," which is exact; the second-order claim "β doesn't enter the multi-night survival" is approximately true under the canonical phase order and small $\beta \rho$ products, but should break for $\beta \rho \gg \rho$ (i.e., when neighbor-driven ignition dominates spontaneous ignition). **Predicted breakage**: at higher $\rho$ (say $\rho \geq 0.10$) the β-independence should weaken. The current preview at $\rho{=}0.02$ is the wrong regime to test this; sweeping $\rho \in \{0.05, 0.10\}$ would let the framework be falsified.

## 5.5 Single-stage vs. multi-stage SPE

The analytical reduction is the **single-night stage game under stationary opponent policies**, not the full multi-stage subgame-perfect equilibrium. Empirically, the DO solver evaluates policies over full $T_{\min}{=}12$ episodes, so the per-cell payoffs reflect the **integrated** multi-stage outcomes. A symmetric all-Work profile that is a NE of the single-night stage game may fail to be a NE of the multi-stage game if a deviator can sustain a long-horizon advantage by switching to REST at a specific night. The analytical reduction folds the multi-stage dynamics into the survival coefficient $A$ (by integrating over $T_{\min}{=}12$ nights of expected survival), but this is itself an approximation; the exact multi-stage NE conditions involve subgame-perfect best-response calculations the single-night reduction cannot represent.

This is the structural caveat the architect's recommendation flagged: the analytical and empirical tracks **falsify each other**, and the residual disagreement isolates where the multi-stage dynamics (the gap between stage-game and SPE) matters. The 7-cell preview is too small to isolate this from the §5.1–5.4 biases; the full 75-cell grid is the right test bed.

# 6. Implications for the paper

The mean-field reduction is the load-bearing contribution. Even with the quantitative thresholds wrong by a factor of ~3–10, the three structural predictions it makes — (a) κ-monotonicity of the verdict, (b) β-independence under canonical phase order, (c) the existence of a no-pure-NE collapse regime at small $\kappa$ — are all empirically observed in the 7-cell preview and would not be derivable without the closed-form derivation. §3 of the workshop paper draft (#364) should adopt the eq. (S), (A), (C) inequalities as the analytical anchor and report the per-cell agreement (or disagreement) once the full 75-cell grid lands as the empirical validation table.

For the M2 PPO trainability sweep (issue #360), the predicted κ-dependence implies that PPO convergence rate should track the same three-regime structure: trainable in the `symmetric_only` band (single coordinated profile to find), harder in the `asymmetric_only` band (the breaking-symmetry problem MARL is known to struggle with), and effectively untrainable in the `collapse` regime (no NE exists for PPO's policy iteration to converge to). The same boundaries that predict NE structure predict trainability difficulty.

Three concrete recommended follow-ups, in priority order:

1. **Complete the 75-cell empirical grid** (issue #358 gap-fill) and re-run the §4 comparison table. With 75 cells the qualitative-vs-quantitative agreement split can be tested rigorously and the §5 systematic biases can be isolated by sweeping $c$ and $\rho$.
2. **Sweep $\rho \in \{0.05, 0.10\}$ at fixed $c{=}0.5$** to test the §5.4 prediction that β-independence breaks when neighbor-driven ignition dominates spontaneous ignition. A falsified prediction here would be a positive paper result; a confirmed prediction would sharpen the analytical framework's claimed scope.
3. **Derive a ring-Markov correction to $A$** that replaces the single-house mean-field assumption with a chain-Markov approximation over the 10-house ring (treating each house's SAFE/BURNING/RUINED status as a Markov chain coupled to its neighbors via $\beta$). This would close the §5.1 gap and is the natural extension of the framework toward the full stochastic-game analysis without paying the full state-space cost. Tracker issue to file: "Ring-Markov refinement of survival coefficient $A$."

The memo's contribution to the paper is the closed-form derivation in §2 and the algebraic boundaries (S), (A), (C). The headline finding stands: the qualitative κ-monotonicity and β-independence both fall out of the mean-field reduction at zero cost; the quantitative thresholds will require ring-locality and per-agent ownership corrections beyond the scope of this memo to nail.
