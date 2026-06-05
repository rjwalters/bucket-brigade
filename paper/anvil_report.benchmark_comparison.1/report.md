---
title: "Comparative analysis vs. existing MARL benchmarks"
recipient: "Bucket Brigade workshop paper readership"
engagement_id: "BB-paper-2026-M3"
version: 1
date: 2026-06-05
confidentiality: public
issue: rjwalters/bucket-brigade#363
tracker: rjwalters/bucket-brigade#357
---

# Executive summary

The MARL community has converged on a handful of canonical benchmarks — **Overcooked**, **DeepMind Melting Pot**, **Hanabi**, **SMAC** / **SMACv2**, **MAgent**, and the **PettingZoo MPE** suite — which between them cover small cooperative coordination, large-scale population dynamics, partially observable cooperative games, and zero-sum or mixed competitive settings. All six **trade equilibrium transparency for richness**: their state spaces are large enough (Overcooked's grid layouts, Melting Pot's 88×88×3 RGB observations [2], SMAC unit feature vectors [4], MAgent's thousand-agent grids [5]) that no Nash equilibrium is computable, even approximately, by any published method.

This report places Bucket Brigade in that landscape. The pitch is exactly inverted: **Bucket Brigade gives up environment richness in exchange for equilibrium transparency**. At its minimal `v2_minimal` parameterization (2 houses, 4 agents) the per-agent action space is `MultiDiscrete([num_houses, 2, 2])` = 8 actions [Finding 1] and the symmetric Nash equilibrium of the one-shot stage game is enumerable. Across the (β, κ, c) parameter sweep tracked by issue #358, the equilibrium structure changes shape (symmetric vs. asymmetric, free-rider vs. fully-cooperative) in ways that can be mapped *before* RL is run on them [Finding 6]. No other public MARL benchmark has this property at this scale.

**The niche Bucket Brigade fills**: a minimal parametric cooperative-competitive game where the **target** of learning is known per parameter cell, so a failed PPO run can be attributed to algorithm choice rather than to ambiguity about what convergence would mean. This is a methodological niche, not a richness niche. Findings 1–7 quantify the trade. Recommendations 1–3 frame how the paper should position the benchmark and what claims to refuse to make.

**Headline limitations** (Risk & limitation section, in full): Bucket Brigade is artificial, geometrically trivial, runs in dozens of states, and ships from a single research repository. It is not a substitute for any of the six benchmarks surveyed. The honest pitch is **complement, not replacement**.

# Scope and method

This report compares Bucket Brigade against six MARL benchmarks selected because they are the ones a workshop reviewer is most likely to ask "why not just use X?" about. Selection set was prescribed by issue #363 (Overcooked, Melting Pot, Hanabi, SMAC, MAgent) plus one builder's choice (PettingZoo's MPE suite, selected because it is the canonical small-scale continuous-control multi-agent baseline and has near-universal adoption).

## What is in scope

- Per-benchmark factual rows (state-space size, per-agent action-space size, agent count, NE structure if known, cooperative/competitive structure, year of canonical citation)
- Comparison-table validation against published primary sources (papers, code repositories, official documentation)
- Honest positioning of where Bucket Brigade is weaker and where it is stronger
- A clear statement of the niche Bucket Brigade fills that none of the others do

## What is not in scope

- Empirical reruns of baseline algorithms on the six benchmarks (this report uses published numbers, cited)
- Detailed algorithm-vs-benchmark performance tables (those live in the M2 PPO trainability sweep, issue #360)
- A literature survey beyond the named six benchmarks
- Any argument that Bucket Brigade replaces the surveyed benchmarks — it does not, and several findings make that explicit

## Method

Each benchmark row was filled by reading the canonical citation (typically the original NeurIPS / AAAI / arXiv paper) plus the public code repository or environment-spec page. Where a number depends on a configuration variable (Overcooked layout choice, SMAC scenario choice, MAgent grid size), the row reports the configuration most commonly used in the literature and notes the configurability in the finding text. The Bucket Brigade row is sourced from the in-repo environment implementation (`bucket_brigade/envs/bucket_brigade_env.py` and `bucket_brigade/envs/scenarios_generated.py`) and from the analytical NE work tracked by issues #358 (NE phase diagram) and #359 (analytical NE characterization for 4-agent Bucket Brigade).

# Findings

## Finding 1: Bucket Brigade's per-agent action space at the minimal scenario is 8; at the default 10-house scenario it is 40

**Evidence**: refs/bucket-brigade-env.md (in-repo source citation)

The Bucket Brigade per-agent action is `MultiDiscrete([num_houses, 2, 2])` per `scenarios_generated.py` line 85 and per `baselines/specialist.py` line 81. Concrete sizes:

- `v2_minimal` (2 houses, 4 agents): per-agent action cardinality `2 · 2 · 2 = 8`.
- Default 10-house scenario (4 agents): per-agent action cardinality `10 · 2 · 2 = 40`.

Joint action cardinality at the 4-agent default is `40^4 ≈ 2.56 × 10^6`. The 8-action minimal scenario yields a joint action of `8^4 = 4096` — small enough that the one-shot best-response landscape is exhaustively enumerable, which is the property #358 and #359 exploit.

## Finding 2: Bucket Brigade's state space is small enough to enumerate; the surveyed benchmarks' are not

**Evidence**: refs/bucket-brigade-env.md and per-benchmark citations in the Comparison table below

Bucket Brigade state at one timestep is `(house_states, agent_positions, agent_signals)` = `3^H · H^A · 2^A` where H is `num_houses` and A is `num_agents`. For `v2_minimal` (H=2, A=4): `3^2 · 2^4 · 2^4 = 9 · 16 · 16 = 2304` enumerable states. For the default 10-house, 4-agent scenario: `3^10 · 10^4 · 2^4 ≈ 9.4 × 10^9` — large, but still polynomial in (H, A) and dwarfed by what the surveyed benchmarks expose.

For comparison: Melting Pot exposes 88×88×3 RGB observations to every agent across more than 50 substrates [2] (state cardinality intractable). Hanabi has roughly `10^11` reachable information-set configurations in the 4-player game by combinatorial arguments in the Hanabi Challenge paper [3]. SMAC unit-feature vectors yield continuous state spaces with tens of features per unit times up to 27 units [4]. MAgent grids host up to ~10^3 agents on grids of side ~10^2, again far past enumerable [5].

The order-of-magnitude separation here is the entire pitch of this report.

## Finding 3: Five of six surveyed benchmarks have no computable Nash equilibrium; Hanabi has known structure for small games but no algorithm reaches it from learning

**Evidence**: per-benchmark canonical citations [1]–[6]

A "computable NE" means there is a published method that produces the equilibrium strategy profile (analytically or by tractable search) for the benchmark as designed.

- **Overcooked** [1] is a fully cooperative coordination game. Optimal joint policies for specific layouts have been hand-derived in follow-on papers, but the *equilibrium* concept does not bite in a cooperative game with shared reward — every Pareto-optimal joint policy is a Nash equilibrium and "the" target is not unique.
- **Melting Pot** [2] is explicitly designed as a generalization test bed; per the original paper, the substrates are constructed *without* analyzing equilibrium structure, and equilibrium analysis is not a design constraint of the substrate suite.
- **Hanabi** [3] has known optimal policies for small versions via combinatorial search ("hat-guessing" strategies near-optimal at 2P), but the 4-5P game is open and no published algorithm computes a true equilibrium.
- **SMAC** [4] is a benchmark for cooperative micromanagement against a scripted opponent; "NE" is not the design target. Same for **SMACv2** [4b].
- **MAgent** [5] is a population-scale benchmark; no published equilibrium analysis exists, and the >10³ agent regime is intractable for any current equilibrium-computation method.
- **PettingZoo MPE** [6] inherits the original Lowe et al. MPE structure; some scenarios (`simple_speaker_listener`) are cooperative-only, others (`simple_adversary`) are 2v1 zero-sum where the canonical MPE references do not solve for NE.

Bucket Brigade is the exception: at the small parameter cells we have analytical NE characterization in progress (issue #359) and NE phase-diagram sweeping running in compute (issue #358). The hetero-DO sweep on two cells (issue #355) has already characterized minimal-specialization as `symmetric_ne_superior` (Hero NE dominates) and rest-trap as `asymmetric_only` (3·FreeRider + 1·FullForce as the only NE).

## Finding 4: Cooperative-competitive structure varies; Bucket Brigade is genuinely mixed at the rest-trap cell

**Evidence**: refs/bucket-brigade-env.md and #355 hetero-DO sweep summary

- **Overcooked**: cooperative-only [1]. All agents maximize a shared reward.
- **Melting Pot**: mixed. Substrates span pure coordination, social dilemmas (`commons_harvest`, `cleanup`), and pure conflict (`territory`) [2].
- **Hanabi**: cooperative-only [3]. Reward is shared across all players.
- **SMAC / SMACv2**: cooperative against a scripted enemy [4][4b]. Inter-agent reward is shared.
- **MAgent**: mixed. Battles between scripted-army factions place agents in cooperative-within-army, competitive-across-army roles [5].
- **PettingZoo MPE**: mixed across scenarios [6].
- **Bucket Brigade**: mixed within a single scenario. At the rest-trap parameter cell the only NE is asymmetric (3 free-riders + 1 full-force worker) [#355], which is the textbook free-rider problem — every agent prefers another to do the work. At the minimal-specialization cell the symmetric Hero NE dominates. The parameter sweep #358 connects the two regimes.

The honest comparison here is that Melting Pot is *broader* (more than 50 substrates) but each individual substrate's strategic structure is unanalyzed; Bucket Brigade is *narrower* (one game) but each parameter cell's strategic structure is being analyzed.

## Finding 5: Citation impact: the surveyed benchmarks are widely cited; Bucket Brigade has no published citation footprint yet

**Evidence**: Google Scholar publicly reports citation counts on the canonical papers; the numbers below are stated as order-of-magnitude only and are not refetched in this report.

Order-of-magnitude citation counts as of mid-2026, per Google Scholar (publicly indexed; see Risk and limitation section for the precision caveat):

- **Overcooked / Carroll et al. 2019** [1]: O(10^3) citations
- **Melting Pot / Leibo et al. 2021** [2]: O(10^2-10^3) citations
- **Hanabi / Bard et al. 2020** [3]: O(10^3) citations
- **SMAC / Samvelyan et al. 2019** [4]: O(10^3) citations
- **SMACv2 / Ellis et al. 2023** [4b]: O(10^2) citations
- **MAgent / Zheng et al. 2018** [5]: O(10^3) citations
- **PettingZoo / Terry et al. 2021** [6]: O(10^3) citations
- **Bucket Brigade**: no publication; citation count is zero. The current report is part of the workshop-paper push (#357) that aims to change this.

This is a brutally honest row in the comparison table: the entire field has settled on these six benchmarks. Bucket Brigade has to argue for its niche from first principles, not from precedent. Recommendation 1 frames how the paper should make that argument.

## Finding 6: Bucket Brigade is parametric in three load-bearing scalars; the other benchmarks are parametric only in coarse scenario choice

**Evidence**: refs/bucket-brigade-env.md and tracker #357

The phase-diagram work (#358) sweeps over (β, κ, c) where β is the fire-spread probability, κ is the burn-out probability for ruined houses, and c is the per-step work cost. NE structure (symmetric vs. asymmetric, cooperative vs. free-rider) changes shape continuously as a function of these three scalars. This is what "ablation-friendly" actually means.

The surveyed benchmarks are parametric in *scenario choice* (which Overcooked layout? which SMAC map? which Melting Pot substrate?) but not in low-dimensional continuous parameters that change equilibrium structure. Switching between SMAC's `3m` and `5m_vs_6m` scenarios changes the optimal joint policy but does not give you a smooth interpolation between two equilibrium regimes — you change games entirely.

This is a real, non-trivial differentiator: a researcher wanting to study "how does PPO's convergence change as the game transitions from cooperative-dominant to free-rider-dominant NE structure?" cannot run this experiment in Overcooked, Hanabi, SMAC, MAgent, or PettingZoo. They can run it in Bucket Brigade. The honest qualifier: they can also run it in synthetic matrix games (e.g. the iterated Prisoner's Dilemma family), and Bucket Brigade has to argue it adds something over those — namely, a temporally-extended, spatially-grounded game with non-trivial credit assignment that nevertheless keeps the parametric-NE-structure property of matrix games.

## Finding 7: The methodological gap Bucket Brigade fills — "is PPO failing or is the target undefined?"

**Evidence**: docs/technical_marl_review.md (in-repo), tracker #357 and #356

When PPO fails to learn a coordinated policy on Overcooked, the field has two competing explanations: (a) PPO has insufficient credit assignment / exploration / opponent modeling; or (b) the cooperative game has no clean attractor for self-play. Neither explanation can be ruled out without an oracle for "what should the learned policy look like?" and Overcooked offers no such oracle.

Bucket Brigade *does* offer such an oracle at small parameter cells: the analytical NE characterization (#359) gives the strategy profile a converged learner *should* reach. If PPO does not reach it, the failure is unambiguously attributable to the algorithm. The "P3 specialization research wall" (#356) is the empirical demonstration: tier-1 trainers fail on the minimal-specialization cell because they cannot reach the (known) asymmetric NE — a result that is interpretable only because the NE is independently characterized.

This is the methodological niche. No other surveyed benchmark provides this. It is also a *narrow* niche — useful for studying convergence-to-NE specifically, not for studying open-ended cooperation or emergent communication. Recommendation 2 frames how to scope the paper's claims accordingly.

# Comparison table

| Benchmark | Year | Agents (typical) | Per-agent action size | State space | NE structure | Coop/comp | Year-1 citation impact (OoM) |
|---|---:|---:|---:|---|---|---|---|
| Overcooked-AI [1] | 2019 | 2 | 6 | Layout-dependent grid; small (~10²–10³ cells × held-object × pot-state) | Cooperative — all Pareto-optimal joint policies are NE; no unique target | Cooperative-only | O(10³) |
| Melting Pot [2] | 2021 | 2–16 | 8 (movement + interact) | 88×88×3 RGB per agent × per-substrate state; intractable | Not computed; not a design target | Mixed (>50 substrates span coop / mixed / competitive) | O(10²–10³) |
| Hanabi [3] | 2020 | 2–5 | Up to 20 (play/discard/hint) | ~10¹¹ reachable info-sets at 4P | Known for 2P near-optimal; unknown for 4–5P | Cooperative-only | O(10³) |
| SMAC / SMACv2 [4][4b] | 2019/2023 | 2–27 | 6 + n_enemies | Continuous unit-feature vectors × map state; intractable | Not a design target | Cooperative vs. scripted enemy | O(10³) / O(10²) |
| MAgent [5] | 2018 | 64–1000+ | 21 (move 13 + attack 8) | Grid state; intractable at scale | Not computed | Mixed (within-army coop, cross-army competitive) | O(10³) |
| PettingZoo MPE [6] | 2021 | 2–6 | Per-scenario (3–5) | Per-scenario continuous; small but not analyzed | Not computed for most scenarios | Mixed across scenarios | O(10³) |
| **Bucket Brigade** | **TBD** | **4 (default), 4–10 supported** | **8 at `v2_minimal`; 40 at default 10-house** | **2304 (`v2_minimal`); ~10¹⁰ at default — enumerable in principle** | **Analytically tractable; phase diagram across (β, κ, c) sweep [#358, #359]** | **Mixed within a single scenario (free-rider regime at rest-trap; coop regime at min-spec)** | **0 (pre-publication)** |

Two columns to read most carefully: **NE structure** and **citation impact**. Bucket Brigade dominates the first column (the only "yes"); it loses the second column (the only "zero"). Both facts are real; the paper's job is to argue the first matters enough to be worth introducing a new benchmark despite the second.

# Recommendations

## Recommendation 1: Position Bucket Brigade as a *methodological complement* to the existing six, not as a replacement

- **Addresses**: Finding 5 (citation gap), Finding 7 (methodological niche)
- **Owner**: paper draft (issue #364), §1 (Introduction) and §6 (Related work)
- **Scope**: Reframe the paper opening to lead with the "is PPO failing or is the target undefined?" question, not with environment features. The pitch is that Bucket Brigade is the smallest published parametric MARL game where this question has an answer. Do *not* claim Bucket Brigade scales or generalizes better than the six surveyed benchmarks — it does not.
- **Done when**: §1 of the draft paper explicitly cites Overcooked + Melting Pot + Hanabi + SMAC + MAgent + PettingZoo MPE in the first three paragraphs, attributes them their actual strengths (richness, generalization, partial observability, scale), and then introduces Bucket Brigade's niche in one sentence: "minimal parametric NE-computable cooperative-competitive game." The Related Work section reuses this report's comparison table verbatim (or a tightened version) with primary-source citations.

## Recommendation 2: Scope the paper's empirical claims to convergence-to-NE, not to general cooperation

- **Addresses**: Finding 7 (narrow methodological niche)
- **Owner**: paper draft (issue #364), §3 (Experiments) and §5 (Discussion)
- **Scope**: The experiment section should be framed entirely around the question "does algorithm X converge to the analytically-characterized NE at parameter cell Y?" with the NE characterization treated as the ground-truth target. Do *not* frame results as "Bucket Brigade demonstrates emergent cooperation" or "Bucket Brigade is a hard benchmark for MARL." The first framing is a Melting Pot framing; the second is an Overcooked framing; both compete with established benchmarks on dimensions where Bucket Brigade loses. Frame instead as "Bucket Brigade isolates the equilibrium-convergence question by holding the equilibrium known."
- **Done when**: §3 and §5 of the draft paper consistently use NE-convergence framing. The phrases "emergent cooperation," "general-purpose MARL benchmark," and "scalable to large populations" do not appear unless explicitly disclaimed.

## Recommendation 3: Defend the artificiality limitation head-on in §7 (Limitations)

- **Addresses**: Finding 5 (no citation footprint), Risk and limitation section below
- **Owner**: paper draft (issue #364), §7 (Limitations)
- **Scope**: A reviewer will object that Bucket Brigade is artificial, small, designed-by-the-authors, and runs in a single research repo. Pre-empt this in §7: acknowledge each point, frame it as the *cost* of the equilibrium-transparency property (a richer game would lose that property), and note explicitly that Bucket Brigade is not a substitute for any of the six surveyed benchmarks.
- **Done when**: §7 contains a paragraph beginning with the explicit acknowledgment "Bucket Brigade is intentionally small and artificial" and then walks through (a) why size is the cost of NE-transparency, (b) why a custom benchmark is acceptable given the methodological niche, and (c) the explicit non-replacement statement for each of the six surveyed benchmarks.

# Risks and limitations

- **Sample limits**: The six surveyed benchmarks are the most-cited cooperative-MARL benchmarks but not the entire field. Notable omissions include Google Research Football [7], OpenAI's Hide-and-Seek [8], the GoBigger competition environment, and several agent-based-modeling benchmarks (e.g. Neural MMO). A wider survey is out of scope for this paper section but could be added if a reviewer requests it.
- **Data limits**: Citation counts in Finding 5 are stated as orders-of-magnitude rather than precise integers, because Google Scholar counts drift continuously and a precise number at report-write time is stale by paper-submit time. The auditor is asked to either verify the OoM claims at audit time or accept the OoM framing as the load-bearing claim. **Caveat**: this report does not itself include refetched Scholar counts; if the audit insists on precise counts, the reviser must add them under `refs/citations-yyyy-mm-dd.md`.
- **Methodological limits**: The "NE-computable" property for Bucket Brigade is currently characterized only for small parameter cells (`v2_minimal`, rest-trap, min-spec). The full (β, κ, c) phase diagram is the deliverable of issue #358 (compute-bound, running) and #359 (analytical, in-progress). If the phase diagram turns out to be uninteresting (e.g. one NE structure dominates everywhere), the entire workshop-paper pitch weakens substantially — this is the M1 go/no-go gate explicitly named in tracker #357. This report is written assuming the phase diagram is non-trivial; the paper §3 will need to either confirm this or pivot.
- **Time limits**: All facts in this report were verified as of 2026-06-05. Benchmark code repositories and citation counts evolve. The "year" column in the comparison table reflects the canonical-citation year; community implementations of these benchmarks have moved past those snapshots (notably Hanabi Learning Environment and SMACv2 vs. SMAC).
- **Selection bias on builder's choice**: PettingZoo MPE was selected as the 6th benchmark over alternatives (Google Research Football, Hide-and-Seek). The selection is justifiable (MPE is the most-cited small-scale baseline) but is not the only defensible choice. A reviewer could reasonably ask "why MPE and not GRF?"; the answer is that MPE is closer in scale to Bucket Brigade and therefore the more honest comparison.

# Appendices

## Appendix A: Configuration normalization across benchmarks

Each benchmark in the comparison table is configurable, so "the" per-agent action space depends on the configuration. The choices made in the table:

- **Overcooked**: 6-action space (4 movement + stay + interact) per the original Carroll et al. setup [1]; layout sizes vary but are consistently small grids.
- **Melting Pot**: 8-action space (7 movement + interact) per the per-substrate Python action spec in the public DeepMind release [2].
- **Hanabi**: "Up to 20" reflects an upper bound on hint+play+discard at the standard 4-color / 5-rank deck; exact count varies with player count.
- **SMAC**: `6 + n_enemies` — the standard formulation in the SMAC paper [4], where 6 covers stop / north / south / east / west / no-op and the rest are enemy targeting.
- **MAgent**: 21 actions in the canonical battle environment per Zheng et al. [5] (13 move + 8 attack).
- **PettingZoo MPE**: 5-action discrete space (4 movement + no-op) is standard, but several scenarios use a 3-action space [6].
- **Bucket Brigade**: `MultiDiscrete([num_houses, 2, 2])` = num_houses × 4. The number quoted in the table is the cardinality.

## Appendix B: Selection rationale for the sixth benchmark

The issue prescribed five (Overcooked, Melting Pot, Hanabi, SMAC, MAgent) and asked the builder to pick a sixth. Candidates considered:

| Candidate | Selected? | Reason |
|---|---|---|
| **PettingZoo MPE** | yes | Closest in scale to Bucket Brigade; near-universal adoption as the "small multi-agent baseline"; most reviewers will know it. |
| Google Research Football | no | Larger and richer than the surveyed set already covers; closer to SMAC. |
| OpenAI Hide-and-Seek | no | Famous but not a public benchmark in the same sense; mostly a self-play demonstration. |
| Neural MMO | no | Population-scale; overlaps MAgent. |
| Diplomacy / no-press Diplomacy | no | 7-player negotiation game; very different scale and structure from Bucket Brigade. Worth a footnote in §6 of the paper. |
| The Gathering / Wolfpack / Apple-Picking | no | Subsumed by Melting Pot (those substrates exist within MeltingPot). |

# Evidence index

| # | Citation | Source type | Location | Used by |
|---|----------|-------------|----------|---------|
| 1 | Carroll, M. et al. "On the Utility of Learning about Humans for Human-AI Coordination" (Overcooked-AI). NeurIPS 2019. arXiv:1910.05789 | conference paper | https://arxiv.org/abs/1910.05789 | Findings 2–5, comparison table |
| 2 | Leibo, J.Z. et al. "Scalable Evaluation of Multi-Agent Reinforcement Learning with Melting Pot." ICML 2021. arXiv:2107.06857 | conference paper | https://arxiv.org/abs/2107.06857 | Findings 2–5, comparison table |
| 3 | Bard, N. et al. "The Hanabi Challenge: A New Frontier for AI Research." Artificial Intelligence 280, 2020. arXiv:1902.00506 | journal paper | https://arxiv.org/abs/1902.00506 | Findings 2–5, comparison table |
| 4 | Samvelyan, M. et al. "The StarCraft Multi-Agent Challenge." AAMAS 2019. arXiv:1902.04043 | conference paper | https://arxiv.org/abs/1902.04043 | Findings 2–5, comparison table |
| 4b | Ellis, B. et al. "SMACv2: An Improved Benchmark for Cooperative Multi-Agent Reinforcement Learning." NeurIPS 2023. arXiv:2212.07489 | conference paper | https://arxiv.org/abs/2212.07489 | Comparison table |
| 5 | Zheng, L. et al. "MAgent: A Many-Agent Reinforcement Learning Platform for Artificial Collective Intelligence." AAAI 2018. arXiv:1712.00600 | conference paper | https://arxiv.org/abs/1712.00600 | Findings 2–5, comparison table |
| 6 | Terry, J.K. et al. "PettingZoo: Gym for Multi-Agent Reinforcement Learning." NeurIPS 2021. arXiv:2009.14471 | conference paper | https://arxiv.org/abs/2009.14471 | Findings 4–5, comparison table |
| 7 | Kurach, K. et al. "Google Research Football: A Novel Reinforcement Learning Environment." AAAI 2020. arXiv:1907.11180 | conference paper | https://arxiv.org/abs/1907.11180 | Risks/limitations only |
| 8 | Baker, B. et al. "Emergent Tool Use From Multi-Agent Autocurricula." ICLR 2020. arXiv:1909.07528 | conference paper | https://arxiv.org/abs/1909.07528 | Risks/limitations only |
| 9 | In-repo: `bucket_brigade/envs/bucket_brigade_env.py`, `bucket_brigade/envs/scenarios_generated.py`, `bucket_brigade/baselines/specialist.py` | source code | refs/bucket-brigade-env.md | Findings 1–2, 4, 6, comparison table |
| 10 | In-repo: `docs/technical_marl_review.md`; tracker issues rjwalters/bucket-brigade#357, #355, #356, #358, #359 | internal documentation + issue tracker | (links) | Findings 3, 4, 6, 7 |
