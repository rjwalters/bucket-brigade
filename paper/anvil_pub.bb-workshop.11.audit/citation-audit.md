# Citation audit for anvil_pub.bb-workshop.11

Every `\citep{}` / `\citet{}` in `main.tex` was enumerated (**18 distinct
keys**, 35 key-occurrences across cite commands) and checked against
`refs.bib` (**18 entries — all 18 keys resolve; no orphan entries**).
Claim support was assessed against the on-disk evidence:
`paper/anvil_pub.bb-workshop/refs/benchmark_comparison.md` (secondary
evidence index for the benchmark keys) and
`paper/anvil_pub.bb-workshop.10.litsearch/` (`notes.md` provenance table +
`candidates.bib` — resolver-verified metadata for the 11 non-benchmark
keys). **No primary PDF of any cited paper is on disk** (carried v10
NC-2), so no citation reaches a full-text `supports` verdict; nothing
reached `does-not-support` either.

## The 7 NEW v11 entries (audit priority 1)

All seven were merged from `anvil_pub.bb-workshop.10.litsearch/candidates.bib`.
I diffed each v11 `refs.bib` entry against the litsearch's resolver-verified
entry: the identifier-verified fields (author, title, year, DOI/arXiv ID,
pages) are **unchanged in all seven**; the only deltas are the cosmetic
normalizations the litsearch itself recommended (rekey `ledyard19952` →
`ledyard1995public` with the "2. " De Gruyter title prefix stripped; retype
`@incollection` with Kagel & Roth editors / Princeton UP; retype
`skyrms2003stag` → `@book` with Cambridge UP; arXiv-preprint house style
`@article` + `journal = {arXiv preprint ...}` for the three §6 entries).
The Skyrms 2003-vs-2004 year convention is disclosed in the `refs.bib`
header comment.

| Key | Resolved | Surrounding claim | Verdict | Notes |
|---|---|---|---|---|
| `lanctot2023population` | yes (NEW) | §6: the exploitability line's "benchmark-style move measures populations on repeated rock–paper–scissors, a single fixed game" | partial | Title is a literal match for the claim ("Population-based Evaluation in Repeated Rock-Paper-Scissors as a Benchmark for Multiagent Reinforcement Learning", arXiv 2303.03196, Lanctot et al. 2023). Resolver-verified metadata on disk (litsearch); framing as the tradition's benchmark move is exactly right per auditor knowledge. Primary PDF off-disk |
| `li2024meta` | yes (NEW) | §6: the line's "current methodological statement for deep MARL estimates equilibria *empirically* from simulated meta-games ... rather than reading them off an analytical map" | partial | "A Meta-Game Evaluation Framework for Deep Multiagent RL", Li & Wellman, arXiv 2405.00243. Wellman's EGTA line is precisely empirical equilibrium estimation from simulated meta-games — claim-crediting correct per auditor knowledge; title supports directly. Primary PDF off-disk |
| `christianos2022pareto` | yes (NEW) | §6: "equilibrium *selection* in fixed matrix games whose equilibria are known by inspection, the climbing/penalty-game tradition" | partial | "Pareto Actor-Critic for Equilibrium Selection in Multi-Agent RL", arXiv 2209.14344. The paper is an equilibrium-selection paper in the climbing/penalty-game lineage — a defensible *modern representative* cite. Note the sentence calls this "the oldest form of NE-ground-truth MARL evaluation" while citing a 2022 paper; the actual primary (Claus & Boutilier 1998) was deliberately not cited because it has no resolvable identifier (litsearch write contract; documented in changelog.md). See non-critical note 3 |
| `diekmann1985volunteer` | yes (NEW) | §2 + App. A.6: "the $N$-player Volunteer's Dilemma"; "the textbook single-shot Volunteer's Dilemma" | partial | Diekmann 1985, J. Conflict Resolution 29(4):605–610, DOI 10.1177/0022002785029004003 — the paper that introduced and named the Volunteer's Dilemma (as an N-person game). Exactly the right source for both sites per auditor knowledge. Primary PDF off-disk |
| `ledyard1995public` | yes (NEW) | §2 + App. A.6: "the $N$-player Public Goods game"; "the textbook public-goods underinvestment gradient" | partial | Ledyard 1995, "Public Goods: A Survey of Experimental Research", Handbook of Experimental Economics (Kagel & Roth eds., Princeton UP), pp. 111–194 — the canonical public-goods survey; underinvestment/free-riding is its central subject. Rekey/retype from `ledyard19952` performed exactly as litsearch recommended. Primary PDF off-disk |
| `skyrms2003stag` | yes (NEW) | §2 + App. A.6: "Stag Hunt"; "the canonical Stag Hunt payoff structure" | partial | Skyrms, *The Stag Hunt and the Evolution of Social Structure*, Cambridge UP, DOI 10.1017/CBO9781139165228 — the canonical modern Stag Hunt reference. Year kept at resolver-verified 2003 (print convention 2004); choice documented in refs.bib. Primary PDF off-disk |
| `shapley1953stochastic` | yes (NEW) | App. A.6: "Every formal result for finite stochastic games~\citep{shapley1953stochastic}---minimax, equilibrium existence in stationary mixed strategies, value iteration convergence---applies directly" | partial | Shapley 1953, PNAS 39(10):1095–1100 — the founding stochastic-games paper; correct definitional cite for the class, and the minimax / iterative-solution results are Shapley's own. Precision caveat: *N-player* equilibrium existence in stationary strategies is due to later work (Fink 1964 / Takahashi 1964), so a strict reading of the em-dash list over-attributes one item. Cite position is definitional (immediately after "finite stochastic games"), so not a claim-support failure; see non-critical note 2 |

## Carried keys (11)

| Key | Resolved | Surrounding claim | Verdict | Notes |
|---|---|---|---|---|
| `overcooked2019` | yes | §1/§6/Tab.3: small-grid fully-cooperative; every Pareto-optimal joint policy is a NE, target not unique; 2 agents, 6 actions, 2019 | partial | Point-for-point vs `refs/benchmark_comparison.md`; unchanged from v10 audit; primary PDF off-disk |
| `meltingpot2021` | yes | §1/§6/Tab.3: 88×88×3 RGB, >50 substrates, no per-substrate equilibrium solution; 2–16 agents, 2021 | partial | Matches evidence index; unchanged from v10; off-disk |
| `hanabi2020` | yes | §1/§6/Tab.3: near-optimal at 2P, 4–5P open; 2–5 agents, 2020 | partial | Matches evidence index; unchanged from v10; off-disk |
| `smac2019` | yes | §1/§6/Tab.3: micromanagement vs scripted opponent, NE not a design target; 2019 | partial | Matches evidence index; unchanged; off-disk |
| `smacv2_2023` | yes | same sites as smac2019; 2023 | partial | Matches evidence index; unchanged; off-disk |
| `magent2018` | yes | §1/§6/Tab.3: thousand-agent battles, no published equilibrium analysis at scale; 2018 | partial | Matches evidence index; unchanged; off-disk |
| `pettingzoo2021` | yes | §1/§6/Tab.3: canonical small multi-agent baseline, no equilibrium characterisation; 2021 | partial | Matches evidence index; unchanged; off-disk |
| `ppo2017` | yes | §4 Protocol: "follows the standard recipe of Schulman et al." | partial (metadata resolver-CONFIRMED) | v10's "unverified" upgraded: litsearch resolver-CONFIRMED against live arXiv metadata (1707.06347), zero corrections; refs.bib header updated accordingly. Full-text claim support still off-disk |
| `mappo2022` | yes | §4 Protocol: "MAPPO-style centralised-critic setup" | partial (metadata resolver-CONFIRMED) | arXiv 2103.01955 / NeurIPS 2022 D&B confirmed by litsearch; off-disk |
| `openspiel2019` | yes | §6: OpenSpiel ships many small games + exact solvers | partial (metadata resolver-CONFIRMED) | arXiv 1908.09453 confirmed by litsearch; off-disk |
| `psro2017` | yes | §6: "the PSRO/NashConv line made exploitability the standard convergence-to-equilibrium metric" | partial (metadata resolver-CONFIRMED) | arXiv 1711.00832 / NeurIPS 2017 confirmed by litsearch; claim-crediting correct (PSRO + NashConv both originate there); off-disk |

## Summary

- **Unresolved keys**: 0 (18/18 resolve; refs.bib has no orphans — every
  entry is cited at least once).
- **Claim-support failures**: 0.
- **Partial verifications**: 18 of 18 — 7 benchmark keys against the
  secondary `refs/benchmark_comparison.md` evidence index (carried v10
  NC-2), 11 keys with on-disk resolver-verified metadata provenance
  (`anvil_pub.bb-workshop.10.litsearch/notes.md` provenance table).
- **Fully unverified (no on-disk evidence of any kind)**: 0 — the v10
  audit's 4 unverified keys were resolver-CONFIRMED by the litsearch
  sibling and the confirmation record is on disk.
- **Citations delta vs v10** matches the changelog attestation exactly:
  +7 (the table above), −0; declined candidates `krever2025guard` and
  `papoudakis2020benchmarking` are correctly absent from both refs.bib
  and main.tex; the unverifiable Claus & Boutilier 1998 web lead is
  correctly NOT cited.
