# Citation audit for anvil_pub.bb-workshop.10

Every `\citep{}` / `\citet{}` in `main.tex` was enumerated (11 distinct keys,
10 cite commands, 22 total citations) and checked against `refs.bib`.
**All 11 keys resolve; no orphan entries remain in refs.bib.** Claim support
was assessed against the author-supplied sources in
`paper/anvil_pub.bb-workshop/refs/` (the `benchmark_comparison.md` evidence
index is the only on-disk source covering the cited papers; no primary PDFs
are on disk).

| Key | Resolved | Surrounding claim | Verdict | Notes |
|---|---|---|---|---|
| `overcooked2019` | yes | §1/§6/Tab.3: small-grid fully-cooperative coordination; every Pareto-optimal joint policy is a NE, target not unique; 2 agents, 6 actions, 2019 | partial | Point-for-point match with `refs/benchmark_comparison.md` Findings 1–4 + evidence index; primary PDF not on disk |
| `meltingpot2021` | yes | §1/§6/Tab.3: 88×88×3 RGB, >50 substrates, design does not solve equilibrium structure; 2–16 agents, 2021 | partial | Matches evidence index [2]; primary PDF not on disk |
| `hanabi2020` | yes | §1/§6/Tab.3: near-optimal hat-guessing at 2P, 4–5P open, no learning algorithm reaches equilibrium; 2–5 agents, 2020 | partial | Matches evidence index [3]; primary PDF not on disk |
| `smac2019` | yes | §1/§6/Tab.3: cooperative micromanagement vs scripted opponent, NE not a design target; 2019 | partial | Matches evidence index [4]; primary PDF not on disk |
| `smacv2_2023` | yes | same sites as smac2019; 2023 | partial | Matches evidence index [4]; primary PDF not on disk |
| `magent2018` | yes | §1/§6/Tab.3: thousand-agent grid battles, no published equilibrium analysis at scale; 2018 | partial | Matches evidence index [5]; primary PDF not on disk |
| `pettingzoo2021` | yes | §1/§6/Tab.3: canonical small multi-agent baseline, no equilibrium characterisation across scenarios; 2021 | partial | Matches evidence index [6]; booktitle now names the NeurIPS Datasets & Benchmarks track (v9 NC-3 resolved); primary PDF not on disk |
| `ppo2017` | yes | §4 Protocol: "The PPO implementation follows the standard recipe of Schulman et al." | unverified — source not on disk | Canonically correct reference for the claim; metadata (arXiv 1707.06347, 2017, author list) matches the auditor's knowledge of the paper |
| `mappo2022` | yes | §4 Protocol: "the multi-agent joint-policy formulation follows the MAPPO-style centralised-critic setup" | unverified — source not on disk | Canonically correct; arXiv 2103.01955, NeurIPS 2022 Datasets & Benchmarks (track now named per v9 NC-3); metadata consistent |
| `openspiel2019` | yes (NEW in v10) | §6 Related work: "OpenSpiel ships dozens of small normal-form and extensive-form games together with exact solvers, so 'did the learner reach equilibrium?' is answerable there too" | unverified — source not on disk | Metadata internally consistent and matches the canonical paper: arXiv 1908.09453 (2019), "OpenSpiel: A Framework for Reinforcement Learning in Games", Lanctot et al. (27-author DeepMind list as published); correctly cited as an arXiv preprint (it has no conference venue). The surrounding claim (many small games + exact equilibrium solvers) is accurate per the auditor's knowledge. Nothing in the entry looks fabricated |
| `psro2017` | yes (NEW in v10) | §6 Related work: "the PSRO/NashConv line made exploitability the standard convergence-to-equilibrium metric for learning in games" | unverified — source not on disk | Metadata internally consistent and matches the canonical paper: "A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning", Lanctot, Zambaldi, Gruslys, Lazaridou, Tuyls, Pérolat, Silver, Graepel, NeurIPS 2017, arXiv 1711.00832 — the paper that introduced PSRO and the NashConv exploitability metric, so the claim-crediting is exactly right. Nothing in the entry looks fabricated |

## Summary

- **Unresolved keys**: 0.
- **Claim-support failures**: 0.
- **Partial (secondary-source-only) verifications**: 7 citation sites over 6
  benchmark keys — verified point-for-point against
  `refs/benchmark_comparison.md`, but the primary PDFs are not in
  `<thread>/refs/` (carried from v9 NC-2, unchanged).
- **Unverified (no source on disk)**: 4 keys (`ppo2017`, `mappo2022`,
  `openspiel2019`, `psro2017`). The two NEW v10 entries were specifically
  re-checked for internal consistency at the reviser's request: titles,
  author lists, years, venues, and arXiv IDs all match the canonical papers
  per the auditor's knowledge (no web fetch performed); neither looks
  fabricated or mis-attributed. Off-disk verification remains an author /
  `pub-litsearch` task.
- The `refs.bib` header comment still self-declares its IDs as
  not-yet-auditor-verified and asks for a re-check — this audit is that
  re-check at knowledge level; primary-source acquisition is still open.
