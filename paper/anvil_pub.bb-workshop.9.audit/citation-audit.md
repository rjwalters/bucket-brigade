# Citation audit — anvil_pub.bb-workshop.9

Scope: every `\citep{}` / `\citet{}` in `main.tex` (20 cite commands, 23 key
references, 9 unique keys) resolved against `refs.bib`, with claim-support
spot-checks against the author-supplied sources in
`paper/anvil_pub.bb-workshop/refs/`.

**Resolution rate: 9/9 unique keys (100%). No unresolved citations.**
The final compiled PDF contains zero `[??]` markers (see `compile-log.txt`).

Claim-support materials on disk: `refs/benchmark_comparison.md` (the
anvil_report evidence-index the six benchmark rows were built from),
`refs/env_spec.md`, `refs/ne_structure.md`, `refs/phase_diagram_table.md`,
`refs/phase_diagram.png`, `refs/_project.md`. **No primary paper PDFs are on
disk**, so claim support for external papers is checked only against the
author-supplied secondary notes; verdicts are `partial` where the notes
corroborate and `unverified` where nothing on disk speaks to the claim.

| Key | Resolved | Cite sites (line) | Surrounding claim | Verdict | Notes |
|---|---|---|---|---|---|
| `overcooked2019` | yes | 155, 1072, 1098 | Fully cooperative coordination game on small grid layouts; every Pareto-optimal joint policy is a NE so "the" target is not unique; 2 agents, 6 actions, 2019 | partial | Supported point-for-point by `refs/benchmark_comparison.md` (Findings 3, 4; comparison table row; action-size note "6-action space per the original Carroll et al. setup"). Primary PDF not on disk. |
| `meltingpot2021` | yes | 156, 1075, 1099 | 88×88×3 RGB obs across >50 substrates; equilibrium structure not solved / not a design target; 2–16 agents, 8 actions, mixed, 2021 | partial | Matches notes Findings 2–4 and table row. Primary PDF not on disk. |
| `hanabi2020` | yes | 157, 1078, 1100 | Near-optimal hat-guessing known at 2P; 4–5P open, no algorithm reaches equilibrium from learning; 2–5 agents, ≤20 actions, coop, 2020 | partial | Matches notes Finding 3 ("hat-guessing near-optimal at 2P... 4-5P game is open") and table row. Primary PDF not on disk. |
| `smac2019` | yes | 157, 1080, 1101 | Cooperative micromanagement vs scripted opponent; NE not a design target; 2–27 agents, 6+n_e actions, 2019 | partial | Matches notes Finding 3/4 and table row. Primary PDF not on disk. |
| `smacv2_2023` | yes | 157, 1080, 1101 | Same claims as SMAC, 2023 refresh | partial | Notes reference [4b] Ellis et al. 2023. Primary PDF not on disk. |
| `magent2018` | yes | 158, 1082, 1102 | Thousand-agent battles on grids; no published equilibrium analysis at scale; 64–10³ agents, 21 actions, mixed, 2018 | partial | Matches notes Finding 3/5 and table row (64–1000+, "21 (move 13 + attack 8)"). Primary PDF not on disk. |
| `pettingzoo2021` | yes | 159, 1084, 1103 | Canonical small multi-agent baseline (MPE); no equilibrium characterisation across scenarios; 2–6 agents, 3–5 actions, 2021 | partial | Matches notes Finding 3/6 and table row. Primary PDF not on disk. |
| `ppo2017` | yes | 562 | "The PPO implementation follows the standard recipe of Schulman et al." | unverified — source not on disk | No PPO source material in `refs/`. Bib entry (arXiv 1707.06347, Schulman et al. 2017) is the canonically correct reference for this claim; author should verify off-disk. |
| `mappo2022` | yes | 563 | "the multi-agent joint-policy formulation follows the MAPPO-style centralised-critic setup of Yu et al." | unverified — source not on disk | No source in `refs/`. Bib entry (arXiv 2103.01955, Yu et al., NeurIPS 2022) is the canonically correct reference; author should verify off-disk. |

## BibTeX entry integrity

`refs.bib` self-declares (header comment) that entries "are not yet
auditor-verified; the pub-audit phase should re-check arXiv IDs and venue
names before submission." Checks performed:

- **arXiv IDs** — all nine eprint fields match the auditor's knowledge of the
  canonical papers: 1910.05789 (Overcooked), 2107.06857 (Melting Pot),
  1902.00506 (Hanabi), 1902.04043 (SMAC), 2212.07489 (SMACv2), 1712.00600
  (MAgent), 2009.14471 (PettingZoo), 1707.06347 (PPO), 2103.01955 (MAPPO).
  No mismatch found. (Web verification not performed; this is a
  knowledge-based check.)
- **Venues** — plausible and standard: NeurIPS (overcooked2019, smacv2_2023,
  pettingzoo2021, mappo2022), ICML (meltingpot2021), AAMAS (smac2019), AAAI
  (magent2018), Artificial Intelligence vol. 280 (hanabi2020), arXiv preprint
  (ppo2017). Non-critical note: `pettingzoo2021` and `mappo2022` appeared in
  the NeurIPS **Datasets and Benchmarks** track; the entries use the plain
  "Advances in Neural Information Processing Systems" string. Acceptable
  shorthand, but the camera-ready may want the track named.
- **No orphan entries**: every `refs.bib` key is cited at least once; every
  cited key exists. bibtex reported no errors (see `compile-log.txt`).

## Summary

- Unresolved citations: **0**
- Claim-support failures (`does-not-support`): **0**
- `partial` (supported by author-supplied secondary notes only): **7 keys**
  (the six benchmark keys + smacv2 counted within them; primary PDFs absent)
- `unverified — source not on disk`: **2 keys** (`ppo2017`, `mappo2022`)
