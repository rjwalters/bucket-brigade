# Citation audit for `anvil_pub.bb-workshop.6`

## Resolution summary

- **Total `\cite{}` keys in main.tex:** 9 (7 `\citep`, 2 `\citet`)
- **Total `@` entries in refs.bib:** 9
- **Unresolved citations after bibtex + 2× pdflatex:** 0
- **Stale keys (cited but not in bib, or in bib but not cited):** 0

| Citation key            | First use site                            | refs.bib entry kind | Resolution |
| ----------------------- | ----------------------------------------- | ------------------- | ---------- |
| `overcooked2019`        | §1 intro, §5 Table 2                      | `@inproceedings`    | resolved   |
| `meltingpot2021`        | §1 intro, §5 Table 2                      | `@inproceedings`    | resolved   |
| `hanabi2020`            | §1 intro, §5 Table 2                      | `@article`          | resolved   |
| `smac2019`              | §1 intro, §5 Table 2 (joint cite w/ v2)   | `@inproceedings`    | resolved   |
| `smacv2_2023`           | §1 intro, §5 Table 2 (joint cite w/ v1)   | `@inproceedings`    | resolved   |
| `magent2018`            | §1 intro, §5 Table 2                      | `@inproceedings`    | resolved   |
| `pettingzoo2021`        | §1 intro, §5 Table 2                      | `@inproceedings`    | resolved   |
| `ppo2017`               | §4 Protocol                               | `@article`          | resolved   |
| `mappo2022`             | §4 Protocol                               | `@inproceedings`    | resolved   |

`bibtex main.blg` reports 0 errors and 0 warnings (no missing keys, no duplicate keys, no missing fields).

## Spot-check: cited claim vs. source claim

| Citation | Paper claim at cite site | Plausible support? |
| -------- | ------------------------ | ------------------ |
| `overcooked2019` | "fully cooperative coordination game on small grid layouts" | yes — matches Carroll et al. Overcooked-AI canonical pitch |
| `meltingpot2021` | "exposes 88×88×3 RGB observations across more than 50 substrates" | yes — matches Leibo et al. Melting Pot v2 specification |
| `hanabi2020`     | "near-optimal hat-guessing strategies known at 2 players, 4–5 player open" | yes — Bard et al. Hanabi Challenge canonical result |
| `smac2019/smacv2_2023` | "cooperative micromanagement against a scripted opponent" | yes — both SMAC and SMACv2 pitch |
| `magent2018`     | "thousand-agent battles on grids" | yes — matches Zheng et al. MAgent design |
| `pettingzoo2021` | "canonical small multi-agent baseline" | yes — Terry et al. PettingZoo MPE positioning |
| `ppo2017`        | "PPO implementation follows the standard recipe" | yes — Schulman et al. PPO original paper |
| `mappo2022`      | "multi-agent joint-policy formulation follows MAPPO-style centralised-critic setup" | yes — Yu et al. MAPPO paper |

No load-bearing cite-claim mismatches detected. The Related Work section's six-benchmark survey table (§5 Table 2) is a structural restatement of each benchmark's canonical pitch; none of the claims overstate what the cited papers carry.

## Carry-forward from v5

v5 audit confirmed `bbenvspec` and `bbnestructure` placeholder cites were removed in favor of Appendix A and Appendix B `\ref{}` cross-references. v6 inherits this resolved state. No new citations introduced in v6 (changelog explicitly says "refs.bib is unchanged").

## Final verdict

**Citation health: clean.** Zero unresolved keys, zero stale keys, zero plausible claim/source mismatches.
