# Citation audit for `anvil_pub.bb-workshop.5`

## Summary

- `\cite{}` keys used in `main.tex`: **9 unique**
- Bib entries in `refs.bib`: **9** (one-to-one, after v5 deletion of `bbenvspec` and `bbnestructure`)
- Unresolved citations in compiled PDF: **0**
- Leftover `\citep`/`\citet` to deleted memo keys: **0** (only string mentions in comments)
- Stray `\citep{diekmann1985}` from changelog item 2: **not present** (cleanly removed)

## Per-`\cite{}` resolution table

| Key            | Used in (§ / line)                          | Surrounding claim                                                              | Bib entry resolves? | Claim-vs-source check                                                                                                                                                            |
|----------------|---------------------------------------------|--------------------------------------------------------------------------------|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| overcooked2019 | §1 intro L86; §5 L584; Table 2 L584         | "Overcooked-AI… fully cooperative coordination game on small grid layouts"     | Yes (NeurIPS 2019)  | Consistent — the cited paper introduces the Overcooked-AI environment and the human-AI coordination axis. Matches \S5's "every Pareto-optimal joint policy is a NE" framing.    |
| meltingpot2021 | §1 intro L86; §5 L564; Table 2 L585         | "Melting Pot… 88×88×3 RGB observations across more than 50 substrates"         | Yes (ICML 2021)     | Consistent — Melting Pot's substrate count and observation format are accurately summarised. NE-non-characterisability claim is a fair editorial framing.                       |
| hanabi2020     | §1 intro L87; §5 L566; Table 2 L586         | "hat-guessing strategies known at 2 players; 4-5 player game is open"          | Yes (AIJ 2020)      | Consistent — Hanabi paper exists, this is the canonical benchmark cite. NE-status framing matches the literature.                                                                |
| smac2019       | §1 intro L87; §5 L567; Table 2 L587         | "cooperative micromanagement against a scripted opponent"                      | Yes (AAMAS 2019)    | Consistent — SMAC paper is the right reference for the scripted-opponent setup.                                                                                                  |
| smacv2_2023    | §1 intro L87; §5 L567; Table 2 L587         | same paragraph as smac2019                                                     | Yes (NeurIPS 2023)  | Consistent — companion citation to smac2019 for the v2 environment.                                                                                                              |
| magent2018     | §1 intro L88; §5 L569; Table 2 L588         | "thousand-agent battles on grids; no published equilibrium analysis at scale"  | Yes (AAAI 2018)     | Consistent — MAgent paper is the right reference for the population-scale battle setup.                                                                                          |
| pettingzoo2021 | §1 intro L89; §5 L571; Table 2 L589         | "canonical small multi-agent baseline with no equilibrium characterisation"    | Yes (NeurIPS 2021)  | Consistent — PettingZoo paper is the right reference; MPE was inherited from the original Lowe et al. MADDPG suite but PettingZoo is now the canonical packaging.                |
| ppo2017        | §4 L431 (`\citet{ppo2017}`)                 | "The PPO implementation follows the standard recipe of \citet{ppo2017}"        | Yes (Schulman et al., arXiv) | Consistent — canonical Schulman PPO citation, used to anchor the PPO recipe choice in §4.                                                                                |
| mappo2022      | §4 L432-433 (`\citet{mappo2022}`)           | "the multi-agent joint-policy formulation follows the MAPPO-style centralised-critic setup of \citet{mappo2022}" | Yes (Yu et al., NeurIPS) | Consistent — MAPPO is the right reference for the centralised-critic-style joint PPO.                                                                                  |

## Verifications driven by the v4 audit and v5 changelog

### v4 M1 (internal-memo cites) carry-forward: RESOLVED in v5

v4 flagged `bbenvspec` and `bbnestructure` as `@misc` placeholders for in-repo memos
that would not exist on arXiv by submission. v5 deletes both bib entries and inlines
the memo content as Appendix A and Appendix B. The five v4 citation sites
(L180, L318, L330, L359, L377) now point at `\ref{app:envspec}` / `\ref{app:nestructure}`.
Verified by `grep` for `bbenvspec`/`bbnestructure` in `main.tex`: only matches are
in the file-header comment (line 3) and in `refs.bib` line 81 (also a comment).
No `\cite{}` or `\citep{}` to the removed keys remains.

### Changelog item 2's "stray `\citep{diekmann1985}`" mention: RESOLVED

The changelog notes a `\citep{diekmann1985}` was introduced during the appendix
draft and then removed. `grep -nE "diekmann1985" main.tex refs.bib` returns
zero matches. Volunteer's-Dilemma paragraph (Appendix A.6) opens with an
informal named attribution only, as the changelog claims.

### `\ref{}` resolution

Every `\ref{}` and `\eqref{}` in the body resolves to its label without
`[??]` markers in the compiled PDF. Both new appendix labels (`app:envspec`,
`app:nestructure`) and their five referenced sub-labels (e.g.,
`app:envspec:variants`, `app:nestructure:gaps`) are defined and reached.

## Verdict

Citation health: **clean.** All 9 unique `\cite{}` keys resolve to bib entries
whose surrounding-claim verdicts are consistent with the cited work. v4's
M1 (internal-memo cites) is fully resolved by the v5 appendix inlining.
No new citation issues introduced.
