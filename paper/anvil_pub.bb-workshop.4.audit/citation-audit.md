# Citation audit — `anvil_pub.bb-workshop.4`

Per-`\cite{}` resolution + claim-support spot-check for `main.tex`. The
paper uses `\citep` (parenthetical) and `\citet` (textual) from `natbib`;
both bound by `plainnat`. All 11 cite keys resolved against `refs.bib`
on the third pdflatex pass (no `[??]` markers in the rendered PDF).

`refs.bib` is **identical to v3** (no new citations). All v3 citation
verdicts carry over.

## Resolution table

| cite key         | type           | resolves? | well-formed bib? | claim-support verdict                                  | flag           |
|------------------|----------------|-----------|------------------|--------------------------------------------------------|----------------|
| `overcooked2019` | inproceedings  | yes       | yes (NeurIPS '19, arXiv:1910.05789) | supports — canonical Overcooked citation in §1, §5, Table 1 | none |
| `meltingpot2021` | inproceedings  | yes       | yes (ICML '21, arXiv:2107.06857)    | supports — canonical Melting Pot citation; "88×88×3 RGB" claim consistent with refs/benchmark_comparison.md F2 | none |
| `hanabi2020`     | article        | yes       | yes (Artif. Intell. vol 280, arXiv:1902.00506) | supports — canonical Hanabi citation; "near-optimal hat-guessing at 2P, open at 4-5P" consistent with refs/benchmark_comparison.md F3 | none |
| `smac2019`       | inproceedings  | yes       | yes (AAMAS '19, arXiv:1902.04043)   | supports — canonical SMAC citation | none |
| `smacv2_2023`    | inproceedings  | yes       | yes (NeurIPS '23, arXiv:2212.07489) | supports — canonical SMACv2 citation | none |
| `magent2018`     | inproceedings  | yes       | yes (AAAI '18, arXiv:1712.00600)    | supports — canonical MAgent citation | none |
| `pettingzoo2021` | inproceedings  | yes       | yes (NeurIPS '21, arXiv:2009.14471) | supports — canonical PettingZoo MPE citation | none |
| `bbenvspec`      | misc (internal)| yes       | well-formed `@misc`, no arXiv ID    | claim-support: §2's deferred env-spec memo present on disk; supports the citation | **major: internal-memo cite; replace with arXiv ID before submission** |
| `bbnestructure` | misc (internal)| yes       | well-formed `@misc`, no arXiv ID    | claim-support: §3's analytical-NE memo present on disk; supports the surrounding claim | **major: internal-memo cite; replace with arXiv ID before submission** |
| `ppo2017`       | article        | yes       | yes (Schulman et al. 2017, arXiv:1707.06347) | supports — `\citet{ppo2017}` introduces PPO | none |
| `mappo2022`     | inproceedings  | yes       | yes (Yu et al. 2022, NeurIPS, arXiv:2103.01955) | supports — `\citet{mappo2022}` for the MAPPO-style centralised-critic setup of `JointPPOTrainer` | none |

## Notes

- **All citations resolve.** Final-pass pdflatex log contains zero `[??]`
  markers; the `Package natbib Warning: There were undefined citations`
  line in pass-1 is the expected pre-bibtex state and disappears in
  passes 2 and 3.
- **No new cite keys in v4.** The v3 `7×4` table-dimensionality nit
  (carried into v3 as M3) is no longer present in v4: the sentence at
  v4 main.tex L313–314 reads "the per-cell predicted-vs-empirical table
  and the full bias accounting appear in [bbnestructure]" without the
  "$7\times 4$" descriptor. v3 M3 is therefore resolved by v4.
- **Internal-memo cites are deliberate placeholders.** `refs.bib` header
  and the body of each `@misc` entry flag these as "to be released as
  arXiv preprint before submission." Flagged as `major` (a workshop
  submission needs externally resolvable identifiers) but NOT critical —
  replacement is a known followup tracked at the thread root.
- **Claim-support verification scope.** Source PDFs for the seven
  surveyed-benchmark citations are not on disk; verdict above is
  "claim-support consistent with the on-disk synthesis in
  refs/benchmark_comparison.md and well-formed against the BibTeX
  year + venue". For the two internal-memo cites the on-disk memos
  were inspected and support the surrounding-sentence claims.
- **No obvious mismatch.** Every cite is used in a context consistent
  with the cited paper's well-known subject matter.
