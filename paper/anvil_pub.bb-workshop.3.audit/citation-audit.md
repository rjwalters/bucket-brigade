# Citation audit — `anvil_pub.bb-workshop.3`

Per-`\cite{}` resolution + claim-support spot-check for `main.tex`. The
paper uses `\citep` (parenthetical) and `\citet` (textual) from `natbib`;
both bound by `plainnat`. All 11 cite keys resolved against `refs.bib`
on the third pdflatex pass (no `[??]` markers in the rendered PDF).

## Resolution table

| cite key         | type           | resolves? | well-formed bib? | claim-support verdict                                  | flag           |
|------------------|----------------|-----------|------------------|--------------------------------------------------------|----------------|
| `overcooked2019` | inproceedings  | yes       | yes (NeurIPS '19, arXiv:1910.05789) | supports — used as canonical Overcooked citation in §1, §5, Table 1 | none |
| `meltingpot2021` | inproceedings  | yes       | yes (ICML '21, arXiv:2107.06857)    | supports — used as canonical Melting Pot citation in §1, §5, Table 1; "88×88×3 RGB" claim is consistent with refs/benchmark_comparison.md Finding 2 | none |
| `hanabi2020`     | article        | yes       | yes (Artif. Intell. vol 280, arXiv:1902.00506) | supports — used as canonical Hanabi citation; "near-optimal hat-guessing at 2P, open at 4-5P" is consistent with refs/benchmark_comparison.md Finding 3 | none |
| `smac2019`       | inproceedings  | yes       | yes (AAMAS '19, arXiv:1902.04043)   | supports — used as canonical SMAC citation; "cooperative micromanagement vs. scripted opponent" is consistent with refs/benchmark_comparison.md Finding 3 | none |
| `smacv2_2023`    | inproceedings  | yes       | yes (NeurIPS '23, arXiv:2212.07489) | supports — used as canonical SMACv2 citation; same as smac2019 framing | none |
| `magent2018`     | inproceedings  | yes       | yes (AAAI '18, arXiv:1712.00600)    | supports — used as canonical MAgent citation; "thousand-agent battles" matches Finding 2 | none |
| `pettingzoo2021` | inproceedings  | yes       | yes (NeurIPS '21, arXiv:2009.14471) | supports — used as canonical PettingZoo MPE citation; "small multi-agent baseline, no equilibrium characterization" matches Finding 3 | none |
| `bbenvspec`      | misc (internal)| yes       | well-formed `@misc`, but internal-memo (no arXiv ID) | claim-support: §2's deferred "formal proofs and a complete notation table" lives in `paper/anvil_memo.env_spec.1/env_spec.md` — present on disk and supports the citation | **major: internal-memo cite; replace with arXiv ID before submission** (followup, not blocking) |
| `bbnestructure` | misc (internal)| yes       | well-formed `@misc`, but internal-memo (no arXiv ID) | claim-support: §3's "$7\times 4$ predicted-vs-empirical table and full bias accounting" lives in `paper/anvil_memo.ne_structure.1/ne_structure.md` — present on disk and supports the surrounding claim. **Minor note**: paper says "$7\times 4$" but the memo's actual table is 7 rows × 6 columns (β, κ, Empirical, Predicted, Match?, Notes). Dimensionality is dressing; the table content is the load-bearing referent. | **major: internal-memo cite; replace with arXiv ID before submission** (followup). **Minor: $7\times 4$ vs 7×6 mismatch in description.** |
| `ppo2017`       | article        | yes       | yes (Schulman et al. 2017, arXiv:1707.06347) | supports — `\citet{ppo2017}` introduces PPO; canonical citation for the algorithm | none |
| `mappo2022`     | inproceedings  | yes       | yes (Yu et al. 2022, NeurIPS, arXiv:2103.01955) | supports — `\citet{mappo2022}` for the MAPPO-style centralised-critic setup of `JointPPOTrainer`; canonical citation for MAPPO | none |

## Notes

- **All citations resolve.** Final-pass pdflatex output contains zero
  `[??]` markers (verified by grep on `/tmp/audit_compile_3.log`); the
  `Package natbib Warning: There were undefined citations` line in the
  pass-1 log is the expected pre-bibtex state and disappears in passes
  2 and 3.
- **Internal-memo cites are deliberate placeholders.** `refs.bib`
  header (lines 6–11) and the body of each `@misc` entry both flag
  these as "to be released as arXiv preprint before submission". The
  auditor flags them as `major` (a workshop submission needs externally
  resolvable identifiers) but does NOT mark them critical — replacement
  is a known followup tracked at the thread root, not a fact-check
  failure on the current draft.
- **Claim-support verification scope.** Source PDFs for the seven
  surveyed-benchmark citations are not on disk; the auditor cannot
  open the actual Overcooked / Melting Pot / Hanabi / SMAC / MAgent /
  PettingZoo / PPO / MAPPO PDFs to verify each surrounding-sentence
  claim against the primary source. The verdict column above is
  "claim-support unverified against primary PDF; consistent with the
  on-disk synthesis in refs/benchmark_comparison.md and well-formed
  against the BibTeX year + venue". For the two internal-memo cites
  (`bbenvspec`, `bbnestructure`), the source is on disk and the
  surrounding-sentence claim was verified.
- **No obvious mismatch.** Every cite is used in a context consistent
  with the cited paper's well-known subject matter (e.g., `overcooked2019`
  appears next to "Overcooked-AI is a fully cooperative coordination
  game", not next to a claim about Hanabi).
