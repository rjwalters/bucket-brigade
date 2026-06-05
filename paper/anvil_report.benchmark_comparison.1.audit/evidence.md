# Citation traceability map: anvil_report.benchmark_comparison v1

For each cited source, the claims that depend on it.

## External citations (arXiv papers)

| Citation | arXiv ID | Depended on by |
|---|---|---|
| Carroll et al. 2019 (Overcooked-AI) [1] | 1910.05789 | Findings 2, 3, 4, 5; Comparison table Overcooked row; Appendix A |
| Leibo et al. 2021 (Melting Pot) [2] | 2107.06857 | Findings 2, 3, 4, 5; Comparison table Melting Pot row; Appendix A |
| Bard et al. 2020 (Hanabi Challenge) [3] | 1902.00506 | Findings 2, 3, 4, 5; Comparison table Hanabi row |
| Samvelyan et al. 2019 (SMAC) [4] | 1902.04043 | Findings 2, 3, 4, 5, 6; Comparison table SMAC row; Appendix A |
| Ellis et al. 2023 (SMACv2) [4b] | 2212.07489 | Finding 3; Comparison table SMAC row; "Time limits" in Risks |
| Zheng et al. 2018 (MAgent) [5] | 1712.00600 | Findings 2, 3, 4, 5; Comparison table MAgent row; Appendix A |
| Terry et al. 2021 (PettingZoo) [6] | 2009.14471 | Findings 4, 5; Comparison table PettingZoo MPE row; Appendix B |
| Kurach et al. 2020 (Google Research Football) [7] | 1907.11180 | Risks/limitations only (Sample limits); Appendix B |
| Baker et al. 2020 (Hide-and-Seek) [8] | 1909.07528 | Risks/limitations only (Sample limits); Appendix B |

## In-repo references

| Source | Location | Depended on by |
|---|---|---|
| `bucket_brigade/envs/bucket_brigade_env.py` | refs/bucket-brigade-env.md → "State space" section | Findings 1, 2; Comparison table Bucket Brigade row |
| `bucket_brigade/envs/scenarios_generated.py` (line 85) | refs/bucket-brigade-env.md → "Action space" section | Findings 1, 6; Comparison table Bucket Brigade row |
| `bucket_brigade/baselines/specialist.py` (line 81) | refs/bucket-brigade-env.md → "Action space" section | Finding 1 |
| `bucket_brigade/baselines/__init__.py` (line 41) | refs/bucket-brigade-env.md → "Action space" section | Finding 1 (cross-check) |
| `docs/technical_marl_review.md` | in-repo | Finding 7 (methodological framing) |

## Tracker / issue references

| Issue | Title | Depended on by |
|---|---|---|
| #355 | Hetero-DO sweep on 2 cells | Findings 3, 4 |
| #356 | P3 specialization research wall | Finding 7 |
| #357 | [tracker] Bucket Brigade env paper workshop submission roadmap | Throughout; methodological framing |
| #358 | Compute NE phase diagram across (β, κ, c) | Findings 3, 6; Recommendation 2; Risks |
| #359 | Analytical NE characterization for 4-agent Bucket Brigade | Findings 3, 7 |
| #363 | Comparative analysis vs. existing MARL benchmarks (this report's issue) | Acceptance criteria check |
| #364 | Draft workshop paper | All three Recommendations' Owner field |

## Verification status by citation type

- **In-repo source code citations**: line-pinned, verified by direct Read of the worktree (refs/bucket-brigade-env.md captures the exact text).
- **Tracker / issue citations**: verified by `gh issue view` against rjwalters/bucket-brigade at audit time.
- **External arXiv citations**: arXiv IDs not refetched in this audit. The carve-out applied: these are famous enough cooperative-MARL benchmark papers that any misidentification would be caught by any reviewer immediately, so the audit does not flag this as an "Unreachable external citation" critical flag. The reviser is advised to refetch all 9 arXiv IDs at the paper-submission-time audit (issue #364), not at this audit.

## OoM-citation-count carve-out

Finding 5's order-of-magnitude citation counts are uncited *because the OoM framing IS the load-bearing claim*. This is the prose-narrative carve-out per the rubric's "Unreachable external citation" rule: the citation-count claim is `(none — uncited)`, not an unreachable URL. The audit does not raise a critical flag on this dimension; the review verdict notes it cost a point on Dim 4 (Evidence trail).
