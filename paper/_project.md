---
recipient: "Bucket Brigade workshop paper readership (cooperative-AI / MARL community)"
engagement_id: "BB-paper-2026-M3"
delivery_format: "markdown"
confidentiality_class: "public"
prior_reports: []
voice_notes: "Technical, MARL-literate audience. Concise. No marketing tone. Honest about Bucket Brigade's narrowness — the value proposition is NE-transparency, not richness."
---

# Engagement: Bucket Brigade — workshop paper writeup track

## Recipient context

Audience is the cooperative-AI / MARL research community, primarily reviewers and readers at AAMAS, NeurIPS Datasets & Benchmarks, and the Cooperative AI workshop. They already know Overcooked, MeltingPot, Hanabi, SMAC, and MAgent. Pitching to them requires answering exactly one question: **why add another environment?** Anything that sounds like marketing kills the paper.

## Engagement scope

These `anvil:report` artifacts feed paper sections, not deliverables. The `benchmark_comparison` thread (issue #363) produces the §"Related benchmarks" section: a 6-row comparison table, honest positioning, and clear statement of the niche Bucket Brigade fills (NE-transparency at a tiny parametric state-space) that none of the existing benchmarks fill.

In scope:
- Per-benchmark facts (state size, action size, agent count, NE structure, cooperative/competitive structure, year, citation proxy)
- Honest treatment of where Bucket Brigade is weaker (artificial, tiny, custom)
- Honest treatment of where it is stronger (NE-computable, parametric, ablation-designed)

Out of scope:
- Full literature survey beyond the named six benchmarks
- Empirical re-runs of those benchmarks (citation-based comparison only)
- Defending Bucket Brigade as a "general-purpose" benchmark — it isn't, and the report says so

## Communication norms

Format: markdown source-of-truth, no rendered PDF required for this internal paper-section deliverable. Tables must render in plain markdown. Citations use standard arXiv/conference shorthand.

## Notes on prior reports

None yet — this is the first artifact in the paper's anvil track.
