# Line-level comments: anvil_report.benchmark_comparison v1

(Polish-only; no blockers.)

## Exec summary, paragraph 2

"At its minimal `v2_minimal` parameterization (2 houses, 4 agents) the per-agent action space is `MultiDiscrete([num_houses, 2, 2])` = 8 actions [Finding 1]" — the inline citation `[Finding 1]` works but is unusual in customer-facing prose. A future revision could replace with "(Finding 1)" for visual consistency with other parenthetical references.

## Finding 2

"Bucket Brigade state at one timestep is `(house_states, agent_positions, agent_signals)`" — minor: the actual code also tracks `last_actions` and `night` and `done`, but these are correctly excluded from the state-cardinality calculation because they are redundant (last_actions) or trivial (night counter, done flag). The exclusion is correct; an audit-pass-with-no-flags requires acknowledging the choice if challenged.

## Comparison table

The table is dense but rendering is fine. Two suggestions for future revision:
- Add a one-line legend ("OoM = order of magnitude per Google Scholar; TBD = pre-publication; intractable = not enumerable at typical configuration").
- Consider splitting "State space" into two columns: "per-agent observation size" vs. "global state cardinality". Some benchmarks have very different numbers in each (Melting Pot has 88×88×3 per agent but the substrate state is much larger). Non-blocking; the current single column is defensible.

## Recommendations 1–3

Each names the downstream owner (issue #364 paper draft). Verify that #364's curator has read these recommendations before claiming. If #364 ships without addressing Recommendation 1's "first-three-paragraphs" criterion, the paper section will fight the comparison table.

## Risks and limitations, "Data limits"

The OoM citation count caveat is honest. Strong suggestion: when the paper goes to submission, refetch Scholar counts for all 8 benchmarks on the day of submission, record under `refs/citations-YYYY-MM-DD.md`, and update the comparison table OoM column with precise counts. This is for the paper, not for this report.
