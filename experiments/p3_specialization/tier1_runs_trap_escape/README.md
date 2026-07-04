# Trap-escape ladder rung 1: budget scaling (issue #444)

Budget-scaled re-runs of the tier-1 `rest_trap` cells (het_ppo + ippo),
testing whether the at-random plateau documented in
`../tier1_runs/tier1_verdict.md` and `docs/PAPER_RESULTS.md` §7 is a
training-budget artifact.

## Layout

**Budget is encoded in the root path, not the cell name.**
`run_tier1_cell.py` hardcodes cell directories as `<trainer>_<scenario>`,
and a budget-suffixed trainer name would break `aggregate_tier1.py`'s
dir-name parse and make budgets indistinguishable in a combined root. Each
budget therefore gets its own root with the standard cell layout and its
own independently aggregated verdict table:

```
4x/    200 iterations × 2048 rollout steps   (tier-1 standard is 50 × 2048)
  het_ppo_rest_trap/   seed_42 … seed_61 + cell_summary.json
  ippo_rest_trap/      seed_42 … seed_61 + cell_summary.json
  tier1_verdict.{md,json} + tier1_verdict_notes.md
16x/   800 iterations × 2048 rollout steps
  (same structure)
```

Provenance: host alc-2, train commit `ed0555af`, 20 seeds (42–61) per
cell. The 16× cells were trained as two concurrent 10-seed halves writing
into the same cell dirs; the committed `cell_summary.json` files were
rebuilt over all 20 seeds via `--summarize-only` (see per-root notes).
Launch logs (`*.log`, gitignored) remain on alc-2.

## Result (full caveats in the per-root `tier1_verdict_notes.md`)

| root | het_ppo | ippo |
|---|---|---|
| `4x/` | `at_random` (CI lo 304.03 vs anchor 304.31) | `at_random` |
| `16x/` | **`escaped_trap`** (307.83, CI [305.00, 310.71]) | `at_random` |

The 16× het_ppo escape is marginal and variance-driven — the mean is flat
across budgets (~ +5/step over random, ≈ 6% of the 83.7/step scripted
headroom) — but it satisfies the pre-registered #444 stopping rule, so
rung 1 terminates the ladder. See `docs/PAPER_RESULTS.md` §9.

## Regenerate summaries + verdicts (no training, seconds locally)

```bash
for spec in "4x:200" "16x:800"; do
  root=${spec%%:*}; iters=${spec##*:}
  for tr in het_ppo ippo; do
    uv run python experiments/p3_specialization/run_tier1_cell.py \
      --trainer $tr --scenario rest_trap \
      --seeds 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 \
      --num-iterations $iters --summarize-only \
      --output-root experiments/p3_specialization/tier1_runs_trap_escape/$root
  done
  uv run python experiments/p3_specialization/aggregate_tier1.py \
    --tier1-root experiments/p3_specialization/tier1_runs_trap_escape/$root
done
```
