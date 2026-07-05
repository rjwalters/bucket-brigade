# Notes — recalibrated_verdict.{json,md} (v1, n=4 sweep)

## Regenerated for reproducibility (issue #461)

The committed `recalibrated_verdict.{json,md}` were originally generated
before the issue #416 re-derivation of `MINSPEC_SPECIALIST` (−22.07 n=50 →
−28.38 n=10k) landed in `bucket_brigade/baselines`, so re-running
`experiments/scripts/recalibrate_phase_diagram_ppo.py` with its defaults
produced a diff (every row's `old_baseline_specialist` metadata field, plus
the md header wording). Issue #461 regenerated both artifacts from the
committed per-seed `cell_*/seed_*/summary.json` files using the current
script — a pure local re-aggregation, no training re-runs. The artifacts now
reproduce byte-for-byte from the script (verified by running it twice and
`cmp`-ing).

**Nothing numeric changed.** The regeneration touched only:

- `old_baseline_specialist`: −22.07 → −28.38 (metadata stamp of the
  *current* canonical constant, see caveat below), and
- the md header paragraph, which now spells out the #416 constant history.

All `old_gap_closed_*`, `gap_closed_homogeneous_*`, and `gap_closed_ne_*`
values, the per-cell baselines, and the ordering-check rankings are
unchanged, so no conclusions drawn from this table shift.

## Caveat: the OLD gap_closed column is a frozen historical metric

The `OLD gap_closed` column is read **verbatim** from the per-seed
`summary.json` files, which were written at training time (#360 sweep,
pre-#416) against the historical `MINSPEC_SPECIALIST = −22.07` (denominator
65.65). It is *not* recomputed by the script. Verification: for
`b0.10_k0.10_c0.50`, `trailing5_team_mean_mean = −98.269` gives
`(−98.269 + 87.72) / 65.65 = −0.16069`, matching the stored
`old_gap_closed_mean` exactly; the canonical 59.34 denominator would give
−0.17777.

The `old_baseline_random` / `old_baseline_specialist` fields, by contrast,
are stamped from the *current* `bucket_brigade.baselines` constants at
regeneration time. So in this v1 artifact the OLD column's actual historical
denominator (65.65) differs from the stamped constant (−28.38 → 59.34). This
is deliberate: the OLD column exists as the original #360 point of
comparison, and per the #434/#438 precedent the superseded constant is not
baked back into the tool to force the stamp to match. Treat the OLD column
as historical context only; the load-bearing columns are the per-cell
`gap_closed_homogeneous` / `gap_closed_ne` ones, which never depended on the
MINSPEC constants (`baseline_source = "per_cell"` on every row).

## Status: superseded by the v2 (n=20) sweep for the paper

All current downstream consumers — the workshop paper figure
(`paper/anvil_pub.bb-workshop.*/figures/src/recalibrated_heatmap.py`),
`experiments/nash/phase_diagram/entropy_vs_trainability.py`, and
`experiments/nash/phase_diagram/noise_buydown_precision.py` — read the
**v2** root (`experiments/p3_specialization/phase_diagram_ppo_v2/`, n=20
seeds, #420/#443), whose verdict artifacts reproduce byte-for-byte on their
own. This v1 (n=4) table is retained as the #360/#413 artifact trail.
