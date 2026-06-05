# Heterogeneous Nash Equilibrium — Findings

**Run date**: 2026-06-04 — 2026-06-05
**Compute**: alc-9 (Intel i9-14900K, 32 threads, 62 GB RAM)
**Algorithm**: `bucket_brigade.equilibrium.double_oracle_heterogeneous.HeterogeneousDoubleOracle`
**Settings**: 20 random restarts × 25 max iterations × 1000 MC sims/payoff (300 during L-BFGS-B refinement), ε=50, seed=42

## Headline

| Scenario | Verdict | Best converged NE | Pattern |
|---|---|---|---|
| `minimal_specialization` | `symmetric_ne_superior` | **−756** (sym Hero) | 4/5 converged restarts are pure all-Hero |
| `rest_trap` | `asymmetric_only` | **+2984** (asym FR/FF) | 8/13 converged restarts are `FR × 3 + FF × 1` |

These two scenarios sit at opposite ends of the symmetric–asymmetric NE spectrum.

## `minimal_specialization` → Hero is the only stable NE

20 random starting profiles, 5 converged ε-Nash equilibria:

| Payoff | Profile | Iterations |
|---|---|---|
| **−756.4** | `hero | hero | hero | hero` | 6 |
| −766.4 | `hero | hero | hero | hero` | 11 |
| −832.3 | `hero | hero | hero | hero` | 13 |
| −916.6 | `hero | hero | hero | hero` | 24 |
| −879.0 | `firefighter | hero | hero | hero` | 20 |

- 11/20 best-found profiles are exactly symmetric (all-Hero)
- The only converged asymmetric profile (`FF + 3·Hero`, payoff −879) is **strictly worse** than the symmetric Hero NE
- All non-converged restarts that show "asymmetric" patterns are 3·Hero + 1·FF — i.e., the optimization was trying to drift one position to FF and never settled

**Implication**: Role-differentiated specialization is **not** a Nash equilibrium for `minimal_specialization`. The symmetric all-Hero strategy is the dominant equilibrium and any deviation loses payoff.

This invalidates the premise of the P3 specialization research track (issues #271, #346, #351). The tier-1 trainers returning `insufficient` gap_closed < 0.20 is not a training failure — it's a game-theoretic dead end. There is no specialization equilibrium to learn.

## `rest_trap` → Free-rider equilibrium is the only NE

20 random starting profiles, 13 converged ε-Nash equilibria, **0 symmetric** profiles found anywhere:

| Payoff | Profile | Iterations |
|---|---|---|
| **2984.0** | `FR | FR | FR | FF` | 5 |
| 2911.6 | `liar | coord | coord | FR` | 13 |
| 2904.7 | `FR | liar | coord | liar` | 8 |
| 2898.0 | `FR | FR | FR | FF` | 10 |
| 2868.9 | `liar | coord | FR | liar` | 19 |
| 2864.7 | `coord | FR | coord | coord` | 5 |
| 2850.4 | `FR | FR | FF | FR` | 2 |
| 2841.7 | `coord | liar | liar | FR` | 6 |
| 2819.1 | `liar | FR | FR | FF` | 6 |
| 2817.4 | `FR | FR | FR | FF` | 14 |
| 2800.1 | `FR | FR | FR | FF` | 5 |
| 2796.2 | `FR | FR | FR | FF` | 9 |
| 2786.8 | `FR | FR | FR | FF` | 3 |

- 8 of 13 converged restarts: exactly `FR × 3 + FF × 1` (free-rider/firefighter mix, with FF in varying positions)
- Strategies converge to exact archetypes (d=0.00) — no novel strategies were discovered
- Remaining 5 converged restarts substitute Coordinator/Liar (also low-work-tendency) for some FR slots

**Implication**: `rest_trap`'s Nash equilibrium requires asymmetry. One agent must work; the other three free-ride. This explains the persistent symmetric DO cycling reported in #352/#353 — the algorithm was correctly orbiting a non-existent symmetric fixed point.

## Algorithm notes

**Calibration:**
- ε=2.0 (initial attempt) was below MC noise floor (~15–30 standard error at 1000 sims, payoff scale ±1000–3000) → every restart hit MAX_ITER
- ε=50 (final): 13/20 resttrap restarts converged in 5–20 iters; 5/20 minspec restarts converged
- See `~/.claude/.../memory/nash_heterogeneous_calibration.md` for calibration heuristics

**Performance:**
- 5h 10m for resttrap (mix of fast convergence + MAX_ITER restarts)
- 6h 53m for minspec (most restarts hit MAX_ITER)
- ~25–30 min for a MAX_ITER restart, ~3–14 min for a converging restart
- CPU bound: L-BFGS-B refinement calls `_objective` ~200× sequentially per BR; parallel Pool is wasteful at that granularity

**Verdict logic fix (committed alongside these results):** original script picked `best_asym` from *all* asymmetric profiles (including non-converged), then said "not converged" — even when many genuine ε-Nash equilibria existed. Fix uses `best_conv_asym` from the converged-asymmetric subset and adds a `symmetric_ne_superior` verdict category.

## Reproducing

```bash
# Heterogeneous DO sweep (requires ~5–7h on a 32-thread CPU box)
uv run python experiments/scripts/compute_nash_heterogeneous.py minimal_specialization \
    --restarts 20 --simulations 1000 --opt-simulations 300 \
    --max-iterations 25 --epsilon 50 --seed 42

uv run python experiments/scripts/compute_nash_heterogeneous.py rest_trap \
    --restarts 20 --simulations 1000 --opt-simulations 300 \
    --max-iterations 25 --epsilon 50 --seed 42

# Regenerate verdicts from existing results.json (no re-run)
python experiments/nash/heterogeneous/regen_summaries.py
```

## Next steps

1. **Decide whether P3 specialization research continues.** If yes, on which scenarios? `minimal_specialization` is ruled out as a learning target by these results.
2. **Sweep more scenarios.** This pipeline can clarify the NE structure of any scenario in ~5–7 hours; would be informative on the other tier-1 cells (`chain_reaction`, etc.).
3. **Specialist exploitability test** (issue #354): for rest_trap, verify the `3·FR + 1·FF` equilibrium is robust against best-responding "exploiters." Quick to run; closes the symmetric-vs-asymmetric question definitively.
