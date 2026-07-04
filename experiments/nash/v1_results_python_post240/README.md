# Nash V1 Results — Post-#240 Re-Derivation

This directory holds the canonical Nash equilibrium results for the 12
V1 scenarios computed **after** the `honesty_bias` signal-channel bug fix
landed in PR #245 (issue #240).

## Status — 2026-07-04

**12 of 12 entries present: 11 converged symmetric equilibria + 1
explicitly non-converged, exploitability-annotated entry (`rest_trap`).**

`rest_trap` is special (issue #445, superseding #349/#352): the symmetric
Double Oracle **cycles instead of converging** on this scenario — twice,
independently (#352 unseeded; #445 seeded with the PR #440
scripted-battery winners, 50/50 iterations, min improvement
11.79/episode vs ε = 0.01). Its `equilibrium.json` carries a
`characterization` block recording that the committed final solve is NOT
a Nash equilibrium (best-response deviations gain ≥ 407.50/episode
against it) and that the canonical characterization is the **asymmetric**
NE (free_rider ×3 + firefighter, 2984.04/episode, ε = 50, #355 —
verdict `asymmetric_only`: no symmetric NE exists). Full analysis:
[`experiments/nash/rest_trap_seeded_do/RESULTS.md`](../rest_trap_seeded_do/RESULTS.md).

History: the original `rest_trap` cell failed at the `equilibrium.json`
write step on 2026-05-16 (ENOSPC incident #269, fixed by the
`df`-precheck in PR #315); the plain re-run (#349) was superseded by the
non-convergence finding (#352) and resolved by #445.

Sweep was launched in tmux session `nash256` on `COMPUTE_HOST_PRIMARY`
(Mac Studio, M-series; ~16 performance cores) as part of issue #256.

Wall-clock estimate: ~5 hours with `xargs -P 2` (2 scenarios in parallel,
each scenario internally parallelizes its `differential_evolution` worker
pool to ~10 processes; total worker count stays ≤ ~25 to leave headroom).

## Expected contents (when complete)

One subdirectory per scenario, each containing `equilibrium.json` with the
stable schema:

```
v1_results_python_post240/
├── chain_reaction/equilibrium.json
├── deceptive_calm/equilibrium.json
├── default/equilibrium.json
├── early_containment/equilibrium.json
├── easy/equilibrium.json
├── greedy_neighbor/equilibrium.json
├── hard/equilibrium.json
├── mixed_motivation/equilibrium.json
├── overcrowding/equilibrium.json
├── rest_trap/equilibrium.json
├── sparse_heroics/equilibrium.json
└── trivial_cooperation/equilibrium.json
```

## Provenance

- **Source code**: PR #245 (signal-channel fix), main at `ad7947b4` when
  the sweep started
- **Algorithm**: Double Oracle with Rust-accelerated payoff evaluator
- **Compute settings**: `--simulations 200 --max-iterations 50 --epsilon 0.01 --seed 42`
- **Host**: `robbs-mac-studio` (Mac Studio, M-series)
- **Sweep launcher**: `/tmp/nash256_wait_and_launch.sh` (tmux session
  `nash256`)

## Follow-up

Once the sweep completes:

1. Rsync results into this directory (run from local machine):
   ```bash
   source .env && HOST="$COMPUTE_HOST_PRIMARY"
   rsync -avz "$HOST:~/GitHub/bb_issue256/experiments/nash/v1_results_python_post240/" \
     experiments/nash/v1_results_python_post240/
   ```
2. Run the diff script to populate `docs/NASH_BENCHMARKS.md`:
   ```bash
   uv run python experiments/nash/scripts/diff_post240.py
   ```
3. Commit results + updated table; reference issue #256 and PR #245.
