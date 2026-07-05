# rest_trap seeded Double Oracle retry — results (issue #445)

**Verdict: the seeded symmetric Double Oracle did NOT converge (cycling, same
as #352), and the final mixed profile is exploitable by at least
407.50/episode. No symmetric (ε ≤ 50)-Nash equilibrium was found. rest_trap's
equilibrium characterization remains the asymmetric frozen NE
(free_rider ×3 + firefighter, 2984.04/episode, converged under the
heterogeneous Double Oracle at ε = 50 in #355) — the frozen anchors are
unchanged.**

## Run configuration

- Command: `compute_nash.py rest_trap --simulations 200 --max-iterations 50
  --epsilon 0.01 --seed 42 --seed-profiles
  experiments/nash/seeds/rest_trap_battery_seeds.json` (seeding hook: PR #452)
- Host: `alc-9` (i9-14900K, 32 threads), branch `feature/issue-445` @
  `bf825407`, tmux `nash-resttrap-seeded`, wall-clock cap 86 000 s
- Outcome: **all 50 iterations completed in 19 857 s (~5.5 h), exit 0** —
  the cap was never hit (vs #352's ~3.1 h/iteration on a Mac Studio; alc-9
  ran ~8× faster per iteration)
- Initial pool (9): 4 default archetypes + 5 battery-winner seeds
  (specialist_owned_house_firefighter, firefighter_any_full_work,
  conditional_firefighter_fire_responsive, always_rest, uniform_random)
- Scenario: β = 0.05, κ = 0.95, c = 0.20, 4 agents, min_nights = 12

Artifacts in this directory:

| File | Content |
|---|---|
| `equilibrium.json` | Final (non-converged) solve: mixed, support 7, payoff 2009.22/episode |
| `nash-resttrap-seeded.log` | Full run log (committed as evidence) |
| `do_trace.csv` | Per-iteration improvement trace (extracted by `experiments/nash/scripts/extract_do_trace.py`) |
| `mixture_exploitability.json` / `.log` | Best-response exploitability of the final mixed profile (`experiments/scripts/test_mixture_exploitability.py`, 1000-sim verification) |
| `specialist_team_profile.json` | Owned-house-firefighter ×4 genome profile (input) |
| `specialist_team_exploitability.json` / `.log` | Per-position exploitability of the genome-mapped specialist team (`experiments/scripts/test_specialist_exploitability.py --profile-file`) |

## 1. Convergence outcome: cycling, not converged

The improvement metric (best-response payoff − restricted-game equilibrium
payoff, per episode) **never approached the ε = 0.01 threshold**:

- min improvement **11.79** (iteration 44), max 316.43, median ~150,
  final (iteration 50) 113.46
- equilibrium payoff oscillated in **[1611.56, 2316.61]**/episode with no
  trend (#352's unseeded run: [1590.60, 2400.88] — the *same basin*)
- support reshuffled continuously: 39 of the 48 best-response strategies
  discovered during the run entered some iteration's support

Full trace: [`do_trace.csv`](./do_trace.csv). Reproduce the table with
`uv run python experiments/nash/scripts/extract_do_trace.py
experiments/nash/rest_trap_seeded_do/nash-resttrap-seeded.log --markdown`.

**Did seeding help? No.** The seeded-basin hypothesis from the issue —
that the oracle was cycling only because it never *proposed* the
high-payoff cooperative region — is **disconfirmed**. The specialist and
battery seeds were in the restricted game from iteration 1, so the
cooperative region was available to every restricted-game solve. The
solver moved *away* from it:

- iteration 1 support: coordinator 0.290 + firefighter_any 0.582 +
  conditional_ff 0.128 at 1754.88/episode — two seeds carried mass
  immediately, yet payoff stayed in the low basin
- the specialist seed (index 4) appeared in only 4 of 50 supports, with
  small weights (0.201 → 0.091 → 0.023 → 0.137, last at iteration 33),
  and is absent from the final solve
- across 50 iterations the initial-pool member with the most support mass
  is the plain firefighter archetype (30 appearances), not any seed

The cooperative profile is not hard to *find* — it is not *stable*
(see §2b).

**Why it cycles (evidence for the #352 noise hypothesis):** the restricted
game is solved from a payoff matrix estimated at 200 simulations/cell. At
the 1000-simulation verification budget, the final solve's "indifferent"
support strategies actually score from −258 to +237/episode against the
mixture (see `mixture_exploitability.json`) — the indifference condition
is an artifact of solver noise, so each new best response reshuffles the
support and the oracle chases noise. #352's Option A (variance reduction)
remains the relevant lever for anyone re-attempting a symmetric solve.

## 2. Exploitability bounds

Both measurements are in the Double Oracle's own game model (heuristic
10-dim genome space; per-episode focal payoff; opponents sample one
strategy per episode). Both scripts exit non-zero on EXPLOITABLE by
design — wrappers must tolerate that.

### 2a. Final mixed profile: exploitable by ≥ 407.50/episode

`test_mixture_exploitability.py` on `equilibrium.json`
(1000-sim verification, local + global BR search at 200 sims, candidate
scan over archetypes/support/battery seeds):

- mixture self-play payoff: **2048.66**/episode (final solve's own
  200-sim estimate was 2009.22 — consistent)
- best deviation found: **`always_rest`, +407.50/episode** (payoff
  2456.16); i.e. the profile is **not an ε-NE for any ε < 407**
- deviation ordering is the rest trap in miniature: rest-leaning
  strategies gain (always_rest +407.50, free_rider archetype +304.04,
  specialist +264.25), work-leaning strategies lose (hero −228.34,
  firefighter_any −274.10)
- optimized best responses: global DE +295.02, local +97.26 (verified) —
  both below the plain always_rest candidate, another sign BR search at
  200 sims is noise-limited

**The log's "Cooperative behavior: 100.0%" claim must not be quoted as a
finding.** It is a *parameter classification* of the final support (all 7
support strategies have work_tendency > 0.5), computed on a
non-converged restricted-game solve. The profile it describes is (a) not
an equilibrium (exploitable by ≥ 407.50, precisely by *refusing to
cooperate*), and (b) pays 2048.66/episode ≈ ≤ 170.7/step — **below the
302.87/step uniform-random baseline**. "100% cooperative" here describes
the support's genomes, not equilibrium behavior or efficiency.

### 2b. Specialist team (owned-house firefighter ×4): exploitable at every position

`test_specialist_exploitability.py --profile-file
specialist_team_profile.json` (1000-sim evaluation, 300-sim BR search):

| Position | Eq payoff | BR payoff | Gap | BR archetype |
|---:|---:|---:|---:|---|
| 0 | 1906.47 | 2511.38 | **+604.91** | hero |
| 1 | 1913.93 | 2765.22 | **+851.29** | firefighter |
| 2 | 1961.13 | 2830.69 | **+869.56** | firefighter |
| 3 | 1965.32 | 2801.77 | **+836.45** | firefighter |

Team payoff 1936.71/episode; every position gains 600–870/episode by
deviating (toward *more* work — against three own-house-only defenders,
unattended fires spread and a working deviator captures large team-reward
gains). The genome-mapped specialist team is **not an equilibrium** and
is nowhere near one.

**Mapping-fidelity caveat (important):** the scripted battery winner
scores 386.60/step ≈ 4639/episode (PR #440), but its genome mapping
(`rest_trap_battery_seeds.json`, closest archetype: free_rider at
d = 0.96) scores only 1936.71/episode ≈ ≤ 161.4/step — the 10-dim
heuristic space does not faithfully express the scripted
"rest-unless-own-house-burns" policy. The bound above therefore applies
to the *genome image* of the specialist team inside the heuristic
strategy space, which is what the Double Oracle actually searched. The
scripted team itself lives outside that space; no equilibrium claim was
ever made for it (it is the `scripted_best` cooperative-payoff anchor,
not an equilibrium anchor), and none is made here.

## 3. Committed characterization (post-#240 format)

`experiments/nash/v1_results_python_post240/rest_trap/equilibrium.json`
now exists, completing #347's table at 12/12 entries: 11 converged
symmetric equilibria + rest_trap as an explicitly *non-converged,
exploitability-annotated* entry. Its `characterization` block records:

- **no symmetric NE found** — two independent symmetric DO runs cycle
  (#352 unseeded from archetypes; this run seeded with battery winners)
- the final mixed profile's measured exploitability lower bound
  (≥ 407.50/episode, deviation `always_rest`)
- the canonical characterization: **asymmetric NE, free_rider ×3 +
  firefighter, team payoff 2984.04/episode, converged at ε = 50 under
  `HeterogeneousDoubleOracle` with 13/20 restarts** (#355;
  `experiments/nash/heterogeneous/rest_trap/`, frozen as
  `bucket_brigade/baselines/release/local/nash/rest_trap-v1.json`)

This matches the #355 heterogeneous verdict `asymmetric_only`: rest_trap
admits **no symmetric equilibrium** — role differentiation is required —
and the symmetric DO's cycling is the symmetric-solver view of that fact.

## 4. Anchor reconciliation: frozen NE unchanged

Nothing found in this run beats or destabilizes the frozen anchors:

| Quantity | Value | Status |
|---|---|---|
| Frozen NE (`rest_trap-v1.json`, FR×3+FF) | 2984.04/episode (≤ 248.67/step) | **unchanged** — every symmetric profile found pays ≤ 2317/episode |
| `SCENARIO_GAP_REFERENCES["rest_trap"].ne_per_step_bound` (`bucket_brigade/baselines/__init__.py`) | 2984.043694076538 / 12 | **no update needed** |
| `scripted_best` trap anchor | 386.60/step | **unchanged** (payoff anchor, not an equilibrium; its genome image is exploitable, consistent with the trap verdict) |
| Trap verdict (NE below random baseline) | 248.67 < 302.87/step | **reinforced** — the best symmetric mixture also sits below random (≤ 170.7/step) |

No coordinated version-bumped anchor update is required.

**Still open (flagged, out of scope here):** #442's
`beta_residuals.md` deferred to this issue the question of whether the
*phase-diagram* asymmetric profiles (e.g. FF|hero|hero|FF at
κ = 0.90, c = 0.50) are themselves ε-NE — that is a different cell than
the rest_trap scenario (β = 0.05, κ = 0.95, c = 0.20) and is **not
resolved by this run**. The `test_mixture_exploitability.py` /
`test_specialist_exploitability.py --profile-file` harnesses shipped here
are the right tools for it; needs a follow-up issue with its own compute
budget.

## Cross-references

- #352 (unseeded cycling incident — this run closes its loop), #347
  (11/12 recovery), #355 (asymmetric NE + `asymmetric_only` verdict),
  #445 (this work), PR #452 (seeding hook), PR #440 (`scripted_best`
  anchor), #442 / `experiments/nash/phase_diagram/beta_residuals.md`
  (deferred phase-diagram ε-NE question)
