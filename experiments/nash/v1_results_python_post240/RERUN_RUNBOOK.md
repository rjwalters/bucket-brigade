# `rest_trap` Nash re-run — operator runbook

This runbook is the operator companion to
[`experiments/scripts/launch_rest_trap_rerun.sh`](../../scripts/launch_rest_trap_rerun.sh).
It exists to fill the **single missing cell** of the V1 post-#240
re-derivation sweep — the `rest_trap` scenario, which is the 12th of 12
and the only one whose `equilibrium.json` is not yet committed to this
directory.

Tracked by [issue #349][issue-349]. The original sweep (#256, tmux
session `nash256` on `COMPUTE_HOST_PRIMARY`, 2026-05-16) completed 11 of
12 scenarios and then ran `rest_trap` into an `ENOSPC` failure at the
final `equilibrium.json` write step. PR #347 recovered the 11 successful
artifacts; the `rest_trap` cell has to be re-run from scratch.

The disk-full failure mode is now permanently prevented by the
`df`-precheck added in PR #315 (closes #269, verified wired at
`compute_nash.py:329`). It's safe to re-run `rest_trap` alone without
re-burning the ~5 hour sweep on the other 11 scenarios — the launcher
defaults match the canonical sweep so the resulting artifact is
schema-comparable.

[issue-349]: https://github.com/rjwalters/bucket-brigade/issues/349

The launch script does **not** run any compute locally — it shells into
a remote host listed in `.env`, pulls latest `main`, builds the Rust
extension, and starts a detached `tmux` session running the driver
([`compute_nash.py`](../../scripts/compute_nash.py)) with the canonical
single-scenario invocation. The actual `rest_trap` cell completes ~25–35
minutes later inside the tmux session (Mac Studio, M-series ~16
performance cores); the operator's job is to launch, wait, then rsync
and verify.

## Prerequisites

- The `.env` file at the repo root must define at least one
  `COMPUTE_HOST_*` alias resolvable via the local `~/.ssh/config`
  (see `.env.example`).
- `COMPUTE_HOST_PRIMARY` is the canonical target (the Mac Studio that
  ran the original `nash256` sweep) but any healthy CPU box works — the
  scenario is CPU-bound, GPU offers no speed-up.
- No need for the #391/#392 phase-diagram performance fixes here.
  `compute_nash.py` is the V1 sweep driver, not the heterogeneous
  phase-diagram driver — its compute graph is small enough that the
  default settings already finish in well under an hour.

## Canonical sweep parameters

These are the **exact** values used by the other 11 sibling scenarios.
Acceptance criterion (from the issue body) requires the `rest_trap`
equilibrium to match this schema, so the launcher defaults are
locked-in:

| Parameter         | Value | Source                                  |
|-------------------|-------|------------------------------------------|
| `simulations`     | 200   | `nash256` sweep (issue #256, PR #347)    |
| `max-iterations`  | 50    | `nash256` sweep                          |
| `epsilon`         | 0.01  | `nash256` sweep                          |
| `seed`            | 42    | `nash256` sweep                          |

The launcher exposes overrides (`--simulations`, `--seed`, etc.) for
experimentation, but **do not pass them for the issue #349 re-run** —
the schema-match check downstream of the diff script will reject any
drift from these values.

## Launch command (copy-paste)

The single-line, default-everything invocation:

```bash
./experiments/scripts/launch_rest_trap_rerun.sh
```

That auto-resolves the host from `.env` (priority:
`PRIMARY → CLUSTER → LAMBDA → GCP`), verifies SSH reachability, bootstraps
the remote, and starts a tmux session named `nash-rest-trap`.

If you want to pick the host explicitly (e.g. `PRIMARY` is busy with
something else):

```bash
./experiments/scripts/launch_rest_trap_rerun.sh --host alc-9
```

To preview without launching (no SSH, no remote side effects):

```bash
./experiments/scripts/launch_rest_trap_rerun.sh --dry-run
```

## Monitoring

The launcher prints the tmux session name (`nash-rest-trap`) and log
path on success. From local:

```bash
# Live attach
ssh <host> -t 'tmux attach -t nash-rest-trap'

# Tail the log
ssh <host> 'tail -f bucket-brigade/experiments/nash/v1_results_python_post240/rest_trap/nash-rest-trap.log'
```

Typical progress lines look like the standard `compute_nash.py` output —
periodic Double Oracle iteration summaries, then a final `✅ Results
saved to: …/equilibrium.json` line.

## After the cell finishes — rsync, verify, commit

Once the tmux session has printed `✅ Results saved to:` and exited, do
the merge locally:

```bash
# 1. Rsync the result back into this checkout.
HOST=$(./experiments/scripts/launch_rest_trap_rerun.sh --dry-run | grep '^Host:' | awk '{print $2}')
rsync -avz \
    "$HOST:bucket-brigade/experiments/nash/v1_results_python_post240/rest_trap/" \
    experiments/nash/v1_results_python_post240/rest_trap/

# 2. Sanity-check the artifact is well-formed.
uv run python -c "
import json
d = json.load(open('experiments/nash/v1_results_python_post240/rest_trap/equilibrium.json'))
assert d['algorithm']['seed'] == 42, d['algorithm']
assert d['algorithm']['num_simulations'] == 200, d['algorithm']
assert d['algorithm']['max_iterations'] == 50, d['algorithm']
assert d['algorithm']['epsilon'] == 0.01, d['algorithm']
print('schema OK:', d['algorithm'])
"

# 3. Re-run the post-240 diff to populate docs/NASH_BENCHMARKS.md with
#    the rest_trap row that has been blank since 2026-05-16.
uv run python experiments/nash/scripts/diff_post240.py

# 4. Update this directory's README.md to flip "11 of 12 complete" to
#    "12 of 12 complete" and drop the rest_trap-as-follow-up paragraph.

# 5. Commit + open a PR referencing #349.
git add experiments/nash/v1_results_python_post240/rest_trap/equilibrium.json
git add experiments/nash/v1_results_python_post240/README.md
git add docs/NASH_BENCHMARKS.md  # if the diff script updated it
git commit -m "exp(nash): re-run rest_trap to complete V1 post-#240 sweep (closes #349)"
```

## Troubleshooting

- **Host unreachable**: the script aborts with exit 4 before consuming
  any compute. Check `ssh -v <host>` and the host's reachability/power.
- **df-precheck triggers**: the script will abort with a clear
  `ERROR: only X.X MiB free on '...'; need at least 100 MiB. Aborting
  before compute.` message — this is the PR #315 guard doing its job.
  Free space on the target host's filesystem and re-launch.
- **Build failure on remote**: if `bucket-brigade-core/build.sh`
  complains about a stale `.so`, the remote venv is missing pip. The
  script seeds pip automatically; if that fails, ssh in and run
  `uv pip install pip && bash bucket-brigade-core/build.sh` once by
  hand.
- **Equilibrium schema mismatch**: if the post-rsync sanity check
  asserts on `algorithm.seed != 42` etc., you (or someone) overrode the
  launcher defaults. Re-launch without the overrides; the canonical
  values are not negotiable for the schema-match AC.
- **Diff script blank `rest_trap` row after merge**: the diff script
  (`experiments/nash/scripts/diff_post240.py`) reads
  `equilibrium.json` from each sibling dir. If the row is blank,
  double-check that the rsync targeted
  `experiments/nash/v1_results_python_post240/rest_trap/` and not a
  sibling tree (e.g. `experiments/scenarios/rest_trap/nash/` — the
  driver's own default output path when `--output-dir` is omitted).
