# Release workflow

This document covers the operator steps to ship a Bucket Brigade
release: building the pip wheel and mirroring frozen baseline artifacts
to HuggingFace.

It is the companion to issue #373 (Distribution: pip wheel + HF
baselines repo, slice 5/5 of #365). The schema and plumbing live in
`bucket_brigade/baselines/release/`; this doc is the **runbook**.

## TL;DR

```bash
# 1. Build BOTH wheels (pure-Python + per-platform Rust)
uv build --wheel
cd bucket-brigade-core && uv run maturin build --release --features python --out ../dist
cd ..

# 2. Smoke-test in a fresh venv — install both wheels at once
python -m venv /tmp/bb-test
/tmp/bb-test/bin/pip install dist/bucket_brigade_core-*.whl dist/bucket_brigade-*.whl
/tmp/bb-test/bin/python -c "import bucket_brigade, bucket_brigade_core; bucket_brigade.make('minimal_specialization-v1').reset()"

# 3. Cut a release (publishes to PyPI automatically via cibuildwheel)
git tag v0.2.0 && git push origin v0.2.0

# 4. (Optional) Upload frozen baselines to HuggingFace — DRY RUN first
uv run python -m scripts.release.upload_to_hf \
    --source-dir bucket_brigade/baselines/release/local

# 5. (Optional) Real HF upload — requires $HF_TOKEN with write access
uv run python -m scripts.release.upload_to_hf \
    --source-dir bucket_brigade/baselines/release/local \
    --confirm
```

## Pip wheel

### What ships

Two distributable packages are produced for every release (see
`.github/workflows/wheels.yml` and issue #404):

1. **`bucket-brigade`** — built by [hatchling](https://hatch.pypa.io/)
   (see `[build-system]` in `pyproject.toml`). Pure-Python, one wheel
   for all platforms. Contents:

   - The full `bucket_brigade/` Python package — env, agents,
     baselines, registry, evolution, equilibrium.
   - The `bucket_brigade/baselines/release/local/` directory, which is
     the in-wheel home for frozen artifacts populated by slice #371.
   - Minimum metadata: `LICENSE`, `README.md`, declared classifiers,
     PyPI project URLs.

   `bucket-brigade` declares `bucket-brigade-core>=0.1.0` as a hard
   runtime dep, so `pip install bucket-brigade` pulls a matching Rust
   wheel automatically.

2. **`bucket-brigade-core`** — built by
   [maturin](https://www.maturin.rs/) and per-platform-packaged by
   [cibuildwheel](https://cibuildwheel.readthedocs.io/) (issue #404).
   One wheel per (OS, arch, Python ABI) combination. Default build
   matrix:

   | OS                | Arch                  | CPython          |
   | ----------------- | --------------------- | ---------------- |
   | Linux (manylinux) | x86_64                | 3.11, 3.12, 3.13 |
   | Linux (manylinux) | aarch64               | 3.11, 3.12, 3.13 |
   | macOS             | arm64 (Apple Silicon) | 3.11, 3.12, 3.13 |
   | macOS             | x86_64 (Intel)        | 3.11, 3.12, 3.13 |

   An sdist is also published so `pip install` can fall back to a
   source build on platforms with no matching wheel (this requires a
   Rust toolchain on the user's machine).

### Dependency tiers

`pyproject.toml` splits dependencies into a small core plus several
optional extras:

| Extra            | What it adds                                                          | Use case                                                          |
| ---------------- | --------------------------------------------------------------------- | ----------------------------------------------------------------- |
| _(core)_         | `numpy`, `scipy`, `gymnasium`, `typer`, `rich`, `tqdm`, `bucket-brigade-core` | Env + hand-coded baselines + Nash solver + frozen-baseline loader + Rust speedup |
| `[rl]`           | `torch`, `tensorboard`, `optuna`, `plotly`, `psutil`                  | PPO training, hyperparameter sweeps                               |
| `[research]`     | `pandas`, `scikit-learn`, `matplotlib`                                | Experiment analysis, tabular reports                              |
| `[huggingface]`  | `huggingface_hub`                                                     | Mirroring / downloading baseline bundles                          |
| `[dev]`          | `pytest`, `ruff`, `mypy`, `pre-commit`, ...                           | Local development                                                 |
| `[all]`          | `[rl,research,huggingface]`                                           | One-shot research workstation install                             |

Goal: a clean `pip install bucket-brigade` pulls everything needed to
import the package and run `make()` with the Rust speedup. Heavy ML /
experiment stack is opt-in.

### Building and testing locally

From the repo root, to reproduce what cibuildwheel does on CI:

```bash
# 1. Pure-Python wheel
uv build --wheel
ls dist/   # bucket_brigade-<version>-py3-none-any.whl

# 2. Rust wheel for the local platform (maturin under the hood)
cd bucket-brigade-core && uv run maturin build --release --features python --out ../dist
cd ..
ls dist/   # bucket_brigade-<version>-py3-none-any.whl + bucket_brigade_core-<version>-cp3XX-...whl
```

Smoke-test in a clean Python venv (catches accidentally-required deps
that shouldn't be in `core`):

```bash
python -m venv /tmp/bb-smoke
# install both wheels in one command so pip sees the local Rust wheel
# satisfying the bucket-brigade-core dep
/tmp/bb-smoke/bin/pip install dist/bucket_brigade_core-*.whl dist/bucket_brigade-*.whl
/tmp/bb-smoke/bin/python -c "
import bucket_brigade, bucket_brigade_core
env = bucket_brigade.make('minimal_specialization-v1')
obs, info = env.reset(seed=0)
print('OK:', type(env).__name__, obs.shape, 'rust:', bucket_brigade_core.__name__)
"
```

This matches the smoke test enforced by
`tests/test_release_wheel.py`.

### Publishing to PyPI (operator step)

**The cibuildwheel workflow handles publishing automatically on tag
push.** To cut a release:

```bash
# 1. Bump version in pyproject.toml AND bucket-brigade-core/pyproject.toml
#    (they MUST be in sync; bucket-brigade declares
#    bucket-brigade-core>=<version>).
git commit -am "release: 0.2.0"
git tag v0.2.0
git push origin main v0.2.0
```

The push of `v*` to GitHub triggers `.github/workflows/wheels.yml`,
which:

1. Builds all `bucket-brigade-core` wheels in the matrix (cibuildwheel
   + maturin).
2. Builds the `bucket-brigade-core` sdist (maturin).
3. Builds the `bucket-brigade` pure-Python wheel + sdist (hatchling).
4. Publishes the lot to PyPI via OIDC **trusted publishing** — no API
   token stored in the repo.

**One-time PyPI side setup (human task)**:
- Claim/create the `bucket-brigade` and `bucket-brigade-core` projects
  on PyPI.
- Configure a "trusted publisher" on each project pointing at this
  repo + workflow file (`wheels.yml`) + job (`publish`) + environment
  (`pypi`). See
  [docs.pypi.org/trusted-publishers](https://docs.pypi.org/trusted-publishers/)
  for the click-through.

**No CI step pushes to PyPI on plain branch / PR runs** — the publish
job is gated on `startsWith(github.ref, 'refs/tags/v')`.

#### Manual publishing fallback

If for some reason the OIDC publish has to be bypassed (e.g. PyPI
trusted-publisher misconfiguration), the operator can do it by hand:

```bash
uv build                                # bucket-brigade (sdist + wheel)
cd bucket-brigade-core && uv run maturin sdist --out ../dist  # core sdist only
uv run twine check dist/*               # sanity check
uv run twine upload --repository testpypi dist/*    # try on TestPyPI first
uv run twine upload dist/*              # real upload (irreversible!)
```

Note: this manual path only publishes the sdist for the Rust package —
per-platform wheels still require the CI matrix.

## HuggingFace baselines repo

### Manifest contract

Every release bundle is identified by a `manifest.json` at its root.
The schema lives in
[`bucket_brigade/baselines/release/manifest.py`](../bucket_brigade/baselines/release/manifest.py)
and is summarised here:

```json
{
  "schema_version": 1,
  "release_version": "0.1.0",
  "release_date": "2026-06-08",
  "source_commit": "abc123",
  "huggingface_repo": "rjwalters/bucket-brigade-baselines",
  "artifacts": [
    {
      "kind": "archetype",
      "name": "hero",
      "filename": "archetypes/hero.pkl",
      "scenario_id": null,
      "sha256": "<hex>",
      "size_bytes": 12345,
      "notes": "Hand-coded Hero archetype"
    }
  ],
  "extra": {}
}
```

Slice #371 owns the **production** of artifacts conforming to this
schema. Slice #373 (this one) owns the schema, the loader, and the
upload/download CLI.

### Layout

Inside the bundle:

```
manifest.json
archetypes/
    hero.pkl
    firefighter.pkl
    ...
nash/
    minimal_specialization-v1.json
    ...
ppo/
    minimal_specialization-v1.pt
    ...
```

This layout is consumed by `resolve_artifact_path` and mirrored
verbatim to the HuggingFace repo.

### Uploading (operator step)

1. Stage the bundle (typically by running #371's freeze script into a
   scratch directory).
2. Dry-run the upload to validate the manifest:

   ```bash
   uv run python -m scripts.release.upload_to_hf --source-dir <staging>
   ```

3. If happy, push for real with an HF token that has write access to
   the target repo:

   ```bash
   export HF_TOKEN=...
   uv run python -m scripts.release.upload_to_hf \
       --source-dir <staging> \
       --confirm
   ```

The script prints the commit URL on success.

### Downloading

End users do not need to run anything special — the wheel ships a copy
of `local/`. If they want to refresh from HF (e.g., a post-release
patch bundle):

```python
from bucket_brigade.baselines.release import hub
cache_dir = hub.download_release()   # ~/.cache/bucket_brigade/baselines/<ver>/
```

`resolve_artifact_path` then picks up the cached copy automatically via
the lookup order documented in
[`paths.py`](../bucket_brigade/baselines/release/paths.py).

## CI smoke tests

`.github/workflows/ci.yml` runs the pytest suite, which includes:

- `tests/test_release_wheel.py` — builds **both** the pure-Python
  wheel (hatchling) and the Rust wheel (maturin), installs them into a
  fresh venv, and verifies `bucket_brigade.make().reset()` works AND
  `bucket_brigade_core` is importable from the venv (issue #404).
- `tests/test_release_manifest.py` — round-trips the manifest schema.
- `tests/test_release_paths.py` — exercises `resolve_artifact_path`
  against a synthetic manifest.
- `tests/test_release_hub.py` — mocks `huggingface_hub` and verifies
  the dry-run upload path + the optional-import error message.

Additionally, `.github/workflows/wheels.yml` runs cibuildwheel on
every PR (build-only, no publish) to verify the per-platform Rust
build matrix is still green.

No CI step actually downloads from HuggingFace; that requires network
and a real bundle to exist, which is #371's job.

## Open follow-ups

- **#371** populates `bucket_brigade/baselines/release/local/` with
  real artifacts + writes the first `manifest.json`.
- **#371** adds the `bucket_brigade.baselines.load_archetype(...)`
  family of API calls that consume `resolve_artifact_path` output.
- **Post-paper**: claim `bucket-brigade` and `bucket-brigade-core` on
  PyPI, configure trusted publishers (see "Publishing to PyPI"),
  then `git tag v0.1.0` to trigger the first real release. Create
  the HF repo `rjwalters/bucket-brigade-baselines` and run the
  upload script.
