# Release workflow

This document covers the operator steps to ship a Bucket Brigade
release: building the pip wheel and mirroring frozen baseline artifacts
to HuggingFace.

It is the companion to issue #373 (Distribution: pip wheel + HF
baselines repo, slice 5/5 of #365). The schema and plumbing live in
`bucket_brigade/baselines/release/`; this doc is the **runbook**.

## TL;DR

```bash
# 1. Build the wheel
uv build --wheel

# 2. Smoke-test in a fresh venv
python -m venv /tmp/bb-test && /tmp/bb-test/bin/pip install dist/bucket_brigade-*.whl
/tmp/bb-test/bin/python -c "import bucket_brigade; bucket_brigade.make('minimal_specialization-v1').reset()"

# 3. (Optional) Upload frozen baselines to HuggingFace — DRY RUN first
uv run python -m scripts.release.upload_to_hf \
    --source-dir bucket_brigade/baselines/release/local

# 4. (Optional) Real upload — requires $HF_TOKEN with write access
uv run python -m scripts.release.upload_to_hf \
    --source-dir bucket_brigade/baselines/release/local \
    --confirm
```

## Pip wheel

### What ships

The wheel is built by [hatchling](https://hatch.pypa.io/) (see
`[build-system]` in `pyproject.toml`) and includes:

- The full `bucket_brigade/` Python package — env, agents, baselines,
  registry, evolution, equilibrium.
- The `bucket_brigade/baselines/release/local/` directory, which is
  the in-wheel home for frozen artifacts populated by slice #371.
- Minimum metadata: `LICENSE`, `README.md`, declared classifiers,
  PyPI project URLs.

The wheel **does NOT** include the Rust core (`bucket-brigade-core/`).
That ships as a separate package — see "Rust core packaging" below.

### Dependency tiers

`pyproject.toml` splits dependencies into a small core plus several
optional extras:

| Extra | What it adds | Use case |
|-------|-------------|----------|
| _(core)_ | `numpy`, `scipy`, `gymnasium`, `typer`, `rich`, `tqdm` | Env + hand-coded baselines + Nash solver + frozen-baseline loader |
| `[rl]` | `torch`, `tensorboard`, `optuna`, `plotly`, `psutil` | PPO training, hyperparameter sweeps |
| `[research]` | `pandas`, `scikit-learn`, `matplotlib` | Experiment analysis, tabular reports |
| `[huggingface]` | `huggingface_hub` | Mirroring / downloading baseline bundles |
| `[dev]` | `pytest`, `ruff`, `mypy`, `pre-commit`, ... | Local development |
| `[all]` | `[rl,research,huggingface]` | One-shot research workstation install |

Goal: a clean `pip install bucket-brigade` pulls only what's needed to
import the package and run `make()`. Heavy ML / experiment stack is
opt-in.

### Building and testing the wheel

From the repo root:

```bash
uv build --wheel
ls dist/   # bucket_brigade-<version>-py3-none-any.whl
```

Smoke-test in a clean Python venv (catches accidentally-required
deps that shouldn't be in `core`):

```bash
python -m venv /tmp/bb-smoke
/tmp/bb-smoke/bin/pip install dist/bucket_brigade-*.whl
/tmp/bb-smoke/bin/python -c "
import bucket_brigade
env = bucket_brigade.make('minimal_specialization-v1')
obs, info = env.reset(seed=0)
print('OK:', type(env).__name__, obs.shape if hasattr(obs, 'shape') else obs.keys())
"
```

This is the same smoke test enforced by
`tests/test_release_wheel.py`.

### Rust core packaging

The Rust extension (`bucket-brigade-core/`) is its own setuptools
package built via `setuptools-rust`. To ship a usable end-user wheel
with the Rust speedup, the user runs both:

```bash
pip install bucket-brigade        # pure-Python wheel
cd bucket-brigade-core && pip install .   # builds the .so
```

The Python package falls back to a slower pure-Python game loop when
the Rust extension is missing, so the core install is still usable on
day 1 without compiling Rust.

> **Rust core PyPI packaging is out of scope for #373** — see #371
> follow-up if we want to ship pre-built Rust wheels per platform via
> `cibuildwheel`.

### Publishing to PyPI (operator step)

Not done automatically. When ready:

```bash
uv build                                # produces sdist + wheel in dist/
uv run twine check dist/*               # sanity check
uv run twine upload --repository testpypi dist/*    # try on TestPyPI first
uv run twine upload dist/*              # real upload (irreversible!)
```

Requires `~/.pypirc` or `$TWINE_PASSWORD` to be set with a PyPI API
token. **No automation in this repo invokes `twine upload`.**

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

- `tests/test_release_wheel.py` — builds the wheel and verifies it
  installs into a fresh venv.
- `tests/test_release_manifest.py` — round-trips the manifest schema.
- `tests/test_release_paths.py` — exercises `resolve_artifact_path`
  against a synthetic manifest.
- `tests/test_release_hub.py` — mocks `huggingface_hub` and verifies
  the dry-run upload path + the optional-import error message.

No CI step actually downloads from HuggingFace; that requires network
and a real bundle to exist, which is #371's job.

## Open follow-ups

- **#371** populates `bucket_brigade/baselines/release/local/` with
  real artifacts + writes the first `manifest.json`.
- **#371** adds the `bucket_brigade.baselines.load_archetype(...)`
  family of API calls that consume `resolve_artifact_path` output.
- **Post-paper**: publish to real PyPI; create the HF repo
  `rjwalters/bucket-brigade-baselines` and run the upload script.
- **Stretch**: `cibuildwheel`-built per-platform wheels for
  `bucket-brigade-core` so users don't need a Rust toolchain.
