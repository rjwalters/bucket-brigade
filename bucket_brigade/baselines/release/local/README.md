# Frozen baseline release bundle (in-wheel copy)

This directory is the **in-wheel** copy of the frozen baselines bundle.
It ships inside `pip install bucket-brigade` and is the first location
checked by `bucket_brigade.baselines.release.resolve_artifact_path`.

**Status (post-#373, pre-#371)**: empty. The plumbing (manifest schema,
loader API, HuggingFace integration) landed in #373; the actual
artifacts will be deposited here by slice #371 (Frozen baseline
release).

## Intended contents (once #371 lands)

```
local/
    manifest.json           # see bucket_brigade/baselines/release/manifest.py
    archetypes/
        hero.pkl
        firefighter.pkl
        free_rider.pkl
        coordinator.pkl
        liar.pkl
    nash/
        minimal_specialization-v1.json
        rest_trap-v1.json
        ...
    ppo/
        minimal_specialization-v1.pt
        ...
```

The `manifest.json` is the single source of truth: every file in this
directory must be referenced by exactly one `ArtifactEntry`.

## How operators populate this

When #371 produces a release-ready bundle:

1. Write artifacts into the above layout under a staging directory
   (NOT into `site-packages`).
2. Run `python -m scripts.release.upload_to_hf --source-dir <staging>
   --dry-run` to validate the manifest.
3. Copy the staging directory into this folder and commit. The next
   wheel build will ship them.
4. (Optional) Mirror to HuggingFace with `--confirm` (operator step).
