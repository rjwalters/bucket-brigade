# Cloudflare Pages Deployment

This site is built for Cloudflare Pages. GH Pages remains the fallback / canonical mirror at `https://rjwalters.github.io/bucket-brigade/`.

## Target topology

- **Canonical URL**: `https://bucket-brigade.rjwalters.info/` (CF Pages, custom subdomain)
- **Legacy URL**: `https://rjwalters.info/bucket-brigade/` (forwards to the canonical URL via `_redirects` in the `rjwalters.info` project)

## One-time CF dashboard setup

1. **Create the Pages project**
   - CF dashboard → Pages → Create a project → Connect to Git
   - Pick the `rjwalters/bucket-brigade` repo, production branch `main`

2. **Build configuration**
   - **Build command**: `bash scripts/cf-pages-build.sh`
   - **Build output directory**: `web/dist`
   - **Root directory**: *(repo root, leave blank)*
   - **Environment variables** (Production, Preview):
     - `VITE_BASE_PATH` = `/`
     - `NODE_VERSION` = `20`
     - `PYTHON_VERSION` = `3.12`
     - `RUST_VERSION` = `stable` *(optional)*

3. **Custom domain**
   - Pages project → Custom domains → Set up custom domain → `bucket-brigade.rjwalters.info`
   - CF auto-creates the CNAME record on the `rjwalters.info` zone

4. **Legacy path redirect**
   - In the `rjwalters.info` Pages project, add to its `_redirects` file:
     ```
     /bucket-brigade/*  https://bucket-brigade.rjwalters.info/:splat  301
     ```
   - Commit and redeploy that project.

## Build environment notes

`scripts/cf-pages-build.sh` installs Rust, `wasm-pack`, and `uv` on demand, then runs the same pipeline as the GH Actions workflow at `.github/workflows/deploy.yml`:

1. `wasm-pack build --target web` inside `bucket-brigade-core/`
2. `uv run python scripts/build_research_content.py`
3. `pnpm install --frozen-lockfile` in `web/`
4. `pnpm run build` in `web/` (which itself runs `generate:types` then `vite build`)

CF Pages build images include Node and Python but not Rust — the script installs Rust lazily so warm builds reuse the cache.

## Vite base path

`web/vite.config.ts` reads `VITE_BASE_PATH` (defaults to `/bucket-brigade/` in production so the GH Pages build still works). Setting `VITE_BASE_PATH=/` in CF Pages flips it for the subdomain deploy.

## Keeping GH Pages alive

Until the CF site is verified, the GH Actions workflow (`.github/workflows/deploy.yml`) continues to publish to `rjwalters.github.io/bucket-brigade/`. Once CF is confirmed working, you can disable the GH Pages workflow:

```bash
gh workflow disable deploy.yml
```

…or remove the file.
