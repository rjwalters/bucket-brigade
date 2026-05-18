#!/usr/bin/env bash
# Cloudflare Pages build script.
#
# Runs the full web build pipeline inside CF Pages' Ubuntu build container:
# install Rust + wasm-pack, build the WASM crate, install uv, regenerate
# research content + TypeScript types, then run the Vite build.
#
# CF Pages dashboard config:
#   Build command:        bash scripts/cf-pages-build.sh
#   Build output dir:     web/dist
#   Root directory:       (repo root)
#   Env vars (optional):
#     VITE_BASE_PATH      "/" for subdomain, "/bucket-brigade/" for path-based
#     RUST_VERSION        stable (default below if unset)
#     PYTHON_VERSION      3.12
#     NODE_VERSION        20

set -euo pipefail

echo "::group::env"
node --version || true
pnpm --version || true
python3 --version || true
echo "::endgroup::"

# Install Rust toolchain if not present (CF Pages images vary)
if ! command -v cargo >/dev/null 2>&1; then
  echo "::group::install rust"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain "${RUST_VERSION:-stable}"
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
  rustup target add wasm32-unknown-unknown
  echo "::endgroup::"
fi
export PATH="$HOME/.cargo/bin:$PATH"

# Install wasm-pack
if ! command -v wasm-pack >/dev/null 2>&1; then
  echo "::group::install wasm-pack"
  curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
  echo "::endgroup::"
fi

# Install uv (CF Pages has Python preinstalled; uv gives us a hermetic venv)
if ! command -v uv >/dev/null 2>&1; then
  echo "::group::install uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  echo "::endgroup::"
fi

echo "::group::build wasm"
# `--features wasm` is required: the `wasm` module in
# `bucket-brigade-core/src/lib.rs` is gated behind `#[cfg(feature = "wasm")]`.
# Without it, wasm-pack still emits a `pkg/` directory, but the WASM binary
# contains no `#[wasm_bindgen]`-exported symbols and the JS/TS shim is a thin
# loader with no class bindings (see issue #338, canonical form documented in
# `web/WASM.md`).
(cd bucket-brigade-core && wasm-pack build --target web --features wasm)
echo "::endgroup::"

echo "::group::research content"
mkdir -p web/public/research
cp -r experiments/scenarios web/public/research/
uv run python scripts/build_research_content.py
echo "::endgroup::"

echo "::group::pnpm install"
corepack enable || true
corepack prepare pnpm@9.0.0 --activate || npm install -g pnpm@9.0.0
(cd web && pnpm install --frozen-lockfile)
echo "::endgroup::"

echo "::group::vite build"
(cd web && NODE_ENV=production pnpm run build)
echo "::endgroup::"

echo "Build complete. Output in web/dist (base path: ${VITE_BASE_PATH:-/bucket-brigade/})"
