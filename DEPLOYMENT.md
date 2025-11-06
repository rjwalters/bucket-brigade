# 🚀 Bucket Brigade Deployment Guide

> ⚠️ **IMPORTANT: Static Site Only**
>
> The Bucket Brigade platform currently operates as a **browser-only static site**. All computation happens client-side using WebAssembly. There is **no backend server required**.
>
> Backend API deployment options described later in this document are **future plans** that are not yet implemented. For current architecture, see [docs/SIMPLIFIED_ARCHITECTURE.md](docs/SIMPLIFIED_ARCHITECTURE.md).

This guide covers deploying the Bucket Brigade platform to various environments.

---

## 📋 Prerequisites

- **Node.js 18+** with pnpm package manager
- **Rust toolchain** (for building WASM)
- **Git** for version control

**Optional** (for development only):
- Python 3.9+ with uv (for running evolution experiments locally)
- Docker (for future backend deployment)

---

## 🏗️ Build Process

### 1. Build WASM Module

```bash
# Install wasm-pack if not already installed
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build Rust core to WebAssembly
cd bucket-brigade-core
wasm-pack build --target web
```

This creates the WASM bindings in `bucket-brigade-core/pkg/`.

### 2. Build Web Frontend

```bash
# Install dependencies
cd web
pnpm install

# Build for production
pnpm run build
```

The built files will be in `web/dist/` ready for deployment.

### 3. Preview Locally

```bash
# From web directory
pnpm run preview
```

Visit `http://localhost:4173` to test the production build.

---

## 🌐 Current Deployment: Static Site Hosting

Since Bucket Brigade is a static site, you can deploy it to any static hosting provider. No server configuration needed!

### Option 1: GitHub Pages (Recommended)

**Automatic deployment via GitHub Actions:**

```yaml
# .github/workflows/deploy-web.yml
name: Deploy Web Interface

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: actions-rust-lang/setup-rust-toolchain@v1
        with:
          toolchain: stable
          target: wasm32-unknown-unknown

      - name: Install wasm-pack
        run: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

      - name: Build WASM
        run: |
          cd bucket-brigade-core
          wasm-pack build --target web

      - name: Setup pnpm
        uses: pnpm/action-setup@v2
        with:
          version: 9

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: 18
          cache: 'pnpm'
          cache-dependency-path: web/pnpm-lock.yaml

      - name: Install and Build
        run: |
          cd web
          pnpm install --frozen-lockfile
          pnpm run build

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./web/dist
```

**Manual deployment:**

```bash
# Build everything
./scripts/build-web.sh  # or follow steps above

# Deploy to gh-pages branch
cd web
npx gh-pages -d dist
```

**Configuration:**
1. Go to repository Settings → Pages
2. Source: Deploy from branch `gh-pages`
3. Your site will be at: `https://[username].github.io/[repo-name]/`

### Option 2: Netlify

**Via Netlify CLI:**

```bash
# Install Netlify CLI globally
npm install -g netlify-cli

# Build the site
cd bucket-brigade-core && wasm-pack build --target web && cd ..
cd web && pnpm install && pnpm run build && cd ..

# Deploy
cd web
netlify deploy --prod --dir=dist
```

**Via Netlify UI:**
1. Connect your GitHub repository
2. Configure build settings:
   - **Base directory**: (leave empty)
   - **Build command**: `cd bucket-brigade-core && wasm-pack build --target web && cd ../web && pnpm install && pnpm run build`
   - **Publish directory**: `web/dist`
3. Add build environment:
   - Node version: `18`

**netlify.toml** (optional):

```toml
[build]
  command = "cd bucket-brigade-core && wasm-pack build --target web && cd ../web && pnpm install && pnpm run build"
  publish = "web/dist"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
```

### Option 3: Vercel

**Via Vercel CLI:**

```bash
# Install Vercel CLI
npm install -g vercel

# Build and deploy
vercel --prod
```

**Via Vercel UI:**
1. Import your repository
2. Framework Preset: **Vite**
3. Root Directory: `web`
4. Build Command: `cd ../bucket-brigade-core && wasm-pack build --target web && cd ../web && pnpm install && pnpm run build`
5. Output Directory: `dist`

**vercel.json** (optional):

```json
{
  "buildCommand": "cd bucket-brigade-core && wasm-pack build --target web && cd web && pnpm install && pnpm run build",
  "outputDirectory": "web/dist",
  "rewrites": [
    { "source": "/(.*)", "destination": "/index.html" }
  ]
}
```

### Option 4: Cloudflare Pages

1. Connect repository via Cloudflare dashboard
2. Build settings:
   - **Build command**: `cd bucket-brigade-core && wasm-pack build --target web && cd ../web && pnpm install && pnpm run build`
   - **Build output directory**: `web/dist`
   - **Root directory**: (leave empty)
3. Environment variables:
   - `NODE_VERSION`: `18`

### Option 5: Self-Hosted (nginx)

**Build and copy files:**

```bash
# Build
cd bucket-brigade-core && wasm-pack build --target web && cd ..
cd web && pnpm install && pnpm run build

# Copy to web server
scp -r dist/* user@server:/var/www/bucket-brigade/
```

**nginx configuration:**

```nginx
server {
    listen 80;
    server_name bucket-brigade.example.com;
    root /var/www/bucket-brigade;
    index index.html;

    # SPA routing
    location / {
        try_files $uri $uri/ /index.html;
    }

    # WASM MIME type
    location ~* \.wasm$ {
        types { application/wasm wasm; }
    }

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|wasm)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
}
```

---

## 🔧 Build Optimization

### Production Build Checklist

- [ ] WASM built with `--release` (wasm-pack does this by default)
- [ ] Vite production build (`pnpm run build`)
- [ ] Assets minified and compressed
- [ ] Source maps disabled in production (check `vite.config.ts`)
- [ ] Analytics configured (if desired)

### Performance Tips

1. **Enable Brotli compression** on your hosting provider
2. **Configure CDN caching** for `.wasm` and `.js` files
3. **Use HTTP/2** for better WASM streaming
4. **Set proper MIME types** for `.wasm` files (`application/wasm`)

---

## 🧪 Testing Deployment

Before deploying to production:

1. **Test production build locally:**
   ```bash
   cd web
   pnpm run build
   pnpm run preview
   ```

2. **Verify WASM loads correctly:**
   - Open browser DevTools → Network tab
   - Check that `.wasm` file loads with `200` status
   - Check for any console errors

3. **Test core functionality:**
   - Run a tournament
   - Verify scenario selection works
   - Check that research data loads

---

## 📊 Monitoring & Analytics

### Google Analytics (Optional)

Add to `web/index.html`:

```html
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX');
</script>
```

### Performance Monitoring

Consider adding:
- **Sentry** for error tracking
- **Web Vitals** for performance metrics
- **LogRocket** for session replay (debugging)

---

## 🚨 Troubleshooting

### WASM fails to load

**Symptoms:** Blank page, console error about WASM

**Solutions:**
1. Check MIME type is set to `application/wasm`
2. Verify CORS headers allow WASM loading
3. Ensure `wasm-pack build` completed successfully
4. Check browser compatibility (needs WebAssembly support)

### Build fails

**Common issues:**
1. **Rust not installed**: Install from https://rustup.rs
2. **wasm-pack missing**: `curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh`
3. **Node version too old**: Upgrade to Node 18+
4. **pnpm not installed**: `npm install -g pnpm`

### 404 on page refresh

**Cause:** Server doesn't handle SPA routing

**Solution:** Add rewrite rules (see platform-specific configs above)

---

## 🔮 Future Plans: Backend API Deployment

> ⚠️ **NOT YET IMPLEMENTED**
>
> The sections below describe future backend deployment options that **do not currently exist**. They are included for planning purposes.
>
> Current status: **Browser-only, no backend required**

<details>
<summary>Click to expand future backend plans</summary>

### Planned Backend Architecture

When implemented, the backend API will provide:
- **Agent registry**: Submit and manage AI policies
- **Tournament orchestration**: Run large-scale competitions
- **Result persistence**: Store tournament results
- **Leaderboards**: Rank policies across scenarios

### Proposed Deployment Options

#### Option 1: Railway

```bash
# Future deployment command (not yet implemented)
railway login
railway init
railway up
```

#### Option 2: Render

```yaml
# Future render.yaml (not yet implemented)
services:
  - type: web
    name: bucket-brigade-api
    env: python
    buildCommand: "pip install -e ."
    startCommand: "uvicorn bucket_brigade.api:app"
```

#### Option 3: Docker Compose

```yaml
# Future docker-compose.yml (not yet implemented)
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
  web:
    build: ./web
    ports:
      - "3000:3000"
```

#### Option 4: Cloud Platforms

- **AWS**: Lambda + API Gateway or ECS Fargate
- **GCP**: Cloud Run or App Engine
- **Azure**: Container Apps or App Service

**Status**: All backend deployment options are in planning phase.

</details>

---

## 📝 Deployment Checklist

### Pre-Deployment

- [ ] Update `package.json` version
- [ ] Update `CHANGELOG.md` with changes
- [ ] Run full test suite (`pytest`, `cargo test`)
- [ ] Build and test production bundle locally
- [ ] Review security headers configuration

### During Deployment

- [ ] Deploy to staging/preview environment first
- [ ] Smoke test critical paths
- [ ] Check browser console for errors
- [ ] Verify WASM loads correctly
- [ ] Test on multiple browsers (Chrome, Firefox, Safari)

### Post-Deployment

- [ ] Monitor error logs for 24 hours
- [ ] Check analytics for unusual traffic
- [ ] Verify SEO/social media previews work
- [ ] Update documentation with new URL if changed

---

## 🔗 Related Documentation

- [SIMPLIFIED_ARCHITECTURE.md](docs/SIMPLIFIED_ARCHITECTURE.md) - System architecture
- [API.md](API.md) - Data structures and future API plans
- [web/README.md](web/README.md) - Web interface details
- [bucket-brigade-core/README.md](bucket-brigade-core/README.md) - Rust core documentation

---

*Last Updated: 2025-11-05*
