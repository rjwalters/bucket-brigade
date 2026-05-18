import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { existsSync } from 'fs'
import { resolve } from 'path'

// Check if WASM package exists
const wasmPkgPath = resolve(__dirname, '../bucket-brigade-core/pkg')
const hasWasm = existsSync(wasmPkgPath)

console.log(`🦀 Rust WASM package: ${hasWasm ? '✅ Available' : '⚠️  Not found - using JS fallback'}`)

// Base path resolution:
//   - VITE_BASE_PATH overrides everything (set in CI for CF Pages, GH Pages, etc.)
//   - Production default is /bucket-brigade/ (matches the GH Pages URL path)
//   - Dev uses /
const basePath =
  process.env.VITE_BASE_PATH ??
  (process.env.NODE_ENV === 'production' ? '/bucket-brigade/' : '/')

// https://vitejs.dev/config/
export default defineConfig({
  base: basePath,
  plugins: [react()],
  server: {
    port: 3000,
    host: true
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      external: (id) => {
        // Make WASM module optional - build will succeed even if it doesn't exist
        if (id.includes('bucket-brigade-core/pkg')) {
          return false; // Try to include it, but don't fail if it's missing
        }
        return false;
      },
      onwarn(warning, warn) {
        // Suppress warnings about missing WASM module
        if (warning.code === 'UNRESOLVED_IMPORT' && warning.message?.includes('bucket-brigade-core/pkg')) {
          return;
        }
        warn(warning);
      }
    }
  },
  optimizeDeps: {
    exclude: ['bucket-brigade-core']
  },
  resolve: {
    alias: hasWasm ? {} : {
      // Provide a stub for the WASM module if it doesn't exist
      '../../../bucket-brigade-core/pkg/bucket_brigade_core': '/src/utils/wasmStub.ts'
    }
  }
})
