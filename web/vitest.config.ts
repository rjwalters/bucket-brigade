/// <reference types="vitest" />
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import { existsSync } from 'fs'
import { resolve } from 'path'

// Mirror the WASM-stub alias from vite.config.ts so component imports that
// transitively load the Rust WASM bundle still resolve in jsdom.
const wasmPkgPath = resolve(__dirname, '../bucket-brigade-core/pkg')
const hasWasm = existsSync(wasmPkgPath)

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
    // IMPORTANT: exclude the Playwright e2e dir (web/tests/) — vitest must not
    // try to execute Playwright specs.
    exclude: ['node_modules', 'dist', 'tests/**', 'playwright-report/**'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html'],
      exclude: ['**/*.test.{ts,tsx}', 'src/test/**'],
    },
  },
  resolve: {
    alias: hasWasm
      ? {}
      : {
          '../../../bucket-brigade-core/pkg/bucket_brigade_core':
            '/src/utils/wasmStub.ts',
        },
  },
})
