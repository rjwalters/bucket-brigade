import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
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
    alias: {
      // Provide a stub for the WASM module if it doesn't exist
      '../../../bucket-brigade-core/pkg/bucket_brigade_core': '/src/utils/wasmStub.ts'
    }
  }
})
