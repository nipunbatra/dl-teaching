import { defineConfig } from 'vite';

// Use the official versioned browser distribution. Keep application code local
// and readable; do not vendor the unrelated model registry into this repository.
const runtime = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1';
export default defineConfig({
  build: { minify: false },
  worker: { format: 'es', rollupOptions: { external: [runtime] } },
});
