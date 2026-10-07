import { defineConfig } from 'vite';

// Keep generated runtime code readable for this teaching lab. Extremely long
// minified lines also confuse key scanners around upstream model class names.
export default defineConfig({ build: { minify: false } });
