import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies API + WebSocket to the single-process backend, so the
// frontend talks to /api and /ws with no CORS or absolute URLs. In production
// the built dist/ is served by its own nginx image (see deploy/).
// This file runs in Node at config time, but the app's tsconfig carries no node types
// (nothing in src/ may touch process) — so declare just the one thing used here.
declare const process: { env: Record<string, string | undefined> };

// Where `npm run dev` proxies /api and /ws. Overridable because the backend is not always
// on its own port: `deploy/asamana.sh` puts nginx in front of it on 8080, and pointing the
// dev server at a running stack (ASAMANA_BACKEND=http://127.0.0.1:8080) beats editing this
// file every time.
const BACKEND = process.env.ASAMANA_BACKEND ?? "http://127.0.0.1:7860";

export default defineConfig({
  plugins: [react()],
  // Frontend is deployed standalone at its own root (nginx / static host).
  base: "/",
  server: {
    proxy: {
      "/api": { target: BACKEND, changeOrigin: true },
      "/ws": { target: BACKEND, changeOrigin: true, ws: true },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
