import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
export default defineConfig(({ mode }) => {
  const apiTarget =
    loadEnv(mode, ".", "LONN_").LONN_DEMO_API_URL || "http://127.0.0.1:8765";
  return {
    plugins: [react(), tailwindcss()],
    server: {
      proxy: {
        "/api": apiTarget,
        "/generate": apiTarget,
        "/route": apiTarget,
        "/models": apiTarget,
        "/connection": apiTarget,
      },
    },
  };
});
