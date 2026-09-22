import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // The API is reached directly rather than proxied, so the CORS configuration in
    // ragkit/api/main.py is exercised in development exactly as it would be in
    // deployment. A proxy here would hide a misconfiguration until production.
  },
});
