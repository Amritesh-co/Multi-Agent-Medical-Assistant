import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/chat': 'http://127.0.0.1:8001',
      '/upload': 'http://127.0.0.1:8001',
      '/generate-speech': 'http://127.0.0.1:8001',
      '/transcribe': 'http://127.0.0.1:8001',
      '/validate': 'http://127.0.0.1:8001',
      '/data': 'http://127.0.0.1:8001',
      '/uploads': 'http://127.0.0.1:8001',
    }
  }
})
