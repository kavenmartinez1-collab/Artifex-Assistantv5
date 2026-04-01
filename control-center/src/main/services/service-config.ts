import * as path from 'path';
import * as fs from 'fs';

export interface ServiceDefinition {
  id: string;
  name: string;
  command: string;
  args: string[];
  cwd: string;
  port: number;
  healthCheck?: string;
  gracefulTimeout: number;
  group: string;
}

/**
 * Detect the Artifex project root by walking up from this file's location
 * looking for a directory that contains both webgpu/ and scripts/.
 */
function findProjectRoot(): string {
  // In compiled form: dist/main/services/service-config.js
  // Walk up: services -> main -> dist -> control-center -> project root
  let dir = __dirname;
  for (let i = 0; i < 10; i++) {
    const parent = path.dirname(dir);
    if (parent === dir) break; // filesystem root
    dir = parent;
    if (
      fs.existsSync(path.join(dir, 'webgpu')) &&
      fs.existsSync(path.join(dir, 'scripts'))
    ) {
      return dir;
    }
  }
  // Fallback: assume standard layout
  return path.resolve(__dirname, '..', '..', '..', '..');
}

const PROJECT_ROOT = findProjectRoot();

export const SERVICE_DEFINITIONS: ServiceDefinition[] = [
  {
    id: 'vite',
    name: 'Vite Dev Server',
    command: 'npx',
    args: ['vite', '--host', '127.0.0.1'],
    cwd: path.join(PROJECT_ROOT, 'webgpu'),
    port: 5173,
    healthCheck: 'http://127.0.0.1:5173/',
    gracefulTimeout: 5000,
    group: 'frontend',
  },
  {
    id: 'dev-server',
    name: 'WebGPU Dev Server',
    command: 'npx',
    args: ['tsx', 'server/dev-server.ts'],
    cwd: path.join(PROJECT_ROOT, 'webgpu'),
    port: 3001,
    healthCheck: 'http://127.0.0.1:3001/',
    gracefulTimeout: 5000,
    group: 'frontend',
  },
  {
    id: 'api-server',
    name: 'Python API Server',
    command: path.join(PROJECT_ROOT, 'venv', 'Scripts', 'python.exe'),
    args: ['main_api.py'],
    cwd: PROJECT_ROOT,
    port: 8000,
    healthCheck: 'http://127.0.0.1:8000/health',
    gracefulTimeout: 10000,
    group: 'backend',
  },
];
