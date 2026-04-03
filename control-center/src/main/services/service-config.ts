import * as path from 'path';
import * as fs from 'fs';

/** Describes a configurable option shown in the UI before starting a service */
export interface ServiceOption {
  key: string;        // CLI flag name (e.g. '--backend')
  label: string;      // UI label
  type: 'select' | 'text' | 'number';
  default: string;
  choices?: string[]; // for 'select' type
  envVar?: string;    // if set, also passed as environment variable
}

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
  /** Launch in a visible terminal window (for interactive CLIs) */
  openTerminal?: boolean;
  /** Configurable options shown in the UI */
  options?: ServiceOption[];
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

// On Windows, npx/npm are .cmd batch files — shell:false won't find them.
// Resolve to the actual .cmd path so spawn works without shell:true.
const NPX = process.platform === 'win32' ? 'npx.cmd' : 'npx';

export const SERVICE_DEFINITIONS: ServiceDefinition[] = [
  // ── Frontend ──
  {
    id: 'vite',
    name: 'Vite Dev Server',
    command: NPX,
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
    command: NPX,
    args: ['tsx', 'server/dev-server.ts'],
    cwd: path.join(PROJECT_ROOT, 'webgpu'),
    port: 3001,
    healthCheck: 'http://127.0.0.1:3001/',
    gracefulTimeout: 5000,
    group: 'frontend',
  },
  // ── Backend ──
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
    options: [
      { key: '--port', label: 'Port', type: 'number', default: '8000' },
      { key: '--backend', label: 'Backend', type: 'select', default: 'transformers',
        choices: ['transformers', 'ollama'], envVar: 'ARTIFEX_BACKEND' },
      { key: '--model', label: 'Model', type: 'text', default: '',
        envVar: 'ARTIFEX_MODEL' },
      { key: '--gateway', label: 'Gateway URL', type: 'text', default: 'http://localhost:8080' },
    ],
  },
  {
    id: 'web-gateway',
    name: 'Web Gateway (local)',
    command: path.join(PROJECT_ROOT, 'venv', 'Scripts', 'python.exe'),
    args: [path.join('web-gateway', 'main.py')],
    cwd: PROJECT_ROOT,
    port: 8080,
    healthCheck: 'http://127.0.0.1:8080/health',
    gracefulTimeout: 5000,
    group: 'backend',
    options: [
      { key: '--port', label: 'Port', type: 'number', default: '8080' },
    ],
    // NOTE: Do NOT start this if Docker is running the web-gateway container.
    // Docker's web-gateway has access to SearXNG on the internal network.
    // The local gateway cannot reach Docker's SearXNG (http://searxng:8080
    // is a Docker-internal hostname). If both compete for port 8080, the
    // local gateway wins but search tools fail.
    //
    // Order of operations:
    //   Docker mode:  docker compose up → port 8080 handled → don't start local
    //   Local mode:   start local gateway → needs SEARXNG_URL configured
  },
  // ── AI Services ──
  {
    id: 'ollama',
    name: 'Ollama LLM Server',
    command: 'ollama',
    args: ['serve'],
    cwd: PROJECT_ROOT,
    port: 11434,
    healthCheck: 'http://127.0.0.1:11434/',
    gracefulTimeout: 10000,
    group: 'ai',
  },
  // ── Applications ──
  {
    id: 'artifex-cli',
    name: 'Artifex CLI',
    command: path.join(PROJECT_ROOT, 'venv', 'Scripts', 'python.exe'),
    args: ['main.py'],
    cwd: PROJECT_ROOT,
    port: 0,
    gracefulTimeout: 5000,
    group: 'apps',
    openTerminal: true,
  },
  {
    id: 'artifex-gui',
    name: 'Artifex GUI',
    command: path.join(PROJECT_ROOT, 'venv', 'Scripts', 'python.exe'),
    args: ['main_gui_qt.py'],
    cwd: PROJECT_ROOT,
    port: 0,
    gracefulTimeout: 5000,
    group: 'apps',
  },
];
