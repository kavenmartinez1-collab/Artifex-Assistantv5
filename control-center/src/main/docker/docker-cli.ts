// Docker CLI wrapper — talks to Docker Engine via child_process.execFile
// Gracefully handles Docker not being installed.

import { execFile } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

// ── Types ──

export interface ContainerInfo {
  id: string;
  name: string;
  image: string;
  status: 'running' | 'stopped' | 'restarting' | 'created';
  ports: string[];      // ["8080:80", "3000:3000"]
  cpu: string;          // "0.5%"
  memory: string;       // "128.3 MiB / 512 MiB"
  uptime: string;       // "2 hours ago"
}

export interface ComposeService {
  name: string;
  image: string;
  ports: string[];
  volumes: string[];
}

// ── Helpers ──

/**
 * Resolve the Docker CLI path. execFile without shell can't find docker
 * if Docker Desktop was started after the Electron app (PATH not inherited).
 * Try common install locations as fallback.
 */
let _resolvedDockerPath: string | null = null;
function resolveDockerPath(): string {
  if (_resolvedDockerPath) return _resolvedDockerPath;

  // Check common Docker Desktop install paths on Windows
  if (process.platform === 'win32') {
    const candidates = [
      'docker',  // try PATH first
      path.join(process.env.ProgramFiles || 'C:\\Program Files', 'Docker', 'Docker', 'resources', 'bin', 'docker.exe'),
      path.join(process.env.LOCALAPPDATA || '', 'Docker', 'resources', 'bin', 'docker.exe'),
      path.join('C:\\Program Files', 'Docker', 'Docker', 'resources', 'bin', 'docker.exe'),
    ];
    for (const p of candidates) {
      if (p === 'docker' || fs.existsSync(p)) {
        _resolvedDockerPath = p;
        return p;
      }
    }
  }

  _resolvedDockerPath = 'docker';
  return 'docker';
}

/** Run a command and return stdout. Rejects on non-zero exit or missing binary. */
function run(cmd: string, args: string[], options?: { cwd?: string; timeout?: number }): Promise<string> {
  // Auto-resolve docker path for Docker commands
  const resolvedCmd = cmd === 'docker' ? resolveDockerPath() : cmd;

  return new Promise((resolve, reject) => {
    execFile(resolvedCmd, args, {
      timeout: options?.timeout ?? 30000,
      maxBuffer: 4 * 1024 * 1024,
      cwd: options?.cwd,
    }, (err, stdout, stderr) => {
      if (err) {
        if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
          // Reset resolved path so it retries next time (Docker may start later)
          _resolvedDockerPath = null;
          reject(new Error(`${cmd} is not installed or not in PATH`));
        } else {
          reject(new Error(stderr?.trim() || err.message));
        }
        return;
      }
      resolve(stdout);
    });
  });
}

/** Normalise Docker's status string into our enum. */
function normaliseStatus(raw: string): ContainerInfo['status'] {
  const lower = raw.toLowerCase();
  if (lower.startsWith('up') || lower.includes('running')) return 'running';
  if (lower.includes('restarting')) return 'restarting';
  if (lower.includes('created')) return 'created';
  return 'stopped';
}

/** Parse a "Ports" string from docker ps into clean mappings.
 *  e.g. "0.0.0.0:8080->80/tcp, :::3000->3000/tcp" -> ["8080:80", "3000:3000"]
 */
function parsePorts(portsStr: string): string[] {
  if (!portsStr || portsStr.trim() === '') return [];
  const mappings: string[] = [];
  const parts = portsStr.split(',').map(s => s.trim());
  for (const part of parts) {
    // Match host:port->container/proto
    const m = part.match(/(?:\d+\.\d+\.\d+\.\d+|:::?)(\d+)->(\d+)/);
    if (m) {
      const mapping = `${m[1]}:${m[2]}`;
      if (!mappings.includes(mapping)) mappings.push(mapping);
    }
  }
  return mappings;
}

// ── Exported API ──

/**
 * Check if Docker CLI is installed and the daemon is reachable.
 */
export async function isDockerInstalled(): Promise<boolean> {
  try {
    await run('docker', ['info', '--format', '{{.ServerVersion}}']);
    return true;
  } catch {
    return false;
  }
}

/**
 * List all containers (running and stopped) with resource usage.
 */
export async function listContainers(): Promise<ContainerInfo[]> {
  // Get container list with --format json (one JSON object per line)
  const psOutput = await run('docker', [
    'ps', '-a', '--format',
    '{{json .}}',
  ]);

  if (!psOutput.trim()) return [];

  // Parse each line as a JSON object
  const containers: ContainerInfo[] = [];
  const lines = psOutput.trim().split('\n');
  for (const line of lines) {
    if (!line.trim()) continue;
    try {
      const obj = JSON.parse(line);
      containers.push({
        id: obj.ID || '',
        name: (obj.Names || '').replace(/^\//, ''),
        image: obj.Image || '',
        status: normaliseStatus(obj.Status || ''),
        ports: parsePorts(obj.Ports || ''),
        cpu: '--',
        memory: '--',
        uptime: obj.Status || '',
      });
    } catch {
      // Skip malformed lines
    }
  }

  // Merge resource stats for running containers
  try {
    const statsOutput = await run('docker', [
      'stats', '--no-stream', '--format',
      '{{json .}}',
    ]);
    if (statsOutput.trim()) {
      const statsLines = statsOutput.trim().split('\n');
      const statsMap = new Map<string, { cpu: string; memory: string }>();
      for (const sl of statsLines) {
        if (!sl.trim()) continue;
        try {
          const s = JSON.parse(sl);
          const name = (s.Name || '').replace(/^\//, '');
          statsMap.set(name, {
            cpu: s.CPUPerc || '--',
            memory: s.MemUsage || '--',
          });
          // Also index by ID prefix
          if (s.ID) statsMap.set(s.ID, { cpu: s.CPUPerc || '--', memory: s.MemUsage || '--' });
        } catch {
          // skip
        }
      }
      for (const c of containers) {
        const stats = statsMap.get(c.name) || statsMap.get(c.id);
        if (stats) {
          c.cpu = stats.cpu;
          c.memory = stats.memory;
        }
      }
    }
  } catch {
    // stats fail for stopped containers, that's fine
  }

  return containers;
}

/**
 * Start a stopped container by name or ID.
 */
export async function startContainer(nameOrId: string): Promise<void> {
  await run('docker', ['start', nameOrId]);
}

/**
 * Stop a running container by name or ID.
 */
export async function stopContainer(nameOrId: string): Promise<void> {
  await run('docker', ['stop', nameOrId]);
}

/**
 * Run docker compose up -d from the given compose file path.
 */
export async function composeUp(composePath: string): Promise<void> {
  const dir = path.dirname(composePath);
  const file = path.basename(composePath);

  // Try "docker compose" (v2 plugin) first, fall back to "docker-compose" (standalone)
  try {
    await run('docker', ['compose', '-f', file, 'up', '-d', '--remove-orphans'], { cwd: dir, timeout: 120000 });
  } catch (e1) {
    const msg = e1 instanceof Error ? e1.message : '';
    // If the compose subcommand is unrecognized, try the standalone binary
    if (msg.includes('is not a docker command') || msg.includes('ENOENT')) {
      await run('docker-compose', ['-f', file, 'up', '-d', '--remove-orphans'], { cwd: dir, timeout: 120000 });
    } else {
      throw e1;
    }
  }
}

/**
 * Run docker compose down from the given compose file path.
 */
export async function composeDown(composePath: string): Promise<void> {
  const dir = path.dirname(composePath);
  const file = path.basename(composePath);

  try {
    await run('docker', ['compose', '-f', file, 'down', '--remove-orphans'], { cwd: dir, timeout: 60000 });
  } catch (e1) {
    const msg = e1 instanceof Error ? e1.message : '';
    if (msg.includes('is not a docker command') || msg.includes('ENOENT')) {
      await run('docker-compose', ['-f', file, 'down', '--remove-orphans'], { cwd: dir, timeout: 60000 });
    } else {
      throw e1;
    }
  }
}

/**
 * Parse a docker-compose.yml file into service definitions.
 * Uses simple line-by-line indentation parsing — no npm YAML library.
 */
export async function parseComposeFile(composePath: string): Promise<ComposeService[]> {
  const content = await fs.promises.readFile(composePath, 'utf-8');
  const lines = content.split(/\r?\n/);

  const services: ComposeService[] = [];
  let inServices = false;
  let currentService: ComposeService | null = null;
  let currentBlock: 'ports' | 'volumes' | null = null;
  let servicesIndent = -1;
  let serviceNameIndent = -1;
  let blockIndent = -1;

  for (const rawLine of lines) {
    // Skip empty lines and comments
    if (rawLine.trim() === '' || rawLine.trim().startsWith('#')) continue;

    const indent = rawLine.length - rawLine.trimStart().length;
    const trimmed = rawLine.trim();

    // Detect top-level "services:" key
    if (/^services\s*:/.test(trimmed) && indent === 0) {
      inServices = true;
      servicesIndent = 0;
      currentService = null;
      currentBlock = null;
      continue;
    }

    if (!inServices) continue;

    // If we encounter another top-level key (volumes:, networks:, etc.), stop parsing services
    if (indent === 0 && /^\S+\s*:/.test(trimmed)) {
      inServices = false;
      if (currentService) services.push(currentService);
      currentService = null;
      continue;
    }

    // Service name detection: first level of indentation under "services:"
    // A service name line looks like "  myservice:" — key with colon, not a list item
    if (!trimmed.startsWith('-') && trimmed.endsWith(':') && indent > servicesIndent) {
      // Check if this is a new service (indent equals the service-name indent we've seen before,
      // or it's the first service name we're encountering)
      if (serviceNameIndent < 0 || indent === serviceNameIndent) {
        // Save previous service
        if (currentService) services.push(currentService);
        serviceNameIndent = indent;
        currentService = {
          name: trimmed.replace(/:$/, '').trim(),
          image: '',
          ports: [],
          volumes: [],
        };
        currentBlock = null;
        continue;
      }
    }

    if (!currentService) continue;

    // Properties inside a service
    if (trimmed.startsWith('image:')) {
      currentService.image = trimmed.replace(/^image:\s*/, '').replace(/["']/g, '').trim();
      currentBlock = null;
      continue;
    }

    if (trimmed === 'ports:') {
      currentBlock = 'ports';
      blockIndent = indent;
      continue;
    }

    if (trimmed === 'volumes:') {
      currentBlock = 'volumes';
      blockIndent = indent;
      continue;
    }

    // If we see any other key at the same indent as a block key, reset the block
    if (currentBlock && indent <= blockIndent && !trimmed.startsWith('-')) {
      currentBlock = null;
      // Fall through to re-parse this line as a property
      if (trimmed.startsWith('image:')) {
        currentService.image = trimmed.replace(/^image:\s*/, '').replace(/["']/g, '').trim();
        continue;
      }
    }

    // List items inside ports/volumes
    if (currentBlock && trimmed.startsWith('-') && indent > blockIndent) {
      const value = trimmed.replace(/^-\s*/, '').replace(/["']/g, '').trim();
      if (value) {
        if (currentBlock === 'ports') {
          currentService.ports.push(value);
        } else if (currentBlock === 'volumes') {
          currentService.volumes.push(value);
        }
      }
      continue;
    }

    // If indent drops back to service-name level or above, save current service
    if (indent <= serviceNameIndent && !trimmed.startsWith('-')) {
      currentBlock = null;
      // This might be a new service name
      if (!trimmed.startsWith('-') && trimmed.endsWith(':') && indent === serviceNameIndent) {
        services.push(currentService);
        currentService = {
          name: trimmed.replace(/:$/, '').trim(),
          image: '',
          ports: [],
          volumes: [],
        };
      }
    }
  }

  // Push last service
  if (currentService) services.push(currentService);

  return services;
}

/**
 * Get the last N lines of logs from a container.
 */
export async function getContainerLogs(nameOrId: string, tail: number = 200): Promise<string> {
  const output = await run('docker', ['logs', '--tail', String(tail), '--timestamps', nameOrId]);
  return output;
}
