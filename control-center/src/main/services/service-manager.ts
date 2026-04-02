import { ChildProcess, spawn, execSync } from 'child_process';
import { SERVICE_DEFINITIONS, ServiceDefinition } from './service-config';
import { isPortInUse } from './port-scanner';
import { LogAggregator } from '../logs/log-aggregator';

export type ServiceStatus = 'stopped' | 'starting' | 'running' | 'stopping' | 'error';

export interface ServiceStatusInfo {
  id: string;
  name: string;
  status: ServiceStatus;
  pid: number | null;
  port: number;
  startTime: number | null;
  errorMessage: string | null;
}

interface ManagedProcess {
  definition: ServiceDefinition;
  process: ChildProcess | null;
  status: ServiceStatus;
  pid: number | null;
  startTime: number | null;
  errorMessage: string | null;
}

export type StatusChangeCallback = (status: ServiceStatusInfo) => void;

export class ServiceManager {
  private processes: Map<string, ManagedProcess> = new Map();
  private logAggregator: LogAggregator;
  private onStatusChange: StatusChangeCallback;

  constructor(logAggregator: LogAggregator, onStatusChange: StatusChangeCallback) {
    this.logAggregator = logAggregator;
    this.onStatusChange = onStatusChange;

    // Initialize managed process entries for each defined service
    for (const def of SERVICE_DEFINITIONS) {
      this.processes.set(def.id, {
        definition: def,
        process: null,
        status: 'stopped',
        pid: null,
        startTime: null,
        errorMessage: null,
      });
    }

    // Auto-detect already-running services by scanning their ports
    this.detectRunningServices();
  }

  /** Scan ports to detect services that are already running externally. */
  async detectRunningServices(): Promise<void> {
    for (const [id, mp] of this.processes) {
      if (mp.definition.port > 0) {
        const busy = await isPortInUse(mp.definition.port);
        if (busy) {
          mp.status = 'running';
          mp.startTime = Date.now();
          mp.errorMessage = null;
          this.emitStatus(id);
          this.logAggregator.addEntry(id, 'info',
            `Detected running on port ${mp.definition.port}`);
        }
      }
    }
  }

  private emitStatus(id: string): void {
    const mp = this.processes.get(id);
    if (!mp) return;
    this.onStatusChange({
      id: mp.definition.id,
      name: mp.definition.name,
      status: mp.status,
      pid: mp.pid,
      port: mp.definition.port,
      startTime: mp.startTime,
      errorMessage: mp.errorMessage,
    });
  }

  private setStatus(id: string, status: ServiceStatus, error?: string): void {
    const mp = this.processes.get(id);
    if (!mp) return;
    mp.status = status;
    if (error !== undefined) mp.errorMessage = error;
    if (status === 'stopped' || status === 'error') {
      mp.process = null;
      mp.pid = null;
      mp.startTime = null;
    }
    this.emitStatus(id);
  }

  /**
   * Start a service, optionally with config overrides from the UI.
   * @param id - Service ID
   * @param configOverrides - Map of option key to value (e.g. {'--port': '9000', '--backend': 'ollama'})
   */
  async start(id: string, configOverrides?: Record<string, string>): Promise<void> {
    const mp = this.processes.get(id);
    if (!mp) throw new Error(`Unknown service: ${id}`);
    if (mp.status === 'running' || mp.status === 'starting') return;

    const def = mp.definition;

    // Build args: base args + all configured options (defaults included).
    // Options are always passed so the service gets the full config —
    // not just overrides. Empty strings are skipped.
    let args = [...def.args];
    const extraEnv: Record<string, string> = {};
    if (def.options) {
      for (const opt of def.options) {
        // Use override if provided, otherwise use default
        const val = configOverrides?.[opt.key] ?? opt.default;
        if (val !== undefined && val !== '') {
          args.push(opt.key, val);
          if (opt.envVar) {
            extraEnv[opt.envVar] = val;
          }
        }
      }
    }

    // Determine effective port (may be overridden)
    let effectivePort = def.port;
    if (configOverrides?.['--port']) {
      effectivePort = parseInt(configOverrides['--port']) || def.port;
    }

    // Check if port is already in use — adopt the existing service
    if (effectivePort > 0) {
      const portBusy = await isPortInUse(effectivePort);
      if (portBusy) {
        mp.status = 'running';
        mp.startTime = Date.now();
        mp.errorMessage = null;
        mp.process = null;
        mp.pid = null;
        this.emitStatus(id);
        this.logAggregator.addEntry(def.id, 'info',
          `Adopted existing service on port ${effectivePort} (already running)`);
        return;
      }
    }

    this.setStatus(id, 'starting');
    mp.errorMessage = null;

    const spawnEnv = { ...process.env, ...extraEnv };

    try {
      // Interactive apps (CLI) get their own visible terminal window
      if (def.openTerminal) {
        const fullCmd = `"${def.command}" ${args.join(' ')}`;
        const child = spawn('powershell.exe', [
          '-NoProfile', '-Command',
          `Start-Process -FilePath cmd.exe -ArgumentList '/k cd /d "${def.cwd}" && ${fullCmd}' -WindowStyle Normal`,
        ], {
          cwd: def.cwd,
          shell: false,
          env: spawnEnv,
          stdio: 'ignore',
        });
        mp.process = null;
        mp.pid = child.pid ?? null;
        mp.startTime = Date.now();
        mp.status = 'running';
        this.emitStatus(id);
        return;
      }

      const child = spawn(def.command, args, {
        cwd: def.cwd,
        shell: false,
        env: spawnEnv,
        stdio: ['ignore', 'pipe', 'pipe'],
        windowsHide: true,
      });

      mp.process = child;
      mp.pid = child.pid ?? null;
      mp.startTime = Date.now();

      // Attach stdout/stderr to log aggregator
      this.logAggregator.attach(def.id, child);

      // Mark as running once we get first stdout data or after a short delay
      let promoted = false;
      const promoteToRunning = () => {
        if (!promoted && mp.status === 'starting') {
          promoted = true;
          mp.status = 'running';
          this.emitStatus(id);
        }
      };

      if (child.stdout) {
        child.stdout.once('data', promoteToRunning);
      }
      // Fallback: promote after 2 seconds if still starting
      setTimeout(promoteToRunning, 2000);

      child.on('error', (err) => {
        this.setStatus(id, 'error', err.message);
      });

      child.on('exit', (code, signal) => {
        if (mp.status === 'stopping') {
          this.setStatus(id, 'stopped');
        } else if (code !== 0) {
          this.setStatus(
            id,
            'error',
            `Exited with code ${code}${signal ? ` (signal: ${signal})` : ''}`
          );
        } else {
          this.setStatus(id, 'stopped');
        }
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this.setStatus(id, 'error', msg);
    }
  }

  /** Find and kill a process by the port it's listening on (Windows). */
  private async killByPort(port: number): Promise<boolean> {
    if (port <= 0) return false;
    try {
      // Get PIDs listening on this port
      const result = execSync(
        `powershell.exe -NoProfile -Command "(Get-NetTCPConnection -LocalPort ${port} -State Listen -ErrorAction SilentlyContinue).OwningProcess"`,
        { encoding: 'utf-8', timeout: 5000 }
      ).trim();
      const pids = result.split('\n').map(s => parseInt(s.trim())).filter(n => n > 0);
      for (const pid of pids) {
        try {
          // Use PowerShell Stop-Process (taskkill doesn't work in PS)
          execSync(
            `powershell.exe -NoProfile -Command "Stop-Process -Id ${pid} -Force -ErrorAction SilentlyContinue"`,
            { timeout: 5000 }
          );
        } catch { /* process may already be dead */ }
      }
      return pids.length > 0;
    } catch {
      return false;
    }
  }

  async stop(id: string): Promise<void> {
    const mp = this.processes.get(id);
    if (!mp) throw new Error(`Unknown service: ${id}`);
    if (mp.status === 'stopped' || mp.status === 'stopping') return;

    const def = mp.definition;
    this.setStatus(id, 'stopping');

    // Strategy: tree-kill by PID first, then kill by port as safety net.
    // On Windows, SIGTERM only kills the parent — taskkill /T /F kills
    // the entire process tree including child workers.

    // Step 1: Tree-kill by PID (if we have one)
    if (mp.pid) {
      this.treeKill(mp.pid);
    } else if (mp.process) {
      try { mp.process.kill(); } catch { /* already dead */ }
    }

    // Step 2: Wait briefly for exit
    if (mp.process) {
      await new Promise<void>((resolve) => {
        const timer = setTimeout(resolve, 2000);
        mp.process!.once('exit', () => { clearTimeout(timer); resolve(); });
      });
    }

    // Step 3: Kill by port as safety net (catches orphaned workers)
    if (def.port > 0) {
      const killed = await this.killByPort(def.port);
      if (killed) {
        this.logAggregator.addEntry(id, 'info',
          `Cleaned up process on port ${def.port}`);
      }
    }

    // Step 4: Ollama-specific — kill orphaned runner subprocesses.
    // Ollama spawns "ollama.exe runner --ollama-engine --model <blob>"
    // processes that hold GPU VRAM. These can become zombies when the
    // Ollama server loses track of them. kill them all on stop.
    if (def.id === 'ollama') {
      await this.killOllamaRunners();
    }

    this.setStatus(id, 'stopped');
  }

  async restart(id: string): Promise<void> {
    await this.stop(id);
    // Small delay to let ports free up
    await new Promise((r) => setTimeout(r, 500));
    await this.start(id);
  }

  async startAll(): Promise<void> {
    const promises = SERVICE_DEFINITIONS.map((def) => this.start(def.id));
    await Promise.allSettled(promises);
  }

  async stopAll(): Promise<void> {
    // Stop all services with a hard timeout to prevent hanging on exit
    const stopPromise = Promise.allSettled(
      SERVICE_DEFINITIONS.map((def) => this.stop(def.id))
    );
    const timeoutPromise = new Promise<void>((resolve) => setTimeout(resolve, 10000));
    await Promise.race([stopPromise, timeoutPromise]);

    // Final safety net: kill anything still on our known ports
    for (const def of SERVICE_DEFINITIONS) {
      if (def.port > 0) {
        await this.killByPort(def.port).catch(() => {});
      }
    }

    // Kill any orphaned Ollama runners holding VRAM
    await this.killOllamaRunners().catch(() => {});
  }

  getStatus(): ServiceStatusInfo[] {
    return SERVICE_DEFINITIONS.map((def) => {
      const mp = this.processes.get(def.id)!;
      return {
        id: def.id,
        name: def.name,
        status: mp.status,
        pid: mp.pid,
        port: def.port,
        startTime: mp.startTime,
        errorMessage: mp.errorMessage,
      };
    });
  }

  /**
   * Kill orphaned Ollama runner processes that hold GPU VRAM.
   * These are subprocesses ("ollama.exe runner --ollama-engine --model ...")
   * that can survive after the Ollama server stops or loses track of them.
   */
  private async killOllamaRunners(): Promise<void> {
    if (process.platform !== 'win32') return;
    try {
      // Find all Ollama processes with "runner" in the command line
      const result = execSync(
        `powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \\"Name='ollama.exe'\\" | Where-Object { $_.CommandLine -match 'runner' } | Select-Object -ExpandProperty ProcessId"`,
        { encoding: 'utf-8', timeout: 10000 }
      ).trim();
      const pids = result.split('\n').map(s => parseInt(s.trim())).filter(n => n > 0);
      for (const pid of pids) {
        try {
          execSync(
            `powershell.exe -NoProfile -Command "Stop-Process -Id ${pid} -Force -ErrorAction SilentlyContinue"`,
            { timeout: 5000 }
          );
          this.logAggregator.addEntry('ollama', 'info',
            `Killed orphaned Ollama runner (PID ${pid})`);
        } catch { /* already dead */ }
      }
      if (pids.length > 0) {
        this.logAggregator.addEntry('ollama', 'info',
          `Cleaned up ${pids.length} Ollama runner process(es) — VRAM freed`);
      }
    } catch {
      // PowerShell query failed — non-fatal
    }
  }

  /**
   * Kill a process tree. On Windows uses taskkill /T /F.
   * On POSIX sends SIGTERM to the process group.
   */
  private treeKill(pid: number): void {
    try {
      if (process.platform === 'win32') {
        execSync(`taskkill /PID ${pid} /T /F`, { stdio: 'ignore' });
      } else {
        process.kill(-pid, 'SIGTERM');
      }
    } catch {
      // Process may already be dead — ignore errors
    }
  }
}
