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

  async start(id: string): Promise<void> {
    const mp = this.processes.get(id);
    if (!mp) throw new Error(`Unknown service: ${id}`);
    if (mp.status === 'running' || mp.status === 'starting') return;

    const def = mp.definition;

    // Check if port is already in use — adopt the existing service
    if (def.port > 0) {
      const portBusy = await isPortInUse(def.port);
      if (portBusy) {
        mp.status = 'running';
        mp.startTime = Date.now();
        mp.errorMessage = null;
        mp.process = null;
        mp.pid = null; // Unknown PID for adopted process
        this.emitStatus(id);
        this.logAggregator.addEntry(def.id, 'info',
          `Adopted existing service on port ${def.port} (already running)`);
        return;
      }
    }

    this.setStatus(id, 'starting');
    mp.errorMessage = null;

    try {
      // Interactive apps (CLI) get their own visible terminal window
      if (def.openTerminal) {
        const fullCmd = `"${def.command}" ${def.args.join(' ')}`;
        const child = spawn('powershell.exe', [
          '-NoProfile', '-Command',
          `Start-Process -FilePath cmd.exe -ArgumentList '/k cd /d "${def.cwd}" && ${fullCmd}' -WindowStyle Normal`,
        ], {
          cwd: def.cwd,
          shell: false,
          env: { ...process.env },
          stdio: 'ignore',
        });
        mp.process = null;
        mp.pid = child.pid ?? null;
        mp.startTime = Date.now();
        mp.status = 'running';
        this.emitStatus(id);
        return;
      }

      const child = spawn(def.command, def.args, {
        cwd: def.cwd,
        shell: true,
        env: { ...process.env },
        stdio: ['ignore', 'pipe', 'pipe'],
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

    // Adopted service (no process handle) — kill by port
    if (!mp.process && !mp.pid) {
      if (def.port > 0) {
        this.setStatus(id, 'stopping');
        const killed = await this.killByPort(def.port);
        if (killed) {
          this.logAggregator.addEntry(id, 'info',
            `Killed adopted process on port ${def.port}`);
        }
      }
      this.setStatus(id, 'stopped');
      return;
    }

    this.setStatus(id, 'stopping');
    const pid = mp.pid!;
    const timeout = mp.definition.gracefulTimeout;

    // Try graceful kill first
    try {
      mp.process!.kill('SIGTERM');
    } catch {
      // Process may already be dead
    }

    // Wait for exit or force-kill after timeout
    await new Promise<void>((resolve) => {
      const timer = setTimeout(() => {
        this.treeKill(pid);
        resolve();
      }, timeout);

      if (mp.process) {
        mp.process.once('exit', () => {
          clearTimeout(timer);
          resolve();
        });
      } else {
        clearTimeout(timer);
        resolve();
      }
    });

    if ((mp.status as string) === 'stopping') {
      this.setStatus(id, 'stopped');
    }
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
    const promises = SERVICE_DEFINITIONS.map((def) => this.stop(def.id));
    await Promise.allSettled(promises);
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
