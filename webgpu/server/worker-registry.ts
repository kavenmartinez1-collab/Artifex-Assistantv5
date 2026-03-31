/**
 * Worker Registry — Tracks connected GPU workers and their state.
 */

import type { WebSocket } from 'ws';
import type {
  WorkerState, WorkerStatus, GPUInfo, ModelInfo, WSMessage,
  WorkerRegisterPayload, WorkerHeartbeatPayload, ModelLoadedPayload,
} from './protocol.js';

export interface ConnectedWorker extends WorkerState {
  ws: WebSocket;
}

export class WorkerRegistry {
  private workers = new Map<string, ConnectedWorker>();
  private onChange: () => void;

  constructor(onChange: () => void) {
    this.onChange = onChange;
  }

  /** Register a new worker connection. */
  register(ws: WebSocket, payload: WorkerRegisterPayload): ConnectedWorker {
    const existing = this.workers.get(payload.workerId);
    if (existing) {
      // Worker reconnected — close old socket, keep stats
      try { existing.ws.close(); } catch {}
      console.log(`[Registry] Worker ${payload.workerId} reconnected`);
    }

    const worker: ConnectedWorker = {
      id: payload.workerId,
      ws,
      gpu: payload.gpu,
      status: 'idle',
      model: null,
      currentTaskId: null,
      connectedAt: Date.now(),
      lastHeartbeat: Date.now(),
      stats: existing?.stats ?? {
        tasksCompleted: 0,
        totalTokens: 0,
        avgTokPerSec: 0,
      },
    };

    this.workers.set(payload.workerId, worker);
    console.log(`[Registry] Worker ${payload.workerId} registered: ${payload.gpu.device} (${payload.gpu.vendor})`);
    this.onChange();
    return worker;
  }

  /** Update worker from heartbeat. */
  heartbeat(workerId: string, payload: WorkerHeartbeatPayload): void {
    const worker = this.workers.get(workerId);
    if (!worker) return;

    worker.lastHeartbeat = Date.now();
    worker.status = payload.status;
    worker.currentTaskId = payload.currentTaskId;
  }

  /** Mark worker as having loaded a model. */
  modelLoaded(workerId: string, payload: ModelLoadedPayload): void {
    const worker = this.workers.get(workerId);
    if (!worker) return;

    worker.model = payload.model;
    worker.status = 'idle';
    console.log(`[Registry] Worker ${workerId} loaded model: ${payload.model.repo} (${payload.model.gpuMemoryMB} MB)`);
    this.onChange();
  }

  /** Mark worker as having unloaded its model. */
  modelUnloaded(workerId: string): void {
    const worker = this.workers.get(workerId);
    if (!worker) return;

    worker.model = null;
    worker.status = 'idle';
    console.log(`[Registry] Worker ${workerId} unloaded model`);
    this.onChange();
  }

  /** Update worker status (e.g., generating, idle). */
  setStatus(workerId: string, status: WorkerStatus, taskId?: string | null): void {
    const worker = this.workers.get(workerId);
    if (!worker) return;

    worker.status = status;
    if (taskId !== undefined) worker.currentTaskId = taskId;
  }

  /** Record a completed task for stats. */
  recordTaskComplete(workerId: string, tokenCount: number, tokPerSec: number): void {
    const worker = this.workers.get(workerId);
    if (!worker) return;

    worker.stats.tasksCompleted++;
    worker.stats.totalTokens += tokenCount;
    // Running average of tok/s
    const n = worker.stats.tasksCompleted;
    worker.stats.avgTokPerSec = worker.stats.avgTokPerSec * ((n - 1) / n) + tokPerSec / n;
    worker.status = 'idle';
    worker.currentTaskId = null;
  }

  /** Remove a disconnected worker. */
  disconnect(ws: WebSocket): string | null {
    for (const [id, worker] of this.workers) {
      if (worker.ws === ws) {
        this.workers.delete(id);
        console.log(`[Registry] Worker ${id} disconnected`);
        this.onChange();
        return id;
      }
    }
    return null;
  }

  /** Find an idle worker, optionally with a specific model loaded. */
  findIdleWorker(requiredModel?: string): ConnectedWorker | null {
    // First: idle workers with the required model already loaded
    if (requiredModel) {
      for (const worker of this.workers.values()) {
        if (worker.status === 'idle' && worker.model?.repo === requiredModel) {
          return worker;
        }
      }
    }
    // Fallback: any idle worker
    for (const worker of this.workers.values()) {
      if (worker.status === 'idle') return worker;
    }
    return null;
  }

  /** Get all workers as serializable state (no WebSocket refs). */
  getStates(): WorkerState[] {
    return [...this.workers.values()].map(({ ws, ...state }) => state);
  }

  /** Get a specific worker. */
  get(workerId: string): ConnectedWorker | undefined {
    return this.workers.get(workerId);
  }

  /** Number of connected workers. */
  get size(): number {
    return this.workers.size;
  }

  /** Prune workers that haven't sent a heartbeat in 30s. */
  pruneStale(timeoutMs = 30_000): void {
    const now = Date.now();
    for (const [id, worker] of this.workers) {
      if (now - worker.lastHeartbeat > timeoutMs) {
        console.log(`[Registry] Pruning stale worker ${id} (no heartbeat for ${timeoutMs / 1000}s)`);
        try { worker.ws.close(); } catch {}
        this.workers.delete(id);
        this.onChange();
      }
    }
  }
}
