/**
 * WebSocket Hub — Central message router for multi-GPU orchestration.
 *
 * Attaches to the existing Express HTTP server. Routes messages between
 * the orchestrator browser, GPU workers, and the task router.
 */

import { WebSocketServer, WebSocket } from 'ws';
import type { Server } from 'http';
import {
  type WSMessage, type MessageType, type TaskSubmitPayload,
  type TaskAssignPayload, type TaskTokenPayload, type TaskCompletePayload,
  type TaskErrorPayload, type WorkerRegisterPayload, type WorkerHeartbeatPayload,
  type ModelLoadedPayload, type ClusterStatusPayload, type TaskState, type TaskStatus,
  createMessage,
} from './protocol.js';
import { WorkerRegistry, type ConnectedWorker } from './worker-registry.js';

export class OrchestrationHub {
  private wss: WebSocketServer;
  private registry: WorkerRegistry;
  private orchestratorWs: WebSocket | null = null;
  private taskQueue: TaskState[] = [];
  private taskCounter = 0;
  private startTime = Date.now();
  private heartbeatInterval: ReturnType<typeof setInterval>;

  constructor(server: Server) {
    this.registry = new WorkerRegistry(() => this.broadcastClusterStatus());

    this.wss = new WebSocketServer({ server, path: '/ws' });
    this.wss.on('connection', (ws) => this.handleConnection(ws));

    // Prune stale workers every 15s
    this.heartbeatInterval = setInterval(() => this.registry.pruneStale(), 15_000);

    console.log(`\x1b[35m[WS Hub] WebSocket orchestration hub ready on /ws\x1b[0m`);
  }

  private handleConnection(ws: WebSocket): void {
    console.log('[WS Hub] New connection');

    ws.on('message', (raw) => {
      try {
        const msg: WSMessage = JSON.parse(raw.toString());
        this.handleMessage(ws, msg);
      } catch (err) {
        console.error('[WS Hub] Invalid message:', err);
      }
    });

    ws.on('close', () => {
      if (ws === this.orchestratorWs) {
        console.log('[WS Hub] Orchestrator disconnected');
        this.orchestratorWs = null;
      } else {
        const workerId = this.registry.disconnect(ws);
        if (workerId) {
          // Re-queue any tasks assigned to this worker
          for (const task of this.taskQueue) {
            if (task.assignedWorker === workerId && task.status === 'running') {
              task.status = 'queued';
              task.assignedWorker = null;
              console.log(`[WS Hub] Re-queued task ${task.id} (worker ${workerId} disconnected)`);
            }
          }
          this.processQueue();
        }
      }
    });

    ws.on('error', (err) => {
      console.error('[WS Hub] WebSocket error:', err.message);
    });
  }

  private handleMessage(ws: WebSocket, msg: WSMessage): void {
    switch (msg.type) {
      // ── Orchestrator messages ──────────────────────────────────────
      case 'orchestrator:connect':
        this.orchestratorWs = ws;
        console.log('[WS Hub] Orchestrator connected');
        this.sendClusterStatus(ws);
        break;

      case 'task:submit':
        this.handleTaskSubmit(msg as WSMessage<TaskSubmitPayload>);
        break;

      // ── Worker messages ───────────────────────────────────────────
      case 'worker:register':
        this.registry.register(ws, msg.payload as WorkerRegisterPayload);
        this.processQueue(); // Check if queued tasks can now be assigned
        break;

      case 'worker:heartbeat':
        this.registry.heartbeat(msg.source, msg.payload as WorkerHeartbeatPayload);
        break;

      case 'worker:model-loaded':
        this.registry.modelLoaded(msg.source, msg.payload as ModelLoadedPayload);
        this.processQueue();
        break;

      case 'worker:model-unloaded':
        this.registry.modelUnloaded(msg.source);
        break;

      case 'task:token':
        this.forwardToOrchestrator(msg);
        break;

      case 'task:complete':
        this.handleTaskComplete(msg as WSMessage<TaskCompletePayload>);
        break;

      case 'task:error':
        this.handleTaskError(msg as WSMessage<TaskErrorPayload>);
        break;

      default:
        console.warn(`[WS Hub] Unknown message type: ${msg.type}`);
    }
  }

  // ── Task Management ─────────────────────────────────────────────────

  private handleTaskSubmit(msg: WSMessage<TaskSubmitPayload>): void {
    const payload = msg.payload;
    const taskId = `task-${++this.taskCounter}`;

    const task: TaskState = {
      id: taskId,
      status: 'queued',
      prompt: payload.prompt,
      assignedWorker: null,
      submittedAt: Date.now(),
    };
    this.taskQueue.push(task);
    console.log(`[WS Hub] Task ${taskId} queued: "${payload.prompt.slice(0, 50)}..."`);

    this.broadcastClusterStatus();
    this.processQueue();
  }

  private processQueue(): void {
    for (const task of this.taskQueue) {
      if (task.status !== 'queued') continue;

      const worker = this.registry.findIdleWorker();
      if (!worker) break; // No idle workers

      this.assignTask(task, worker);
    }
  }

  private assignTask(task: TaskState, worker: ConnectedWorker): void {
    task.status = 'running';
    task.assignedWorker = worker.id;
    task.startedAt = Date.now();

    this.registry.setStatus(worker.id, 'generating', task.id);

    // Find the original submit payload from the task
    const assignPayload: TaskAssignPayload = {
      taskId: task.id,
      prompt: task.prompt,
      sampling: { maxTokens: 2048, temperature: 0.7 }, // defaults
    };

    const msg = createMessage('task:assign', 'hub', assignPayload);
    this.send(worker.ws, msg);

    console.log(`[WS Hub] Task ${task.id} assigned to worker ${worker.id}`);
    this.broadcastClusterStatus();
  }

  private handleTaskComplete(msg: WSMessage<TaskCompletePayload>): void {
    const { taskId, fullText, tokenCount, durationMs, tokPerSec } = msg.payload;
    const task = this.taskQueue.find(t => t.id === taskId);
    if (task) {
      task.status = 'complete';
      task.completedAt = Date.now();
      task.tokenCount = tokenCount;
      task.result = fullText;
    }

    this.registry.recordTaskComplete(msg.source, tokenCount, tokPerSec);
    this.forwardToOrchestrator(msg);
    this.broadcastClusterStatus();
    this.processQueue(); // Worker is now idle, check for queued tasks

    console.log(`[WS Hub] Task ${taskId} complete: ${tokenCount} tokens at ${tokPerSec.toFixed(1)} tok/s`);
  }

  private handleTaskError(msg: WSMessage<TaskErrorPayload>): void {
    const { taskId, error } = msg.payload;
    const task = this.taskQueue.find(t => t.id === taskId);
    if (task) {
      task.status = 'error';
      task.completedAt = Date.now();
      task.error = error;
    }

    if (task?.assignedWorker) {
      this.registry.setStatus(task.assignedWorker, 'idle', null);
    }

    this.forwardToOrchestrator(msg);
    this.broadcastClusterStatus();
    this.processQueue();

    console.error(`[WS Hub] Task ${taskId} failed: ${error}`);
  }

  // ── Communication ───────────────────────────────────────────────────

  private send(ws: WebSocket, msg: WSMessage): void {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(msg));
    }
  }

  private forwardToOrchestrator(msg: WSMessage): void {
    if (this.orchestratorWs) {
      this.send(this.orchestratorWs, msg);
    }
  }

  private sendClusterStatus(ws: WebSocket): void {
    const payload: ClusterStatusPayload = {
      workers: this.registry.getStates(),
      tasks: this.taskQueue.slice(-100), // Last 100 tasks
      hubUptime: Date.now() - this.startTime,
    };
    this.send(ws, createMessage('cluster:status', 'hub', payload));
  }

  private broadcastClusterStatus(): void {
    if (this.orchestratorWs) {
      this.sendClusterStatus(this.orchestratorWs);
    }
  }

  // ── Cleanup ─────────────────────────────────────────────────────────

  close(): void {
    clearInterval(this.heartbeatInterval);
    this.wss.close();
  }
}
