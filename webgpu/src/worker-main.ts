/**
 * Artifex GPU Worker — Entry point for worker browser windows.
 *
 * Initializes WebGPU, connects to the orchestration hub via WebSocket,
 * and waits for task assignments. Reuses the full inference pipeline.
 */

import { initWebGPU, type GPUContext } from './engine/gpu-device';
import type {
  WSMessage, WorkerRegisterPayload, WorkerHeartbeatPayload,
  ModelLoadedPayload, TaskAssignPayload, TaskTokenPayload,
  TaskCompletePayload, TaskErrorPayload, GPUInfo,
} from '../server/protocol';

// ── DOM ─────────────────────────────────────────────────────────────────

const $ = (id: string) => document.getElementById(id)!;
const workerIdEl = $('worker-id');
const gpuInfoEl = $('gpu-info');
const statusEl = $('status');
const hubStatusEl = $('hub-status');
const modelInfoEl = $('model-info');
const taskInfoEl = $('task-info');
const statsEl = $('stats');
const logEl = $('log');

function workerLog(msg: string): void {
  const time = new Date().toLocaleTimeString();
  const entry = document.createElement('div');
  entry.className = 'log-entry';
  entry.innerHTML = `<span class="log-time">[${time}]</span> ${msg}`;
  logEl.appendChild(entry);
  logEl.scrollTop = logEl.scrollHeight;
  // Keep last 200 entries
  while (logEl.children.length > 200) logEl.removeChild(logEl.firstChild!);
}

function setStatus(status: string, cls = ''): void {
  statusEl.textContent = status;
  statusEl.className = `value ${cls}`;
}

// ── State ───────────────────────────────────────────────────────────────

let gpu: GPUContext | null = null;
let ws: WebSocket | null = null;
let workerId = `worker-${Math.random().toString(36).slice(2, 6)}`;
let heartbeatInterval: ReturnType<typeof setInterval> | null = null;
let tasksCompleted = 0;
let totalTokens = 0;

// ── WebSocket ───────────────────────────────────────────────────────────

function createMessage<T>(type: string, payload: T): string {
  return JSON.stringify({
    type,
    id: `${workerId}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    timestamp: Date.now(),
    source: workerId,
    payload,
  });
}

function connectToHub(): void {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${protocol}//${location.host}/ws`;

  workerLog(`Connecting to hub: ${url}`);
  hubStatusEl.textContent = 'Connecting...';

  ws = new WebSocket(url);

  ws.onopen = () => {
    hubStatusEl.textContent = 'Connected';
    hubStatusEl.style.color = 'var(--success)';
    workerLog('Connected to orchestration hub');

    // Register with hub
    const gpuInfo: GPUInfo = {
      vendor: gpu!.adapterInfo.vendor || 'unknown',
      device: gpu!.adapterInfo.device || gpu!.adapterInfo.description || 'Unknown GPU',
      architecture: gpu!.adapterInfo.architecture || '',
      maxBufferMB: Math.round(gpu!.maxBufferSize / (1024 * 1024)),
    };

    const payload: WorkerRegisterPayload = { workerId, gpu: gpuInfo };
    ws!.send(createMessage('worker:register', payload));
    workerLog(`Registered as ${workerId}`);
    workerIdEl.textContent = workerId;

    // Start heartbeat
    if (heartbeatInterval) clearInterval(heartbeatInterval);
    heartbeatInterval = setInterval(() => {
      if (ws?.readyState === WebSocket.OPEN) {
        const hb: WorkerHeartbeatPayload = {
          status: statusEl.textContent?.toLowerCase().includes('generat') ? 'generating' : 'idle',
          currentTaskId: null,
        };
        ws.send(createMessage('worker:heartbeat', hb));
      }
    }, 10_000);
  };

  ws.onmessage = (event) => {
    try {
      const msg: WSMessage = JSON.parse(event.data);
      handleHubMessage(msg);
    } catch (err) {
      workerLog(`Invalid message from hub: ${err}`);
    }
  };

  ws.onclose = () => {
    hubStatusEl.textContent = 'Disconnected — reconnecting...';
    hubStatusEl.style.color = 'var(--error)';
    workerLog('Disconnected from hub, reconnecting in 3s...');
    if (heartbeatInterval) clearInterval(heartbeatInterval);
    setTimeout(connectToHub, 3000);
  };

  ws.onerror = (err) => {
    workerLog(`WebSocket error`);
  };
}

// ── Message Handling ────────────────────────────────────────────────────

async function handleHubMessage(msg: WSMessage): Promise<void> {
  switch (msg.type) {
    case 'task:assign':
      await handleTaskAssign(msg.payload as TaskAssignPayload);
      break;

    case 'worker:load-model':
      workerLog(`Hub requested model load: ${JSON.stringify(msg.payload)}`);
      break;

    case 'task:cancel':
      workerLog('Task cancelled by hub');
      setStatus('Idle', 'status-idle');
      break;

    default:
      workerLog(`Unknown message type: ${msg.type}`);
  }
}

async function handleTaskAssign(task: TaskAssignPayload): Promise<void> {
  workerLog(`Task assigned: ${task.taskId} — "${task.prompt.slice(0, 80)}..."`);
  setStatus('Generating...', 'status-generating');
  taskInfoEl.textContent = `${task.taskId}: ${task.prompt.slice(0, 60)}...`;

  const startTime = Date.now();
  let tokenCount = 0;
  let fullText = '';

  try {
    // TODO: Replace with actual inference session call
    // For now, send a placeholder response to prove the pipeline works
    workerLog('NOTE: Inference not yet wired — sending placeholder response');

    const placeholder = `[Worker ${workerId}] Received task: "${task.prompt.slice(0, 100)}".\nInference engine integration pending.`;
    fullText = placeholder;
    tokenCount = placeholder.split(' ').length;

    // Stream tokens
    for (const word of placeholder.split(' ')) {
      const tokenPayload: TaskTokenPayload = {
        taskId: task.taskId,
        token: word + ' ',
        tokenIndex: tokenCount++,
      };
      ws?.send(createMessage('task:token', tokenPayload));
      await new Promise(r => setTimeout(r, 50));
    }

    // Complete
    const durationMs = Date.now() - startTime;
    const tokPerSec = tokenCount / (durationMs / 1000);

    const completePayload: TaskCompletePayload = {
      taskId: task.taskId,
      fullText,
      tokenCount,
      durationMs,
      tokPerSec,
    };
    ws?.send(createMessage('task:complete', completePayload));

    tasksCompleted++;
    totalTokens += tokenCount;
    statsEl.textContent = `${tasksCompleted} tasks | ${totalTokens} tokens`;
    workerLog(`Task ${task.taskId} complete: ${tokenCount} tokens in ${(durationMs / 1000).toFixed(1)}s`);

  } catch (err) {
    const errorPayload: TaskErrorPayload = {
      taskId: task.taskId,
      error: err instanceof Error ? err.message : String(err),
    };
    ws?.send(createMessage('task:error', errorPayload));
    workerLog(`Task ${task.taskId} failed: ${errorPayload.error}`);
  }

  setStatus('Idle', 'status-idle');
  taskInfoEl.textContent = 'None';
}

// ── Init ────────────────────────────────────────────────────────────────

async function init(): Promise<void> {
  workerLog('Initializing WebGPU...');

  try {
    gpu = await initWebGPU();
    const label = gpu.adapterInfo.device || gpu.adapterInfo.description || 'Unknown GPU';
    const maxMB = Math.round(gpu.maxBufferSize / (1024 * 1024));
    gpuInfoEl.textContent = `${label} (${maxMB} MB)`;
    workerLog(`GPU: ${label} | ${gpu.adapterInfo.vendor} | ${maxMB} MB max buffer`);

    setStatus('Idle', 'status-idle');
    connectToHub();

  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    gpuInfoEl.textContent = 'No WebGPU';
    setStatus(`Error: ${msg}`, 'status-error');
    workerLog(`WebGPU init failed: ${msg}`);
  }
}

init();
