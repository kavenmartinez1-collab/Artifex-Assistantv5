/**
 * Orchestration Protocol — Message types for the WebSocket hub.
 *
 * All communication between the orchestrator browser, GPU workers,
 * and the hub uses these typed JSON messages.
 */

// ── Message Types ─────────────────────────────────────────────────────────

export type MessageType =
  // Worker → Hub
  | 'worker:register'       // Worker announces itself with GPU info
  | 'worker:heartbeat'      // Periodic alive signal with status
  | 'worker:model-loaded'   // Worker finished loading a model
  | 'worker:model-unloaded' // Worker unloaded its model
  // Hub → Worker
  | 'task:assign'           // Hub assigns a task to a worker
  | 'task:cancel'           // Hub cancels a running task
  | 'worker:load-model'     // Hub tells worker to load a specific model
  // Worker → Hub → Orchestrator
  | 'task:token'            // Streaming token from generation
  | 'task:complete'         // Generation finished
  | 'task:error'            // Generation failed
  // Orchestrator → Hub
  | 'orchestrator:connect'  // Orchestrator browser connected
  | 'task:submit'           // Orchestrator submits a new task
  // Hub → Orchestrator
  | 'cluster:status'        // Full cluster state update
  | 'task:update';          // Individual task status change

// ── Message Envelope ──────────────────────────────────────────────────────

export interface WSMessage<T = unknown> {
  type: MessageType;
  id: string;               // Unique message ID
  timestamp: number;        // Date.now()
  source: string;           // Sender: "orchestrator", "worker-0", "hub"
  payload: T;
}

// ── Payload Types ─────────────────────────────────────────────────────────

export interface GPUInfo {
  vendor: string;
  device: string;
  architecture: string;
  maxBufferMB: number;
}

export interface WorkerRegisterPayload {
  workerId: string;
  gpu: GPUInfo;
}

export interface WorkerHeartbeatPayload {
  status: WorkerStatus;
  currentTaskId: string | null;
  vramUsedMB?: number;
  tokPerSec?: number;
}

export interface ModelInfo {
  repo: string;
  tensorCount: number;
  gpuMemoryMB: number;
}

export interface ModelLoadedPayload {
  model: ModelInfo;
}

export type WorkerStatus = 'idle' | 'loading' | 'generating' | 'error';

// ── Task Types ────────────────────────────────────────────────────────────

export interface SamplingConfig {
  temperature?: number;
  topP?: number;
  topK?: number;
  maxTokens?: number;
  repetitionPenalty?: number;
}

export interface TaskSubmitPayload {
  prompt: string;
  systemPrompt?: string;
  sampling?: SamplingConfig;
  requiredModel?: string;    // Must be loaded on the assigned worker
  priority?: number;         // 0 = normal, 1 = high
  metadata?: Record<string, unknown>;  // Opaque data passed through to result
}

export interface TaskAssignPayload {
  taskId: string;
  prompt: string;
  systemPrompt?: string;
  sampling: SamplingConfig;
}

export interface TaskTokenPayload {
  taskId: string;
  token: string;
  tokenIndex: number;
  tokPerSec?: number;
}

export interface TaskCompletePayload {
  taskId: string;
  fullText: string;
  tokenCount: number;
  durationMs: number;
  tokPerSec: number;
}

export interface TaskErrorPayload {
  taskId: string;
  error: string;
}

// ── Cluster Status ────────────────────────────────────────────────────────

export interface WorkerState {
  id: string;
  gpu: GPUInfo;
  status: WorkerStatus;
  model: ModelInfo | null;
  currentTaskId: string | null;
  connectedAt: number;
  lastHeartbeat: number;
  stats: {
    tasksCompleted: number;
    totalTokens: number;
    avgTokPerSec: number;
  };
}

export type TaskStatus = 'queued' | 'assigned' | 'running' | 'complete' | 'error';

export interface TaskState {
  id: string;
  status: TaskStatus;
  prompt: string;
  assignedWorker: string | null;
  submittedAt: number;
  startedAt?: number;
  completedAt?: number;
  tokenCount?: number;
  result?: string;
  error?: string;
}

export interface ClusterStatusPayload {
  workers: WorkerState[];
  tasks: TaskState[];
  hubUptime: number;
}

// ── Helpers ───────────────────────────────────────────────────────────────

let _msgCounter = 0;

export function createMessage<T>(
  type: MessageType,
  source: string,
  payload: T,
): WSMessage<T> {
  return {
    type,
    id: `${source}-${Date.now()}-${++_msgCounter}`,
    timestamp: Date.now(),
    source,
    payload,
  };
}
