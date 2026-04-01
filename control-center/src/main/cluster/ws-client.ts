import WebSocket from 'ws';

export interface WorkerInfo {
  id: string;
  gpu: string;          // "RTX 5060 Ti"
  vram: number;         // total VRAM MB
  vramUsed: number;     // used VRAM MB
  status: 'idle' | 'busy' | 'offline';
  currentTask: string | null;
  tokPerSec: number;
  lastSeen: number;     // timestamp
}

export interface TaskInfo {
  id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  worker: string | null;
  model: string;
  tokens: number;
  speed: number;        // tok/s
  created: number;
  completed: number | null;
}

interface HubMessage {
  type: string;
  workers?: WorkerInfo[];
  tasks?: TaskInfo[];
  worker?: WorkerInfo;
  task?: TaskInfo;
  error?: string;
}

type UpdateCallback = (workers: WorkerInfo[], tasks: TaskInfo[]) => void;

export class ClusterClient {
  private hubUrl: string;
  private ws: WebSocket | null = null;
  private connected: boolean = false;
  private shouldReconnect: boolean = false;
  private reconnectDelay: number = 1000;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private workers: Map<string, WorkerInfo> = new Map();
  private tasks: Map<string, TaskInfo> = new Map();
  private listeners: UpdateCallback[] = [];

  private static readonly MIN_RECONNECT_DELAY = 1000;
  private static readonly MAX_RECONNECT_DELAY = 30000;
  private static readonly RECONNECT_MULTIPLIER = 2;

  constructor(hubUrl: string) {
    this.hubUrl = hubUrl;
  }

  connect(): void {
    this.shouldReconnect = true;
    this.reconnectDelay = ClusterClient.MIN_RECONNECT_DELAY;
    this.doConnect();
  }

  disconnect(): void {
    this.shouldReconnect = false;

    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.ws) {
      this.ws.removeAllListeners();
      if (this.ws.readyState === WebSocket.OPEN ||
          this.ws.readyState === WebSocket.CONNECTING) {
        this.ws.close(1000, 'Client disconnect');
      }
      this.ws = null;
    }

    if (this.connected) {
      this.connected = false;
      this.emitUpdate();
    }
  }

  isConnected(): boolean {
    return this.connected;
  }

  getWorkers(): WorkerInfo[] {
    return Array.from(this.workers.values());
  }

  getTasks(): TaskInfo[] {
    return Array.from(this.tasks.values());
  }

  onUpdate(cb: UpdateCallback): void {
    this.listeners.push(cb);
  }

  removeListener(cb: UpdateCallback): void {
    const idx = this.listeners.indexOf(cb);
    if (idx !== -1) {
      this.listeners.splice(idx, 1);
    }
  }

  submitTask(model: string, prompt: string): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }
    this.ws.send(JSON.stringify({
      type: 'submit-task',
      model,
      prompt,
    }));
  }

  // --- Private ---

  private doConnect(): void {
    if (this.ws) {
      this.ws.removeAllListeners();
      if (this.ws.readyState === WebSocket.OPEN ||
          this.ws.readyState === WebSocket.CONNECTING) {
        this.ws.close();
      }
      this.ws = null;
    }

    try {
      this.ws = new WebSocket(this.hubUrl);
    } catch {
      this.scheduleReconnect();
      return;
    }

    this.ws.on('open', () => {
      this.connected = true;
      this.reconnectDelay = ClusterClient.MIN_RECONNECT_DELAY;
      this.emitUpdate();
    });

    this.ws.on('message', (data: WebSocket.Data) => {
      this.handleMessage(data);
    });

    this.ws.on('close', () => {
      const wasConnected = this.connected;
      this.connected = false;
      if (wasConnected) {
        this.emitUpdate();
      }
      this.scheduleReconnect();
    });

    this.ws.on('error', () => {
      // 'close' event always follows 'error', so reconnect is handled there.
      // Just mark disconnected if currently connected.
      if (this.connected) {
        this.connected = false;
        this.emitUpdate();
      }
    });
  }

  private scheduleReconnect(): void {
    if (!this.shouldReconnect) return;
    if (this.reconnectTimer !== null) return;

    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      if (this.shouldReconnect) {
        this.doConnect();
      }
    }, this.reconnectDelay);

    // Exponential backoff
    this.reconnectDelay = Math.min(
      this.reconnectDelay * ClusterClient.RECONNECT_MULTIPLIER,
      ClusterClient.MAX_RECONNECT_DELAY
    );
  }

  private handleMessage(data: WebSocket.Data): void {
    let msg: HubMessage;
    try {
      msg = JSON.parse(data.toString());
    } catch {
      return; // Ignore malformed messages
    }

    switch (msg.type) {
      case 'cluster-state':
        // Full state snapshot from hub
        this.workers.clear();
        this.tasks.clear();
        if (msg.workers) {
          for (const w of msg.workers) {
            this.workers.set(w.id, w);
          }
        }
        if (msg.tasks) {
          for (const t of msg.tasks) {
            this.tasks.set(t.id, t);
          }
        }
        this.emitUpdate();
        break;

      case 'worker-update':
        // Single worker changed
        if (msg.worker) {
          if (msg.worker.status === 'offline') {
            this.workers.delete(msg.worker.id);
          } else {
            this.workers.set(msg.worker.id, msg.worker);
          }
          this.emitUpdate();
        }
        break;

      case 'task-update':
        // Single task changed
        if (msg.task) {
          this.tasks.set(msg.task.id, msg.task);
          this.emitUpdate();
        }
        break;

      case 'workers':
        // Batch workers update
        if (msg.workers) {
          this.workers.clear();
          for (const w of msg.workers) {
            this.workers.set(w.id, w);
          }
          this.emitUpdate();
        }
        break;

      case 'tasks':
        // Batch tasks update
        if (msg.tasks) {
          this.tasks.clear();
          for (const t of msg.tasks) {
            this.tasks.set(t.id, t);
          }
          this.emitUpdate();
        }
        break;

      default:
        // Unknown message type, ignore
        break;
    }
  }

  private emitUpdate(): void {
    const workers = this.getWorkers();
    const tasks = this.getTasks();
    for (const cb of this.listeners) {
      try {
        cb(workers, tasks);
      } catch {
        // Listener threw — don't break the loop
      }
    }
  }
}
