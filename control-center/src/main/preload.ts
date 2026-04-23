import { contextBridge, ipcRenderer, IpcRendererEvent } from 'electron';

export interface ServiceStatus {
  id: string;
  name: string;
  status: 'stopped' | 'starting' | 'running' | 'stopping' | 'error';
  pid: number | null;
  port: number;
  startTime: number | null;
  errorMessage: string | null;
}

export interface LogEntry {
  timestamp: number;
  source: string;
  severity: 'info' | 'warn' | 'error';
  text: string;
}

type ServiceUpdateCallback = (status: ServiceStatus) => void;
type LogEntryCallback = (entry: LogEntry) => void;
type ClusterStatusCallback = (status: unknown) => void;
type QuantProgressCallback = (progress: unknown) => void;

// Track listeners so we can remove them properly
const serviceListeners = new Map<ServiceUpdateCallback, (_e: IpcRendererEvent, s: ServiceStatus) => void>();
const logListeners = new Map<LogEntryCallback, (_e: IpcRendererEvent, e: LogEntry) => void>();
const clusterListeners = new Map<ClusterStatusCallback, (_e: IpcRendererEvent, s: unknown) => void>();
const quantListeners = new Map<QuantProgressCallback, (_e: IpcRendererEvent, p: unknown) => void>();

contextBridge.exposeInMainWorld('artifex', {
  // ── Service Management ──
  listServices: (): Promise<ServiceStatus[]> => {
    return ipcRenderer.invoke('services:list');
  },

  startService: (id: string, configOverrides?: Record<string, string>): Promise<void> => {
    return ipcRenderer.invoke('services:start', id, configOverrides);
  },

  getServiceOptions: (): Promise<Array<{ id: string; options: unknown[] }>> => {
    return ipcRenderer.invoke('services:get-options');
  },

  stopService: (id: string): Promise<void> => {
    return ipcRenderer.invoke('services:stop', id);
  },

  restartService: (id: string): Promise<void> => {
    return ipcRenderer.invoke('services:restart', id);
  },

  startAll: (): Promise<void> => {
    return ipcRenderer.invoke('services:start-all');
  },

  stopAll: (): Promise<void> => {
    return ipcRenderer.invoke('services:stop-all');
  },

  // ── Log Management ──
  getLogBuffer: (count?: number): Promise<LogEntry[]> => {
    return ipcRenderer.invoke('logs:get-buffer', count);
  },

  exportLogs: (filePath: string): Promise<void> => {
    return ipcRenderer.invoke('logs:export', filePath);
  },

  clearLogs: (): Promise<void> => {
    return ipcRenderer.invoke('logs:clear');
  },

  // ── Event Listeners ──
  onServiceUpdate: (cb: ServiceUpdateCallback): void => {
    const wrapped = (_e: IpcRendererEvent, status: ServiceStatus) => cb(status);
    serviceListeners.set(cb, wrapped);
    ipcRenderer.on('service-update', wrapped);
  },

  onLogEntry: (cb: LogEntryCallback): void => {
    const wrapped = (_e: IpcRendererEvent, entry: LogEntry) => cb(entry);
    logListeners.set(cb, wrapped);
    ipcRenderer.on('log-entry', wrapped);
  },

  onClusterStatus: (cb: ClusterStatusCallback): void => {
    const wrapped = (_e: IpcRendererEvent, status: unknown) => cb(status);
    clusterListeners.set(cb, wrapped);
    ipcRenderer.on('cluster-status', wrapped);
  },

  onQuantProgress: (cb: QuantProgressCallback): void => {
    const wrapped = (_e: IpcRendererEvent, progress: unknown) => cb(progress);
    quantListeners.set(cb, wrapped);
    ipcRenderer.on('quant-progress', wrapped);
  },

  // ── Remove Listeners ──
  removeServiceListener: (cb: ServiceUpdateCallback): void => {
    const wrapped = serviceListeners.get(cb);
    if (wrapped) {
      ipcRenderer.removeListener('service-update', wrapped);
      serviceListeners.delete(cb);
    }
  },

  removeLogListener: (cb: LogEntryCallback): void => {
    const wrapped = logListeners.get(cb);
    if (wrapped) {
      ipcRenderer.removeListener('log-entry', wrapped);
      logListeners.delete(cb);
    }
  },

  removeClusterListener: (cb: ClusterStatusCallback): void => {
    const wrapped = clusterListeners.get(cb);
    if (wrapped) {
      ipcRenderer.removeListener('cluster-status', wrapped);
      clusterListeners.delete(cb);
    }
  },

  removeQuantListener: (cb: QuantProgressCallback): void => {
    const wrapped = quantListeners.get(cb);
    if (wrapped) {
      ipcRenderer.removeListener('quant-progress', wrapped);
      quantListeners.delete(cb);
    }
  },

  // ── Model Management ──
  scanModels: (): Promise<unknown[]> => {
    return ipcRenderer.invoke('models:scan');
  },

  scanOllamaModels: (): Promise<unknown[]> => {
    return ipcRenderer.invoke('models:scan-ollama');
  },

  listModelNamesByBackend: (): Promise<{ transformers: string[]; ollama: string[]; llama_cpp: string[] }> => {
    return ipcRenderer.invoke('models:list-names');
  },

  deleteModel: (modelPath: string): Promise<void> => {
    return ipcRenderer.invoke('models:delete', modelPath);
  },

  deleteOllamaModel: (name: string): Promise<void> => {
    return ipcRenderer.invoke('models:delete-ollama', name);
  },

  // ── Docker Management ──
  docker: {
    isDockerInstalled: (): Promise<boolean> => {
      return ipcRenderer.invoke('docker:is-installed');
    },

    listContainers: (): Promise<unknown[]> => {
      return ipcRenderer.invoke('docker:list-containers');
    },

    startContainer: (nameOrId: string): Promise<void> => {
      return ipcRenderer.invoke('docker:start-container', nameOrId);
    },

    stopContainer: (nameOrId: string): Promise<void> => {
      return ipcRenderer.invoke('docker:stop-container', nameOrId);
    },

    composeUp: (composePath: string): Promise<void> => {
      return ipcRenderer.invoke('docker:compose-up', composePath);
    },

    composeDown: (composePath: string): Promise<void> => {
      return ipcRenderer.invoke('docker:compose-down', composePath);
    },

    parseComposeFile: (composePath: string): Promise<unknown[]> => {
      return ipcRenderer.invoke('docker:parse-compose', composePath);
    },

    getContainerLogs: (nameOrId: string, tail?: number): Promise<string> => {
      return ipcRenderer.invoke('docker:container-logs', nameOrId, tail);
    },
  },

  // ── Quantization ──
  listQuantModels: (): Promise<unknown[]> => {
    return ipcRenderer.invoke('quant:list-models');
  },

  listQuantRecipes: (): Promise<unknown[]> => {
    return ipcRenderer.invoke('quant:list-recipes');
  },

  runQuantization: (config: unknown): Promise<void> => {
    return ipcRenderer.invoke('quant:run', config);
  },

  cancelQuantization: (): Promise<void> => {
    return ipcRenderer.invoke('quant:cancel');
  },

  isQuantRunning: (): Promise<boolean> => {
    return ipcRenderer.invoke('quant:is-running');
  },

  profileModel: (modelPath: string, numSamples?: number): Promise<unknown> => {
    return ipcRenderer.invoke('quant:profile', modelPath, numSamples);
  },

  // ── Config / State ──
  getConfig: (): Promise<unknown> => {
    return ipcRenderer.invoke('config:get');
  },

  updateConfig: (partial: Record<string, unknown>): Promise<void> => {
    return ipcRenderer.invoke('config:update', partial);
  },
});
