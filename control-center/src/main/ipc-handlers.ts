import { ipcMain, BrowserWindow, dialog } from 'electron';
import * as path from 'path';
import { ServiceManager } from './services/service-manager';
import { LogStore } from './logs/log-store';
import { loadConfig, updateConfig } from './state/persistence';
import { scanModels, deleteModel } from './models/model-scanner';
import {
  isDockerInstalled,
  listContainers,
  startContainer,
  stopContainer,
  composeUp,
  composeDown,
  parseComposeFile,
  getContainerLogs,
} from './docker/docker-cli';
import { QuantRunner, QuantConfig } from './quantization/quant-runner';

export function registerAllHandlers(
  _mainWindow: BrowserWindow,
  serviceManager: ServiceManager,
  logStore: LogStore
): void {
  // Use the same project root detection as service-config.ts
  // __dirname is dist/main/ → need to walk up to find the actual project root
  // (the directory containing webgpu/ and scripts/)
  let projectRoot = path.resolve(__dirname, '..', '..');
  // Walk up to find the real project root (above control-center/)
  let dir = __dirname;
  for (let i = 0; i < 10; i++) {
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
    const fs = require('fs');
    if (fs.existsSync(path.join(dir, 'webgpu')) && fs.existsSync(path.join(dir, 'scripts'))) {
      projectRoot = dir;
      break;
    }
  }
  const quantRunner = new QuantRunner(projectRoot);
  quantRunner.setWindow(_mainWindow);
  // ── Service Handlers ──

  ipcMain.handle('services:list', async () => {
    return serviceManager.getStatus();
  });

  ipcMain.handle('services:start', async (_event, id: string, configOverrides?: Record<string, string>) => {
    await serviceManager.start(id, configOverrides);
  });

  ipcMain.handle('services:stop', async (_event, id: string) => {
    await serviceManager.stop(id);
  });

  ipcMain.handle('services:restart', async (_event, id: string) => {
    await serviceManager.restart(id);
  });

  ipcMain.handle('services:get-options', async () => {
    // Return service option definitions for the UI
    const { SERVICE_DEFINITIONS } = require('./services/service-config');
    return SERVICE_DEFINITIONS.map((def: any) => ({
      id: def.id,
      options: def.options || [],
    }));
  });

  ipcMain.handle('services:start-all', async () => {
    await serviceManager.startAll();
  });

  ipcMain.handle('services:stop-all', async () => {
    await serviceManager.stopAll();
  });

  // ── Log Handlers ──

  ipcMain.handle('logs:get-buffer', async (_event, count?: number) => {
    return logStore.getRecent(count || 1000);
  });

  ipcMain.handle('logs:export', async (_event, filePath?: string) => {
    let targetPath = filePath;
    if (!targetPath) {
      const result = await dialog.showSaveDialog({
        title: 'Export Logs',
        defaultPath: `artifex-logs-${Date.now()}.jsonl`,
        filters: [
          { name: 'JSON Lines', extensions: ['jsonl'] },
          { name: 'All Files', extensions: ['*'] },
        ],
      });
      if (result.canceled || !result.filePath) return;
      targetPath = result.filePath;
    }
    logStore.exportToFile(targetPath);
  });

  ipcMain.handle('logs:clear', async () => {
    logStore.clear();
  });

  // ── Model Handlers ──

  ipcMain.handle('models:scan', async () => {
    return scanModels(projectRoot);
  });

  ipcMain.handle('models:delete', async (_event, modelPath: string) => {
    return deleteModel(modelPath);
  });

  // ── Docker Handlers ──

  ipcMain.handle('docker:is-installed', async () => {
    return isDockerInstalled();
  });

  ipcMain.handle('docker:list-containers', async () => {
    return listContainers();
  });

  ipcMain.handle('docker:start-container', async (_event, nameOrId: string) => {
    await startContainer(nameOrId);
  });

  ipcMain.handle('docker:stop-container', async (_event, nameOrId: string) => {
    await stopContainer(nameOrId);
  });

  ipcMain.handle('docker:compose-up', async (_event, composePath: string) => {
    await composeUp(composePath);
  });

  ipcMain.handle('docker:compose-down', async (_event, composePath: string) => {
    await composeDown(composePath);
  });

  ipcMain.handle('docker:parse-compose', async (_event, composePath: string) => {
    return parseComposeFile(composePath);
  });

  ipcMain.handle('docker:container-logs', async (_event, nameOrId: string, tail?: number) => {
    return getContainerLogs(nameOrId, tail);
  });

  // ── Quantization Handlers ──

  ipcMain.handle('quant:list-models', async () => {
    return quantRunner.listModels();
  });

  ipcMain.handle('quant:list-recipes', async () => {
    return quantRunner.listRecipes();
  });

  ipcMain.handle('quant:run', async (_event, config: QuantConfig) => {
    return quantRunner.quantize(config);
  });

  ipcMain.handle('quant:profile', async (_event, modelPath: string, numSamples?: number) => {
    return quantRunner.profile(modelPath, numSamples);
  });

  ipcMain.handle('quant:cancel', async () => {
    quantRunner.cancel();
  });

  ipcMain.handle('quant:is-running', async () => {
    return quantRunner.isRunning();
  });

  // ── Config Handlers ──

  ipcMain.handle('config:get', async () => {
    return loadConfig();
  });

  ipcMain.handle('config:update', async (_event, partial: Record<string, unknown>) => {
    updateConfig(partial);
  });
}
