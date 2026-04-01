import { ipcMain, BrowserWindow, dialog } from 'electron';
import { ServiceManager } from './services/service-manager';
import { LogStore } from './logs/log-store';
import { loadConfig, updateConfig } from './state/persistence';

export function registerAllHandlers(
  _mainWindow: BrowserWindow,
  serviceManager: ServiceManager,
  logStore: LogStore
): void {
  // ── Service Handlers ──

  ipcMain.handle('services:list', async () => {
    return serviceManager.getStatus();
  });

  ipcMain.handle('services:start', async (_event, id: string) => {
    await serviceManager.start(id);
  });

  ipcMain.handle('services:stop', async (_event, id: string) => {
    await serviceManager.stop(id);
  });

  ipcMain.handle('services:restart', async (_event, id: string) => {
    await serviceManager.restart(id);
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

  // ── Config Handlers ──

  ipcMain.handle('config:get', async () => {
    return loadConfig();
  });

  ipcMain.handle('config:update', async (_event, partial: Record<string, unknown>) => {
    updateConfig(partial);
  });
}
