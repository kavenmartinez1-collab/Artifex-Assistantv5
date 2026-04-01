import { app, BrowserWindow, Tray, Menu, nativeImage } from 'electron';
import * as path from 'path';
import { registerAllHandlers } from './ipc-handlers';
import { ServiceManager } from './services/service-manager';
import { LogAggregator } from './logs/log-aggregator';
import { LogStore } from './logs/log-store';
import { loadConfig, saveConfig, updateConfig } from './state/persistence';

let mainWindow: BrowserWindow | null = null;
let tray: Tray | null = null;
let serviceManager: ServiceManager;
let logAggregator: LogAggregator;
let logStore: LogStore;

function createWindow(): void {
  const config = loadConfig();
  const bounds = config.windowBounds || { width: 1200, height: 800 };

  mainWindow = new BrowserWindow({
    width: bounds.width,
    height: bounds.height,
    x: bounds.x,
    y: bounds.y,
    minWidth: 900,
    minHeight: 600,
    backgroundColor: '#0f111a',
    titleBarStyle: 'default',
    darkTheme: true,
    title: 'Artifex Control Center',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  // Renderer files live in src/renderer/ (not compiled by tsc)
  const rendererPath = path.join(__dirname, '..', '..', 'src', 'renderer', 'index.html');
  mainWindow.loadFile(rendererPath);

  // Save window bounds on resize/move
  const saveBounds = () => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      const [x, y] = mainWindow.getPosition();
      const [width, height] = mainWindow.getSize();
      updateConfig({ windowBounds: { x, y, width, height } });
    }
  };
  mainWindow.on('resize', saveBounds);
  mainWindow.on('move', saveBounds);

  mainWindow.on('close', async (e) => {
    e.preventDefault();
    // Save which services were running
    const statuses = serviceManager.getStatus();
    const activeIds = statuses
      .filter((s) => s.status === 'running')
      .map((s) => s.id);
    updateConfig({ activeServicesOnClose: activeIds });

    // Stop all services gracefully
    await serviceManager.stopAll();
    saveConfig(loadConfig());

    mainWindow?.destroy();
    mainWindow = null;
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

function createTray(): void {
  // Create a simple 16x16 tray icon (cyan dot on transparent)
  const icon = nativeImage.createFromBuffer(
    Buffer.alloc(16 * 16 * 4, 0),
    { width: 16, height: 16 }
  );
  tray = new Tray(icon);
  tray.setToolTip('Artifex Control Center');

  const contextMenu = Menu.buildFromTemplate([
    {
      label: 'Show Window',
      click: () => {
        if (mainWindow) {
          mainWindow.show();
          mainWindow.focus();
        }
      },
    },
    {
      label: 'Start All Services',
      click: () => serviceManager.startAll(),
    },
    {
      label: 'Stop All Services',
      click: () => serviceManager.stopAll(),
    },
    { type: 'separator' },
    {
      label: 'Quit',
      click: () => {
        mainWindow?.close();
        app.quit();
      },
    },
  ]);
  tray.setContextMenu(contextMenu);

  tray.on('click', () => {
    if (mainWindow) {
      if (mainWindow.isVisible()) {
        mainWindow.hide();
      } else {
        mainWindow.show();
        mainWindow.focus();
      }
    }
  });
}

function initServices(): void {
  const config = loadConfig();
  logStore = new LogStore(config.logBufferSize || 10000);
  logAggregator = new LogAggregator(logStore);

  serviceManager = new ServiceManager(logAggregator, (status) => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('service-update', status);
    }
  });

  // Push log entries to renderer
  logStore.onEntry = (entry) => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('log-entry', entry);
    }
  };
}

app.whenReady().then(() => {
  initServices();
  createWindow();
  createTray();

  if (mainWindow) {
    registerAllHandlers(mainWindow, serviceManager, logStore);
  }
});

app.on('window-all-closed', () => {
  app.quit();
});

app.on('before-quit', async () => {
  await serviceManager?.stopAll();
});
