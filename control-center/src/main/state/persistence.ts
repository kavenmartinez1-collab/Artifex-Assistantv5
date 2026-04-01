import * as fs from 'fs';
import * as path from 'path';
import { app } from 'electron';

export interface WindowBounds {
  x?: number;
  y?: number;
  width: number;
  height: number;
}

export interface AppConfig {
  windowBounds?: WindowBounds;
  activeServicesOnClose: string[];
  autoRestoreServices: boolean;
  logBufferSize: number;
  activePanel: string;
  clusterHubUrl?: string;
  [key: string]: unknown;
}

const DEFAULT_CONFIG: AppConfig = {
  windowBounds: { width: 1200, height: 800 },
  activeServicesOnClose: [],
  autoRestoreServices: false,
  logBufferSize: 10000,
  activePanel: 'services',
};

let cachedConfig: AppConfig | null = null;
let saveTimer: ReturnType<typeof setTimeout> | null = null;

function getConfigPath(): string {
  return path.join(app.getPath('userData'), 'artifex-config.json');
}

/**
 * Load config from disk. Returns cached version if already loaded.
 */
export function loadConfig(): AppConfig {
  if (cachedConfig) return cachedConfig;

  const configPath = getConfigPath();
  try {
    if (fs.existsSync(configPath)) {
      const raw = fs.readFileSync(configPath, 'utf-8');
      const parsed = JSON.parse(raw);
      cachedConfig = { ...DEFAULT_CONFIG, ...parsed };
    } else {
      cachedConfig = { ...DEFAULT_CONFIG };
    }
  } catch {
    cachedConfig = { ...DEFAULT_CONFIG };
  }
  return cachedConfig!;
}

/**
 * Save the full config to disk immediately.
 */
export function saveConfig(config: AppConfig): void {
  cachedConfig = config;
  const configPath = getConfigPath();
  try {
    const dir = path.dirname(configPath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    fs.writeFileSync(configPath, JSON.stringify(config, null, 2), 'utf-8');
  } catch {
    // Config save failure is non-fatal
  }
}

/**
 * Merge a partial config update. Debounces writes to at most once per second.
 */
export function updateConfig(partial: Partial<AppConfig>): void {
  const current = loadConfig();
  cachedConfig = { ...current, ...partial };

  if (saveTimer) clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    if (cachedConfig) saveConfig(cachedConfig);
    saveTimer = null;
  }, 1000);
}
