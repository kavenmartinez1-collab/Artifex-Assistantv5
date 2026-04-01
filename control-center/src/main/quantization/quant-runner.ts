import { spawn, ChildProcess } from 'child_process';
import { BrowserWindow } from 'electron';
import * as path from 'path';
import * as fs from 'fs';

export interface QuantConfig {
  modelPath: string;
  outputPath: string;
  keepBF16: string[];
  bits: number;
  groupSize: number;
  numSamples: number;
  seqLength: number;
  actorder: boolean;
  hadamard: boolean;
  klt: boolean;
  recipe: string | null;
  profileSSM: boolean;
}

export interface QuantProgress {
  phase: 'loading' | 'calibrating' | 'quantizing' | 'saving' | 'profiling' | 'done' | 'error';
  layer: number;
  totalLayers: number;
  sublayer: string;
  percent: number;
  eta: string;
  message: string;
}

export class QuantRunner {
  private process: ChildProcess | null = null;
  private mainWindow: BrowserWindow | null = null;
  private startTime = 0;
  private projectRoot: string;

  constructor(projectRoot: string) {
    this.projectRoot = projectRoot;
  }

  setWindow(win: BrowserWindow) {
    this.mainWindow = win;
  }

  isRunning(): boolean {
    return this.process !== null;
  }

  getVenvPython(): string {
    return path.join(this.projectRoot, 'venv', 'Scripts', 'python.exe');
  }

  getQuantScript(): string {
    return path.join(this.projectRoot, 'scripts', 'quantize_gptq.py');
  }

  /** List available recipes in the recipes/ directory */
  listRecipes(): { name: string; path: string; description: string }[] {
    const recipesDir = path.join(this.projectRoot, 'recipes');
    if (!fs.existsSync(recipesDir)) return [];

    return fs.readdirSync(recipesDir)
      .filter(f => f.endsWith('.json'))
      .map(f => {
        const fullPath = path.join(recipesDir, f);
        try {
          const data = JSON.parse(fs.readFileSync(fullPath, 'utf-8'));
          return {
            name: f.replace('.json', ''),
            path: fullPath,
            description: data.description || `${data.global?.bits || '?'}-bit, gs=${data.global?.group_size || '?'}`,
          };
        } catch {
          return { name: f, path: fullPath, description: 'Invalid recipe' };
        }
      });
  }

  /** List available models in models/ directory */
  listModels(): { name: string; path: string; sizeGB: string; isQuantized: boolean }[] {
    const modelsDir = path.join(this.projectRoot, 'models');
    if (!fs.existsSync(modelsDir)) return [];

    return fs.readdirSync(modelsDir, { withFileTypes: true })
      .filter(d => d.isDirectory())
      .map(d => {
        const modelPath = path.join(modelsDir, d.name);
        const configPath = path.join(modelPath, 'config.json');
        let isQuantized = false;
        if (fs.existsSync(configPath)) {
          try {
            const cfg = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
            isQuantized = !!cfg.quantization_config;
          } catch { /* ignore */ }
        }

        // Calculate size
        let totalBytes = 0;
        try {
          const files = fs.readdirSync(modelPath);
          for (const f of files) {
            const stat = fs.statSync(path.join(modelPath, f));
            totalBytes += stat.size;
          }
        } catch { /* ignore */ }

        return {
          name: d.name,
          path: modelPath,
          sizeGB: (totalBytes / (1024 ** 3)).toFixed(2) + ' GB',
          isQuantized,
        };
      })
      .sort((a, b) => a.name.localeCompare(b.name));
  }

  /** Run SSM profiling */
  async profile(modelPath: string, numSamples = 32): Promise<void> {
    return this.runQuantScript([
      '--model', modelPath,
      '--profile-ssm',
      '--profile-samples', String(numSamples),
      '--output', 'ssm-profile.json',
      '--device', 'cuda',
    ]);
  }

  /** Run quantization with the given config */
  async quantize(config: QuantConfig): Promise<void> {
    const args: string[] = [
      '--model', config.modelPath,
      '--output', config.outputPath,
      '--bits', String(config.bits),
      '--group-size', String(config.groupSize),
      '--num-samples', String(config.numSamples),
      '--seq-length', String(config.seqLength),
      '--device', 'cuda',
      '--report-mse',
    ];

    if (config.keepBF16.length > 0) {
      args.push('--keep-bf16', ...config.keepBF16);
    }
    if (!config.actorder) args.push('--no-actorder');
    if (config.hadamard) args.push('--hadamard');
    if (config.klt) args.push('--klt');
    if (config.recipe) args.push('--recipe', config.recipe);

    return this.runQuantScript(args);
  }

  /** Cancel running quantization */
  cancel(): void {
    if (this.process) {
      this.process.kill('SIGTERM');
      setTimeout(() => {
        if (this.process) {
          this.process.kill('SIGKILL');
        }
      }, 5000);
    }
  }

  private async runQuantScript(args: string[]): Promise<void> {
    if (this.process) {
      throw new Error('Quantization already in progress');
    }

    const pythonPath = this.getVenvPython();
    const scriptPath = this.getQuantScript();

    if (!fs.existsSync(pythonPath)) {
      throw new Error(`Python not found at ${pythonPath}`);
    }
    if (!fs.existsSync(scriptPath)) {
      throw new Error(`Quantize script not found at ${scriptPath}`);
    }

    this.startTime = Date.now();
    this.emit({ phase: 'loading', layer: 0, totalLayers: 32, sublayer: '', percent: 0, eta: '', message: 'Starting...' });

    return new Promise<void>((resolve, reject) => {
      this.process = spawn(pythonPath, [scriptPath, ...args], {
        cwd: this.projectRoot,
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
      });

      let stderr = '';

      this.process.stdout?.on('data', (data: Buffer) => {
        const text = data.toString();
        this.parseProgress(text);
        // Also forward as log entry
        this.emitLog(text);
      });

      this.process.stderr?.on('data', (data: Buffer) => {
        stderr += data.toString();
        this.emitLog(data.toString());
      });

      this.process.on('close', (code: number | null) => {
        this.process = null;
        if (code === 0) {
          this.emit({ phase: 'done', layer: 32, totalLayers: 32, sublayer: '', percent: 100, eta: '', message: 'Quantization complete!' });
          resolve();
        } else {
          const errMsg = stderr.trim().split('\n').pop() || `Exit code ${code}`;
          this.emit({ phase: 'error', layer: 0, totalLayers: 32, sublayer: '', percent: 0, eta: '', message: errMsg });
          reject(new Error(errMsg));
        }
      });

      this.process.on('error', (err: Error) => {
        this.process = null;
        this.emit({ phase: 'error', layer: 0, totalLayers: 32, sublayer: '', percent: 0, eta: '', message: err.message });
        reject(err);
      });
    });
  }

  private parseProgress(text: string) {
    // Parse layer progress: "Layer 5/31 done (53.2s, 8 quantized)"
    const layerMatch = text.match(/Layer (\d+)\/(\d+) done/);
    if (layerMatch) {
      const layer = parseInt(layerMatch[1]) + 1;
      const total = parseInt(layerMatch[2]) + 1;
      const percent = Math.round((layer / total) * 90); // 90% for layers, 10% for saving
      const elapsed = (Date.now() - this.startTime) / 1000;
      const rate = elapsed / layer;
      const remaining = rate * (total - layer);
      const eta = remaining > 60 ? `${Math.round(remaining / 60)}m` : `${Math.round(remaining)}s`;
      this.emit({ phase: 'quantizing', layer, totalLayers: total, sublayer: '', percent, eta, message: `Layer ${layer}/${total}` });
    }

    // Parse sublayer: "[5] linear_attn.in_proj_qkv [8192x4096] INT4 MSE=..."
    const subMatch = text.match(/\[(\d+)\]\s+(\S+)\s+\[.*?\]\s+(INT[48]|INT8|E8|BF16|RTN)/);
    if (subMatch) {
      this.emit({ phase: 'quantizing', layer: -1, totalLayers: -1, sublayer: `${subMatch[2]} (${subMatch[3]})`, percent: -1, eta: '', message: text.trim() });
    }

    // Parse loading
    if (text.includes('Loading model')) {
      this.emit({ phase: 'loading', layer: 0, totalLayers: 32, sublayer: '', percent: 2, eta: '', message: 'Loading model...' });
    }

    // Parse calibration
    if (text.includes('Computing initial hidden states')) {
      this.emit({ phase: 'calibrating', layer: 0, totalLayers: 32, sublayer: '', percent: 5, eta: '', message: 'Computing calibration data...' });
    }

    // Parse saving
    if (text.includes('Writing model.safetensors')) {
      this.emit({ phase: 'saving', layer: 32, totalLayers: 32, sublayer: '', percent: 95, eta: '', message: 'Saving model...' });
    }

    // Parse profile
    if (text.includes('SSM Activation Profiling')) {
      this.emit({ phase: 'profiling', layer: 0, totalLayers: 32, sublayer: '', percent: 10, eta: '', message: 'Profiling SSM activations...' });
    }
  }

  private emit(progress: QuantProgress) {
    if (this.mainWindow && !this.mainWindow.isDestroyed()) {
      this.mainWindow.webContents.send('quant-progress', progress);
    }
  }

  private emitLog(text: string) {
    if (this.mainWindow && !this.mainWindow.isDestroyed()) {
      this.mainWindow.webContents.send('log-entry', {
        timestamp: Date.now(),
        source: 'quantizer',
        severity: text.toLowerCase().includes('error') ? 'error'
          : text.toLowerCase().includes('warn') ? 'warn' : 'info',
        text: text.trim(),
      });
    }
  }
}
