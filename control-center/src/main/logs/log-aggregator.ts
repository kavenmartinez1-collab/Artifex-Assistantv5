import { ChildProcess } from 'child_process';
import { LogStore, LogEntry } from './log-store';

/**
 * Attaches to child process stdout/stderr, buffers partial lines,
 * determines severity, and pushes completed log entries to the LogStore.
 */
export class LogAggregator {
  private logStore: LogStore;
  private lineBuffers: Map<string, { stdout: string; stderr: string }> = new Map();

  constructor(logStore: LogStore) {
    this.logStore = logStore;
  }

  /** Add a manual log entry (not from a child process). */
  addEntry(source: string, severity: 'info' | 'warn' | 'error', text: string): void {
    this.logStore.push({ timestamp: Date.now(), source, severity, text });
  }

  /**
   * Attach to a child process's stdout and stderr.
   * Each line is parsed into a LogEntry and pushed to the store.
   */
  attach(serviceId: string, child: ChildProcess): void {
    // Initialize line buffers for partial data
    this.lineBuffers.set(serviceId, { stdout: '', stderr: '' });

    if (child.stdout) {
      child.stdout.on('data', (chunk: Buffer) => {
        this.processChunk(serviceId, chunk.toString(), 'stdout');
      });
    }

    if (child.stderr) {
      child.stderr.on('data', (chunk: Buffer) => {
        this.processChunk(serviceId, chunk.toString(), 'stderr');
      });
    }

    // Flush remaining buffer on exit
    child.on('exit', () => {
      this.flushBuffer(serviceId, 'stdout');
      this.flushBuffer(serviceId, 'stderr');
      this.lineBuffers.delete(serviceId);
    });
  }

  private processChunk(
    serviceId: string,
    data: string,
    stream: 'stdout' | 'stderr'
  ): void {
    const buffers = this.lineBuffers.get(serviceId);
    if (!buffers) return;

    buffers[stream] += data;
    const lines = buffers[stream].split('\n');

    // Last element is either '' (line ended with \n) or a partial line
    buffers[stream] = lines.pop() || '';

    for (const line of lines) {
      const trimmed = line.replace(/\r$/, '');
      if (trimmed.length === 0) continue;

      const entry: LogEntry = {
        timestamp: Date.now(),
        source: serviceId,
        severity: this.determineSeverity(trimmed, stream),
        text: trimmed,
      };
      this.logStore.push(entry);
    }
  }

  private flushBuffer(serviceId: string, stream: 'stdout' | 'stderr'): void {
    const buffers = this.lineBuffers.get(serviceId);
    if (!buffers) return;

    const remaining = buffers[stream].replace(/\r$/, '');
    if (remaining.length > 0) {
      const entry: LogEntry = {
        timestamp: Date.now(),
        source: serviceId,
        severity: this.determineSeverity(remaining, stream),
        text: remaining,
      };
      this.logStore.push(entry);
    }
    buffers[stream] = '';
  }

  private determineSeverity(
    text: string,
    stream: 'stdout' | 'stderr'
  ): 'info' | 'warn' | 'error' {
    const lower = text.toLowerCase();

    // Explicit error indicators
    if (lower.includes('error') || lower.includes('fail') || lower.includes('fatal')) {
      return 'error';
    }

    // Warnings
    if (lower.includes('warn') || lower.includes('deprecat')) {
      return 'warn';
    }

    // stderr is at least a warning
    if (stream === 'stderr') {
      return 'warn';
    }

    return 'info';
  }
}
