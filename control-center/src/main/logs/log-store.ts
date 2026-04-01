import * as fs from 'fs';

export interface LogEntry {
  timestamp: number;
  source: string;
  severity: 'info' | 'warn' | 'error';
  text: string;
}

/**
 * Ring-buffer log store. Holds at most `maxSize` entries.
 * Provides callbacks for new entries, retrieval, export, and clearing.
 */
export class LogStore {
  private buffer: (LogEntry | null)[];
  private head: number = 0;
  private count: number = 0;
  private maxSize: number;

  /** Called for each new entry pushed. Set externally. */
  public onEntry: ((entry: LogEntry) => void) | null = null;

  constructor(maxSize: number = 10000) {
    this.maxSize = maxSize;
    this.buffer = new Array(maxSize).fill(null);
  }

  /**
   * Add a log entry to the ring buffer.
   */
  push(entry: LogEntry): void {
    this.buffer[this.head] = entry;
    this.head = (this.head + 1) % this.maxSize;
    if (this.count < this.maxSize) this.count++;

    if (this.onEntry) {
      this.onEntry(entry);
    }
  }

  /**
   * Return the last `n` entries in chronological order.
   */
  getRecent(n?: number): LogEntry[] {
    const requested = Math.min(n || this.count, this.count);
    const result: LogEntry[] = [];

    // Start index: head points to the next write position
    // Most recent entry is at (head - 1), oldest in buffer is at (head) if full
    let start: number;
    if (this.count < this.maxSize) {
      // Buffer not yet full; entries are 0..count-1
      start = Math.max(0, this.count - requested);
      for (let i = start; i < this.count; i++) {
        const entry = this.buffer[i];
        if (entry) result.push(entry);
      }
    } else {
      // Buffer is full; oldest is at head, newest at head-1
      start = (this.head - requested + this.maxSize) % this.maxSize;
      for (let i = 0; i < requested; i++) {
        const idx = (start + i) % this.maxSize;
        const entry = this.buffer[idx];
        if (entry) result.push(entry);
      }
    }

    return result;
  }

  /**
   * Clear all entries.
   */
  clear(): void {
    this.buffer = new Array(this.maxSize).fill(null);
    this.head = 0;
    this.count = 0;
  }

  /**
   * Export all stored entries to a file as JSON Lines.
   */
  exportToFile(filePath: string): void {
    const entries = this.getRecent(this.count);
    const lines = entries.map((e) => JSON.stringify(e)).join('\n');
    fs.writeFileSync(filePath, lines + '\n', 'utf-8');
  }
}
