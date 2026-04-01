/**
 * Maintains time-series data for sparkline graphs in the cluster panel.
 * Each worker can have multiple named metrics (e.g. "tokPerSec", "vramUsed").
 * Points are stored as fixed-length ring buffers for efficient memory use.
 */
export class ClusterTimeSeries {
  private maxPoints: number;
  // Map<workerId, Map<metric, number[]>>
  private series: Map<string, Map<string, number[]>> = new Map();

  constructor(maxPoints: number = 60) {
    this.maxPoints = maxPoints;
  }

  /**
   * Add a data point for a given worker and metric.
   * If the series exceeds maxPoints, the oldest point is dropped.
   */
  addPoint(workerId: string, metric: string, value: number): void {
    let workerMetrics = this.series.get(workerId);
    if (!workerMetrics) {
      workerMetrics = new Map();
      this.series.set(workerId, workerMetrics);
    }

    let points = workerMetrics.get(metric);
    if (!points) {
      points = [];
      workerMetrics.set(metric, points);
    }

    points.push(value);
    if (points.length > this.maxPoints) {
      points.shift();
    }
  }

  /**
   * Get the time-series array for a given worker and metric.
   * Returns a copy of the array. Returns empty array if no data exists.
   */
  getSeries(workerId: string, metric: string): number[] {
    const workerMetrics = this.series.get(workerId);
    if (!workerMetrics) return [];

    const points = workerMetrics.get(metric);
    if (!points) return [];

    return points.slice();
  }

  /**
   * Get all worker IDs that have at least one metric recorded.
   */
  getWorkerIds(): string[] {
    return Array.from(this.series.keys());
  }

  /**
   * Remove all series data for a specific worker.
   */
  removeWorker(workerId: string): void {
    this.series.delete(workerId);
  }

  /**
   * Clear all stored time-series data.
   */
  clear(): void {
    this.series.clear();
  }

  /**
   * Get a serializable snapshot of all series data for IPC transfer.
   * Returns { workerId: { metric: number[] } }
   */
  getSnapshot(): Record<string, Record<string, number[]>> {
    const result: Record<string, Record<string, number[]>> = {};
    for (const [workerId, metrics] of this.series) {
      result[workerId] = {};
      for (const [metric, points] of metrics) {
        result[workerId][metric] = points.slice();
      }
    }
    return result;
  }
}
