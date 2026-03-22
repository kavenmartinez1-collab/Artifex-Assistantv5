/**
 * Metrics reporter — POSTs telemetry back to the dev server.
 * Used for automated development feedback loop.
 */

const METRICS_URL = '/metrics';

export async function reportMetric(event: string, data: Record<string, unknown>): Promise<void> {
  try {
    await fetch(METRICS_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ event, data }),
    });
  } catch {
    // Dev server may not be running — fail silently
    console.debug(`[metrics] Failed to report: ${event}`);
  }
}

export function reportError(context: string, error: unknown): void {
  const message = error instanceof Error ? error.message : String(error);
  const stack = error instanceof Error ? error.stack : undefined;
  reportMetric('webgpu-error', { context, message, stack });
  console.error(`[${context}]`, error);
}

/**
 * Time a function and report the duration.
 */
export async function timed<T>(
  event: string,
  label: string,
  fn: () => T | Promise<T>
): Promise<T> {
  const start = performance.now();
  const result = await fn();
  const elapsed = performance.now() - start;
  reportMetric(event, { label, elapsed_ms: Math.round(elapsed * 100) / 100 });
  return result;
}
