/**
 * Artifex bridge client — lets the Python framework (Qt GUI / CLI / agent
 * loop) drive THIS page's WebGPU inference session.
 *
 * The page long-polls the Python-side bridge server (core/webgpu_bridge.py,
 * 127.0.0.1:8790 by default) for chat jobs, runs them through the live
 * InferenceSession, and streams tokens back with ordered POSTs. When no
 * bridge server is running the loop backs off quietly — zero cost, zero
 * config; selecting the "webgpu" backend in the Qt GUI is all it takes.
 *
 * Event POST responses may carry {cancel: true}; the running generation is
 * then aborted via its GenerationHandle.
 */

import type { SamplingConfig, GenerationHandle, OnTokenCallback } from './engine/generate';

const DEFAULT_PORT = 8790;

interface BridgeJob {
  id: string;
  kind: string;
  messages: Array<{ role: string; content: string }>;
  sampling: SamplingConfig & { [key: string]: unknown };
  enableThinking?: boolean;
}

interface BridgeSessionLike {
  chat: (
    messages: Array<{ role: string; content: string }>,
    sampling?: SamplingConfig,
    onToken?: OnTokenCallback,
    opts?: { enableThinking?: boolean },
  ) => GenerationHandle;
}

export interface BridgeInfo {
  ready: boolean;
  model?: string;
  ctx?: number;
  arch?: string;
}

function bridgeBase(): string {
  const port = Number(localStorage.getItem('artifexBridgePort')) || DEFAULT_PORT;
  return `http://127.0.0.1:${port}`;
}

const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));

async function post(path: string, body: unknown): Promise<Record<string, unknown>> {
  const resp = await fetch(bridgeBase() + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  try {
    return (await resp.json()) as Record<string, unknown>;
  } catch {
    return {};
  }
}

/**
 * Start the bridge polling loop. Call once at app init; it watches the
 * session getter each cycle, so model loads/unloads need no re-wiring.
 */
export function startBridgeClient(
  getSession: () => BridgeSessionLike | null,
  getInfo: () => BridgeInfo,
  onStatus?: (msg: string) => void,
): void {
  let announcedModel: string | null = null;
  let attached = false;

  (async function loop(): Promise<never> {
    for (;;) {
      const session = getSession();
      if (!session) {
        announcedModel = null;
        await sleep(3000);
        continue;
      }
      try {
        const info = getInfo();
        if (announcedModel !== (info.model ?? null)) {
          await post('/bridge/hello', info);
          announcedModel = info.model ?? null;
          if (!attached) {
            attached = true;
            onStatus?.(`Bridge attached — Python framework can now use this session (port ${bridgeBase().split(':').pop()})`);
          }
        }
        const resp = await fetch(bridgeBase() + '/bridge/job?wait=25');
        if (resp.status === 200) {
          const job = (await resp.json()) as BridgeJob;
          if (job && job.kind === 'chat') {
            await runJob(getSession, job);
          }
        }
        // 204 = no job this poll; loop straight back into the long poll.
      } catch {
        // Bridge server not running (or died) — back off quietly.
        if (attached) onStatus?.('Bridge detached (Python side not reachable)');
        attached = false;
        announcedModel = null;
        await sleep(10_000);
      }
    }
  })();
}

async function runJob(
  getSession: () => BridgeSessionLike | null,
  job: BridgeJob,
): Promise<void> {
  const session = getSession();
  if (!session) {
    await post('/bridge/event', { id: job.id, type: 'error', error: 'no session loaded' });
    return;
  }

  // Ordered event delivery: every POST rides one promise chain so token
  // batches can never arrive out of order. Responses feed the cancel flag.
  let chain: Promise<void> = Promise.resolve();
  let cancelled = false;
  const send = (event: Record<string, unknown>): Promise<void> => {
    chain = chain.then(async () => {
      const reply = await post('/bridge/event', { id: job.id, ...event });
      if (reply && reply.cancel === true) cancelled = true;
    }).catch(() => { cancelled = true; });
    return chain;
  };

  let buf = '';
  const flush = () => {
    if (!buf) return;
    const text = buf;
    buf = '';
    void send({ type: 'token', text });
  };

  try {
    const handle = session.chat(
      job.messages,
      job.sampling,
      (piece: string) => { buf += piece; },
      { enableThinking: job.enableThinking !== false },
    );

    const flusher = setInterval(() => {
      flush();
      if (cancelled) handle.abort();
    }, 80);

    let result;
    try {
      result = await handle.result;
    } finally {
      clearInterval(flusher);
    }
    flush();
    await send({
      type: 'done',
      stats: {
        numTokens: result.numTokens,
        tokensPerSecond: result.tokensPerSecond,
        promptTokens: result.promptTokens ?? null,
        stopReason: result.stopReason,
      },
    });
    await chain;
  } catch (err) {
    flush();
    await send({ type: 'error', error: String(err) });
    await chain;
  }
}
