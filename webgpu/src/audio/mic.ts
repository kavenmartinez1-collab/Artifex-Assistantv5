/**
 * Microphone capture (browser) — Phase W5.
 *
 * Self-contained mic → 16 kHz mono f32, no cloud/OS speech API. getUserMedia +
 * MediaRecorder capture the mic; the blob is decoded locally (decodeAudioData)
 * and resampled to 16 kHz mono via an OfflineAudioContext — exactly the format
 * Whisper's frontend expects. Nothing leaves the machine.
 */

export interface Recorder {
  /** Stop recording and return the captured audio as 16 kHz mono f32 [-1,1]. */
  stop(): Promise<Float32Array>;
  /** Abort without producing audio (releases the mic). */
  cancel(): void;
}

/** Downmix to mono and resample to 16 kHz using an offline render graph. */
async function resample16kMono(buf: AudioBuffer): Promise<Float32Array> {
  const SR = 16000;
  // mono source buffer at the original rate (averaged channels)
  const off = new OfflineAudioContext(1, Math.ceil((buf.length / buf.sampleRate) * SR), SR);
  const mono = off.createBuffer(1, buf.length, buf.sampleRate);
  const dst = mono.getChannelData(0);
  for (let ch = 0; ch < buf.numberOfChannels; ch++) {
    const d = buf.getChannelData(ch);
    for (let i = 0; i < d.length; i++) dst[i] += d[i] / buf.numberOfChannels;
  }
  const src = off.createBufferSource();
  src.buffer = mono;
  src.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();
  return rendered.getChannelData(0).slice();
}

export async function startMic(): Promise<Recorder> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const rec = new MediaRecorder(stream);
  const chunks: Blob[] = [];
  rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  rec.start();

  const release = () => stream.getTracks().forEach((t) => t.stop());

  return {
    async stop(): Promise<Float32Array> {
      const stopped = new Promise<void>((res) => { rec.onstop = () => res(); });
      rec.stop();
      await stopped;
      release();
      const blob = new Blob(chunks, { type: rec.mimeType || 'audio/webm' });
      const ab = await blob.arrayBuffer();
      const ctx = new AudioContext();
      try {
        const decoded = await ctx.decodeAudioData(ab);
        return await resample16kMono(decoded);
      } finally {
        await ctx.close();
      }
    },
    cancel(): void {
      try { rec.stop(); } catch { /* already stopped */ }
      release();
    },
  };
}
