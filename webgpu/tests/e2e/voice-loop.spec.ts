/**
 * Voice-loop smoke test (headed): proves the on-device TTS→STT runtime end to
 * end in the actual browser, without a physical mic. In-page we synthesize a
 * sentence with Piper (CPU HiFiGAN, no GPU needed), resample 22050→16000, and
 * feed it into the Whisper transcribe() path. If Whisper reads it back, the
 * whole browser runtime (290 MB fetch, safetensors parse, encoder/decoder,
 * transformers.js detokenize) works — the parts node parity can't cover.
 *
 * Run: npx playwright test voice-loop --headed
 */
import { test, expect } from '@playwright/test';

test('TTS → STT round-trips a sentence in-browser', async ({ page }) => {
  test.setTimeout(300_000);
  page.on('console', (m) => console.log(`[page] ${m.text()}`));

  await page.goto('/');

  const text = await page.evaluate(async () => {
    const SENTENCE = 'The quick brown fox jumps over the lazy dog.';
    const tts = await import('/src/audio/tts.ts');
    const stt = await import('/src/audio/stt.ts');

    // Piper synth on CPU (device=null → CPU HiFiGAN); ~22050 Hz mono.
    const res = await tts.speak(null, SENTENCE, {});

    // resample to 16 kHz mono for Whisper's frontend
    const SR = 16000;
    const off = new OfflineAudioContext(1, Math.ceil((res.audio.length / res.sampleRate) * SR), SR);
    const src16 = off.createBuffer(1, res.audio.length, res.sampleRate);
    const chan = new Float32Array(res.audio.length);
    chan.set(res.audio);
    src16.copyToChannel(chan, 0);
    const node = off.createBufferSource();
    node.buffer = src16;
    node.connect(off.destination);
    node.start();
    const rendered = await off.startRendering();
    const audio16k = rendered.getChannelData(0).slice();

    return stt.transcribe(audio16k);
  });

  console.log(`[voice-loop] transcription: ${JSON.stringify(text)}`);
  expect(text.toLowerCase()).toContain('quick brown fox');
});
