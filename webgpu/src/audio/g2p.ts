/**
 * Piper G2P (grapheme-to-phoneme) — Phase P5.
 *
 * Two layers:
 *   1. phonemize(text)      — English text → espeak-ng en-us IPA phoneme stream,
 *                             matching piper's EspeakPhonemizer. Backed by a WASM
 *                             build of espeak-ng (pinned commit 212928b, the same
 *                             the piper wheel bundles). See scripts/espeak-build/.
 *   2. phonemesToIds(...)   — phoneme stream → the model's input id sequence,
 *                             mirroring piper.phoneme_ids.phonemes_to_ids exactly.
 *
 * Gated by scripts/test-g2p-parity.mts against scripts/piper_fixture/g2p_corpus.json
 * (>= 90% exact word-level phoneme match; id mapping must be exact).
 */
import createEspeak, { type EspeakModule } from './espeak/espeak.js';

export type PhonemeIdMap = Record<string, number[]>;

// piper.phoneme_ids constants (see phoneme_id_map in the voice .onnx.json):
//   _ = PAD (0), ^ = BOS (1), $ = EOS (2), ' ' = word separator (3)
const PAD = '_';
const BOS = '^';
const EOS = '$';

/**
 * phonemes_to_ids parity port. Builds:
 *   [BOS, PAD] + flatten([id(p), PAD] for p in phonemes) + [EOS]
 * Unknown phonemes are skipped (piper logs + drops them). A phoneme whose map
 * entry has multiple ids contributes all of them (piper extends the list).
 */
export function phonemesToIds(phonemes: string[], idMap: PhonemeIdMap): number[] {
  const ids: number[] = [...idMap[BOS], ...idMap[PAD]];
  for (const p of phonemes) {
    const mapped = idMap[p];
    if (!mapped) continue; // unknown phoneme — piper drops it
    ids.push(...mapped, ...idMap[PAD]);
  }
  ids.push(...idMap[EOS]);
  return ids;
}

// espeak module is heavy to init (loads the data FS); cache it across calls.
let espeakPromise: Promise<{
  mod: EspeakModule;
  phonemize: (text: string) => number;
}> | null = null;

async function getEspeak() {
  if (!espeakPromise) {
    espeakPromise = (async () => {
      // Resolve espeak.wasm / espeak.data next to espeak.js in both Node and the
      // browser. Emscripten's .data loader otherwise resolves against the CWD.
      const base = new URL('./espeak/', import.meta.url);
      const locateFile = (path: string): string => {
        const u = new URL(path, base);
        if (u.protocol === 'file:') {
          let p = decodeURIComponent(u.pathname); // Node wants an OS path, not a file:// URL
          if (/^\/[A-Za-z]:/.test(p)) p = p.slice(1); // Windows: /C:/… → C:/…
          return p;
        }
        return u.href;
      };
      const mod = await createEspeak({ locateFile });
      const init = mod.cwrap('bridge_init', 'number', ['string']);
      const setVoice = mod.cwrap('bridge_set_voice', 'number', ['string']);
      if (init('/espeak-ng-data') !== 0) throw new Error('espeak: init failed');
      if (setVoice('en-us') !== 0) throw new Error('espeak: set_voice en-us failed');
      const phonemize = mod.cwrap('bridge_phonemize', 'number', ['string']) as (t: string) => number;
      return { mod, phonemize };
    })();
  }
  return espeakPromise;
}

/**
 * English text → espeak-ng en-us IPA phonemes (piper EspeakPhonemizer parity).
 *
 * Mirrors piper.phonemize_espeak.EspeakPhonemizer.phonemize post-processing:
 * strip (lang) switch flags, append the clause terminator (with a trailing
 * space after , : ;), NFD-normalise, and split into codepoints. Returns the
 * flat codepoint list across all clauses (word separators are the ' ' phoneme),
 * which is exactly what gen_g2p_corpus.py flattens for the parity fixture.
 */
export async function phonemize(text: string): Promise<string[]> {
  const { mod, phonemize: run } = await getEspeak();
  const ptr = run(text);
  const raw = mod.UTF8ToString(ptr);
  mod._bridge_free(ptr);

  const phonemes: string[] = [];
  for (const line of raw.split('\n')) {
    if (line === '') continue;
    const [phonStr = '', term = ''] = line.split('\t');
    let clause = phonStr.replace(/\([^)]+\)/g, ''); // drop (lang) switch flags
    clause += term;
    if (term === ',' || term === ':' || term === ';') clause += ' ';
    for (const cp of clause.normalize('NFD')) phonemes.push(cp);
  }
  return phonemes;
}
