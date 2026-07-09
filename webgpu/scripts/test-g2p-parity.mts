/**
 * Piper G2P parity gate — Phase P5.
 *
 * Node-side (no GPU). Feeds each corpus sentence through the TypeScript G2P
 * (src/audio/g2p.ts) and scores it against the espeak-ng ground truth in
 * scripts/piper_fixture/g2p_corpus.json (regenerate via gen_g2p_corpus.py).
 *
 * Two gates:
 *   - word-level exact phoneme match  >= 90%   (the P5 contract)
 *   - phoneme-id sequence exact match  (per sentence, informational + hard-fail
 *     if a matching phoneme string maps to different ids — that's a mapping bug)
 *
 * Run: npx tsx scripts/test-g2p-parity.mts
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { phonemize, phonemesToIds, type PhonemeIdMap } from '../src/audio/g2p';

const here = dirname(fileURLToPath(import.meta.url));
const fixDir = resolve(here, 'piper_fixture');
const corpus: {
  text: string;
  phonemes: string[];
  words: string[];
  ids: number[];
}[] = JSON.parse(readFileSync(resolve(fixDir, 'g2p_corpus.json'), 'utf8'));

// phoneme_id_map lives with the voice (repo-root models dir, same as the generator).
const voiceJson = resolve(here, '../../models/piper-voices/en_US-joe-medium.onnx.json');
const idMap: PhonemeIdMap = JSON.parse(readFileSync(voiceJson, 'utf8')).phoneme_id_map;

// espeak splits words on its space phoneme; mirror that so we can compare word-by-word.
function splitWords(phonemes: string[]): string[] {
  const words: string[] = [];
  let cur: string[] = [];
  for (const p of phonemes) {
    if (p === ' ') {
      if (cur.length) words.push(cur.join(''));
      cur = [];
    } else {
      cur.push(p);
    }
  }
  if (cur.length) words.push(cur.join(''));
  return words;
}

let totalWords = 0;
let matchedWords = 0;
let sentExact = 0;
let idMismatch = 0;
const misses: string[] = [];

for (const entry of corpus) {
  const got = await phonemize(entry.text);
  const gotWords = splitWords(got);
  const wantWords = entry.words;

  // word-level match: align by index (both start from the same sentence)
  const n = Math.max(gotWords.length, wantWords.length);
  for (let i = 0; i < n; i++) {
    totalWords++;
    if (gotWords[i] !== undefined && gotWords[i] === wantWords[i]) {
      matchedWords++;
    } else if (misses.length < 40) {
      misses.push(`  "${wantWords[i] ?? '∅'}"  got "${gotWords[i] ?? '∅'}"`);
    }
  }

  // whole-sentence phoneme match
  const exact = got.length === entry.phonemes.length &&
    got.every((p, i) => p === entry.phonemes[i]);
  if (exact) sentExact++;

  // id-mapping check: if our phonemes exactly match the fixture, our ids must too.
  if (exact) {
    const gotIds = phonemesToIds(got, idMap);
    const idsOk = gotIds.length === entry.ids.length &&
      gotIds.every((v, i) => v === entry.ids[i]);
    if (!idsOk) idMismatch++;
  }
}

const wordPct = (matchedWords / totalWords) * 100;
console.log(`\nG2P parity: ${corpus.length} sentences, ${totalWords} words`);
console.log(`  word-level exact match : ${matchedWords}/${totalWords} = ${wordPct.toFixed(2)}%  (gate >= 90%)`);
console.log(`  sentence exact match   : ${sentExact}/${corpus.length}`);
console.log(`  id-mapping mismatches  : ${idMismatch}  (must be 0)`);

if (misses.length) {
  console.log('\nfirst word misses (want → got):');
  for (const m of misses) console.log(m);
}

const pass = wordPct >= 90 && idMismatch === 0;
console.log(`\n=== G2P parity: ${pass ? 'PASS' : 'FAIL'} ===`);
process.exit(pass ? 0 : 1);
