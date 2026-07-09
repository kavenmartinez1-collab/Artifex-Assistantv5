/*
 * WASM bridge for espeak-ng, a faithful port of piper1-gpl's espeakbridge.c
 * (src/piper/espeakbridge.c @ v1.4.2) minus the CPython glue. Same espeak API
 * calls, same clause terminator categorisation, so the phoneme stream matches
 * piper.EspeakPhonemizer byte-for-byte.
 *
 * Exposes three functions to JS (via emscripten cwrap):
 *   bridge_init(dataDir)   -> 0 ok / -1 fail   (espeak_Initialize)
 *   bridge_set_voice(name) -> 0 ok / -1 fail   (espeak_SetVoiceByName)
 *   bridge_phonemize(text) -> malloc'd UTF-8 string, one clause per line:
 *                                 "<phonemes>\t<terminator>\t<eos 0|1>\n"
 *                             free with bridge_free().
 *
 * The JS side (src/audio/g2p.ts) then does piper's post-processing: strip
 * (lang) switch flags, append the terminator, NFD-normalise, split to
 * codepoints, and group by end-of-sentence.
 */
#include <espeak-ng/speak_lib.h>
#include <emscripten/emscripten.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* clause terminator flags — copied verbatim from piper espeakbridge.c */
#define CLAUSE_INTONATION_FULL_STOP 0x00000000
#define CLAUSE_INTONATION_COMMA 0x00001000
#define CLAUSE_INTONATION_QUESTION 0x00002000
#define CLAUSE_INTONATION_EXCLAMATION 0x00003000

#define CLAUSE_TYPE_CLAUSE 0x00040000
#define CLAUSE_TYPE_SENTENCE 0x00080000

#define CLAUSE_PERIOD (40 | CLAUSE_INTONATION_FULL_STOP | CLAUSE_TYPE_SENTENCE)
#define CLAUSE_COMMA (20 | CLAUSE_INTONATION_COMMA | CLAUSE_TYPE_CLAUSE)
#define CLAUSE_QUESTION (40 | CLAUSE_INTONATION_QUESTION | CLAUSE_TYPE_SENTENCE)
#define CLAUSE_EXCLAMATION (45 | CLAUSE_INTONATION_EXCLAMATION | CLAUSE_TYPE_SENTENCE)
#define CLAUSE_COLON (30 | CLAUSE_INTONATION_FULL_STOP | CLAUSE_TYPE_CLAUSE)
#define CLAUSE_SEMICOLON (30 | CLAUSE_INTONATION_COMMA | CLAUSE_TYPE_CLAUSE)

EMSCRIPTEN_KEEPALIVE
int bridge_init(const char *data_dir) {
    return espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, data_dir, 0) < 0 ? -1 : 0;
}

EMSCRIPTEN_KEEPALIVE
int bridge_set_voice(const char *voice) {
    return espeak_SetVoiceByName(voice) == EE_OK ? 0 : -1;
}

EMSCRIPTEN_KEEPALIVE
char *bridge_phonemize(const char *text_in) {
    const char *text = text_in;
    size_t cap = 4096, len = 0;
    char *out = (char *)malloc(cap);
    if (!out) return NULL;
    out[0] = '\0';

    while (text != NULL) {
        int terminator = 0;
        const char *phonemes = espeak_TextToPhonemesWithTerminator(
            (const void **)&text, espeakCHARS_AUTO, espeakPHONEMES_IPA, &terminator);

        terminator &= 0x000FFFFF;
        const char *ts = "";
        if (terminator == CLAUSE_PERIOD) ts = ".";
        else if (terminator == CLAUSE_QUESTION) ts = "?";
        else if (terminator == CLAUSE_EXCLAMATION) ts = "!";
        else if (terminator == CLAUSE_COMMA) ts = ",";
        else if (terminator == CLAUSE_COLON) ts = ":";
        else if (terminator == CLAUSE_SEMICOLON) ts = ";";
        int eos = ((terminator & CLAUSE_TYPE_SENTENCE) == CLAUSE_TYPE_SENTENCE) ? 1 : 0;

        const char *p = phonemes ? phonemes : "";
        size_t need = strlen(p) + strlen(ts) + 8; /* \t \t d \n \0 */
        while (len + need >= cap) {
            cap *= 2;
            char *grown = (char *)realloc(out, cap);
            if (!grown) { free(out); return NULL; }
            out = grown;
        }
        len += (size_t)sprintf(out + len, "%s\t%s\t%d\n", p, ts, eos);
    }
    return out;
}

EMSCRIPTEN_KEEPALIVE
void bridge_free(char *p) { free(p); }
