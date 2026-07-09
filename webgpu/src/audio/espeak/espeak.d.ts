/**
 * Ambient types for the generated emscripten espeak-ng bridge module
 * (espeak.js + espeak.wasm + espeak.data, built by scripts/espeak-build/build.sh).
 * The .js is a checked-in build artifact; this .d.ts gives it a typed surface.
 */
export interface EspeakModule {
  cwrap(name: string, ret: string | null, args: (string | null)[]): (...args: unknown[]) => number;
  UTF8ToString(ptr: number): string;
  _bridge_free(ptr: number): void;
}

export default function createEspeak(moduleArg?: Record<string, unknown>): Promise<EspeakModule>;
