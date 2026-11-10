// Artifex Holo tab — proof of concept.
//
// One still image + a depth map (Depth-Anything-V2-Small, made offline) are
// drawn as a WebGL "hologram": depth parallax for the 3D look, idle sway and
// breathing, blinks, glitches, and a jaw/lip warp that moves while a reply
// streams in or while Piper reads it aloud. Everything lives in this file;
// app.html only adds the tab, the view div and a few `window.Holo &&` hooks,
// so deleting api/static/holo/ (and those lines) removes it cleanly.
(function () {
  "use strict";
  const BASE = (document.currentScript && document.currentScript.src.replace(/[^/]*$/, "")) || "/app/holo/";
  // The shape of a picture avatar (uploaded ones come from /v1/holo/avatars
  // like this); landmarks are image pixels, y down. These values are only the
  // fallback for a photo whose face points are missing.
  const DEFAULT_AVATAR = {
    id: "default", name: "Holo", gender: "female", persona: "", w: 896, h: 1152, status: "ready",
    landmarks: {
      mouth: { c: [448, 560], dir: [1, 0], hw: 42 },
      eyes: [{ c: [368, 430], dir: [1, 0], hw: 40, hh: 20 },
             { c: [528, 430], dir: [1, 0], hw: 40, hh: 20 }],
    },
  };
  const AVATAR_KEY = "ax_holo_avatar";
  const DEFAULT_EDIT_KEY = "ax_holo_default";   // this phone's edits to the built-in avatar

  // The built-in avatar has no server manifest: its name, persona, voice set
  // and dots can still be changed, and are kept on this phone.
  function defaultAvatar() {
    let o = {};
    try { o = JSON.parse(localStorage.getItem(DEFAULT_EDIT_KEY) || "{}") || {}; } catch (e) {}
    const a = Object.assign({}, DEFAULT_AVATAR);
    if (typeof o.name === "string" && o.name.trim()) a.name = o.name.trim().slice(0, 40);
    if (typeof o.persona === "string") a.persona = o.persona.slice(0, 2500);
    if (["female", "male", "neutral"].includes(o.gender)) a.gender = o.gender;
    if (o.landmarks && o.landmarks.eyes && o.landmarks.mouth) { a.landmarks = o.landmarks; a.landmark_source = "user"; }
    return a;
  }
  // The second built-in has no picture at all: "Ember", an orb of light drawn
  // entirely by a shader (rings, traces and a core that follow the voice).
  const ORB_AVATAR = { id: "orb", kind: "orb", name: "Ember", gender: "male", persona: "", status: "ready",
                       voice_pref: /alan|joe/i };
  const ORB_EDIT_KEY = "ax_holo_orb";           // this phone's edits to it
  const BUILTIN_FIRST = "orb";                   // the avatar a new phone starts with
  function orbAvatar() {
    let o = {};
    try { o = JSON.parse(localStorage.getItem(ORB_EDIT_KEY) || "{}") || {}; } catch (e) {}
    const a = Object.assign({}, ORB_AVATAR);
    if (typeof o.name === "string" && o.name.trim()) a.name = o.name.trim().slice(0, 40);
    if (typeof o.persona === "string") a.persona = o.persona.slice(0, 2500);
    if (["female", "male", "neutral"].includes(o.gender)) a.gender = o.gender;
    return a;
  }
  function isBuiltin(a) { return !!a && (a.id === "default" || a.id === "orb"); }
  // This build ships no portrait: the orb is the only built-in, and every
  // picture avatar is one the user made from a photo.
  const HAS_PORTRAIT = false;
  function builtinById(id) { return id === "orb" || !HAS_PORTRAIT ? orbAvatar() : defaultAvatar(); }
  function builtins() {
    if (!HAS_PORTRAIT) return [orbAvatar()];
    return BUILTIN_FIRST === "orb" ? [orbAvatar(), defaultAvatar()] : [defaultAvatar(), orbAvatar()];
  }
  // What this phone last showed, when that was a built-in (uploaded ones
  // arrive with the avatar list); else the shipped first choice.
  function startAvatar() {
    let want = null; try { want = localStorage.getItem(AVATAR_KEY); } catch (e) {}
    return want === "orb" || (want === "default" && HAS_PORTRAIT) ? builtinById(want) : builtins()[0];
  }
  function builtinUrls(a) {
    return a.kind === "orb" ? { orb: true } : { img: BASE + "avatar.jpg", depth: BASE + "avatar_depth.png" };
  }
  const PROMPT_KEY = "ax_holo_prompt";   // both | system | persona
  const PROMPT_MODES = [["both", "Artifex + persona"], ["system", "Artifex prompt only"], ["persona", "Persona only"]];
  const VOICE_KEY = "ax_holo_voice", SPEAK_KEY = "ax_holo_speak";
  // Her brain, smartest first. All three sit on the 5060 Ti alone, so they
  // leave the 6700 XT to the Orpheus (tara) voice; any other model on the PC
  // can be picked too (tap "core" in the HUD), and Tara falls back to Jenny
  // when that model needs both cards.
  const HOLO_BRAINS = ["qwen3.5-9b-abliterated", "qwen3.6-35b-a3b", "gemma-4-e4b-abliterated"];
  const MODEL_KEY = "ax_holo_model";

  const VS = `attribute vec2 aPos; varying vec2 vUv;
void main(){ vUv = aPos*0.5+0.5; gl_Position = vec4(aPos,0.0,1.0); }`;

  const FS = `precision highp float;
varying vec2 vUv;
uniform sampler2D uImg, uDepth;
uniform vec2 uRes, uTilt;
uniform vec3 uView;   // pan x, pan y, zoom
uniform float uRot;   // twist, radians
uniform float uT, uMouth, uBlink, uGlitch, uThink, uSpeak;
uniform vec2 uMC, uMD; uniform float uMW;
uniform vec4 uE0, uE1; uniform vec2 uE0d, uE1d;
uniform vec2 IMG;     // avatar size in pixels

float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

// Eyelids meet: each half of the eye samples outward toward its own lid, so
// the lash lines slide to the middle (monotonic map, continuous at the edge).
vec2 blinkWarp(vec2 p, vec2 c, vec2 d, float hw, float hh, float b){
  vec2 n = vec2(-d.y, d.x);
  vec2 q = p - c;
  float ex = dot(q, d) / hw, ey = dot(q, n) / hh;
  float hl = sqrt(max(0.0, 1.0 - ex*ex));
  if (hl < 0.02 || b < 0.001) return p;
  float v = abs(ey) / hl;
  if (v > 1.6) return p;
  float f = v + b * (1.6 - v) / 1.6;
  return c + d*ex*hw + n*(sign(ey)*f*hl*hh);
}

void main(){
  // contain-fit the portrait into the canvas, image coords y-down
  float ca = uRes.x/uRes.y, ia = IMG.x/IMG.y;
  vec2 sc = ca > ia ? vec2(ca/ia, 1.0) : vec2(1.0, ia/ca);
  vec2 q = (vUv - 0.5 - uView.xy) * uRes;          // pixels, so rotation keeps proportions
  float cr = cos(uRot), sr = sin(uRot);
  q = vec2(cr * q.x + sr * q.y, -sr * q.x + cr * q.y);
  vec2 sv = q / uRes / uView.z + 0.5;
  vec2 uv = (sv - 0.5) * sc / 0.96 + 0.5;
  uv.y = 1.0 - uv.y;
  uv.y -= 0.015;

  // glitch: a few horizontal slices jump sideways
  float row = floor(vUv.y * 46.0);
  float gk = hash(vec2(row, floor(uT * 24.0)));
  if (gk < uGlitch * 0.35) uv.x += (hash(vec2(row, uT)) - 0.5) * 0.06 * uGlitch;

  // idle body motion: slow sway around the neck, breathing
  vec2 piv = vec2(0.5, 0.9);
  float ang = sin(uT*0.41)*0.006 + sin(uT*0.17)*0.004;
  vec2 r = uv - piv; r.x *= IMG.x/IMG.y;
  r = mat2(cos(ang), -sin(ang), sin(ang), cos(ang)) * r;
  r.x /= IMG.x/IMG.y; uv = piv + r;
  uv.y = 1.0 - (1.0 - uv.y) * (1.0 + 0.004*sin(uT*1.5));

  // depth parallax (fixed-point refine so steps don't tear as much)
  vec2 off = uTilt * 0.045;
  vec2 uv0 = uv;
  for (int i = 0; i < 8; i++) {
    float dd = texture2D(uDepth, uv).r;
    uv = uv0 + off * (dd - 0.55);
  }

  // face: blink + talking jaw, in image pixels
  vec2 p = uv * IMG;
  p = blinkWarp(p, uE0.xy, uE0d, uE0.z, uE0.w, uBlink);
  p = blinkWarp(p, uE1.xy, uE1d, uE1.z, uE1.w, uBlink);
  vec2 mn = vec2(-uMD.y, uMD.x);
  vec2 mq = p - uMC;
  float mx = dot(mq, uMD) / uMW, my = dot(mq, mn);
  float open = uMouth * 14.0;
  float ws = 1.0 - smoothstep(0.55, 1.25, abs(mx));
  float wi = max(0.0, 1.0 - mx*mx);
  float gap = open * wi;
  float inside = 0.0;
  if (my > -0.5 && open > 0.01) {
    float g = 1.0 - smoothstep(14.0, 80.0, my);
    p -= mn * open * ws * g;
    inside = smoothstep(-0.5, 0.6, my) * (1.0 - smoothstep(gap - 1.4, gap, my)) * step(0.6, gap);
  }
  uv = p / IMG;

  float edge = smoothstep(0.0, 0.05, uv.x) * smoothstep(1.0, 0.95, uv.x)
             * smoothstep(0.0, 0.03, uv.y) * smoothstep(1.0, 0.8, uv.y);
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) edge = 0.0;

  float caw = 0.0012 + uGlitch * 0.006 + uThink * 0.0008;
  vec3 col = vec3(texture2D(uImg, uv + vec2(caw, 0.0)).r,
                  texture2D(uImg, uv).g,
                  texture2D(uImg, uv - vec2(caw, 0.0)).b);
  // mouth interior + a hint of upper teeth
  vec3 mouthCol = mix(vec3(0.16, 0.02, 0.08), vec3(0.85, 0.82, 0.92),
                      smoothstep(0.32, 0.05, my / max(gap, 0.001)) * step(4.0, gap));
  col = mix(col, mouthCol, inside);

  float d = texture2D(uDepth, uv).r;
  float fig = smoothstep(0.17, 0.31, d);
  float e = 2.0 / IMG.x;
  float gx = texture2D(uDepth, uv + vec2(e, 0.0)).r - texture2D(uDepth, uv - vec2(e, 0.0)).r;
  float gy = texture2D(uDepth, uv + vec2(0.0, e)).r - texture2D(uDepth, uv - vec2(0.0, e)).r;
  float rim = clamp(length(vec2(gx, gy)) * 14.0, 0.0, 1.0);

  // hologram grade
  float lum = dot(col, vec3(0.3, 0.59, 0.11));
  vec3 holo = mix(vec3(0.0, 0.85, 1.0), vec3(1.0, 0.35, 0.8), uThink * 0.6 + 0.15 * sin(uT * 0.7 + vUv.y * 3.0));
  col = mix(col, holo * (0.35 + lum * 1.2), 0.22 + 0.18 * uThink);
  col += vec3(0.0, 0.9, 1.0) * rim * 0.55 * fig;
  float sl = 0.84 + 0.16 * sin(vUv.y * uRes.y * 1.6);
  float band = exp(-pow((fract(vUv.y * 0.7 - uT * (0.10 + 0.25 * uThink)) - 0.5) * 16.0, 2.0));
  col *= sl;
  col += holo * band * 0.22;
  col += (hash(vUv * uRes + uT) - 0.5) * 0.06;
  col *= 1.0 + 0.12 * uSpeak;

  float flick = 0.93 + 0.07 * sin(uT * 53.0) * sin(uT * 7.3) - uGlitch * 0.25;
  float a = mix(0.1, 0.93, fig) * edge * flick;
  a *= 0.92 + 0.08 * band;

  // rising motes
  vec2 g2 = vUv * vec2(18.0, 26.0); g2.y -= uT * (0.6 + uThink);
  vec2 cell = floor(g2), fr = fract(g2) - 0.5;
  float h = hash(cell);
  vec2 jit = vec2(hash(cell + 3.1), hash(cell + 7.7)) - 0.5;
  float mote = step(0.86, h) * smoothstep(0.09, 0.0, length(fr - jit * 0.6)) * (0.5 + 0.5 * sin(uT * 3.0 + h * 40.0));
  mote *= smoothstep(0.0, 0.35, vUv.y) * 0.8;

  vec3 outc = col * a + vec3(0.2, 0.95, 1.0) * mote;
  gl_FragColor = vec4(outc, clamp(a + mote, 0.0, 1.0));
}`;

  // The orb: three nested shells of broken light bands (each pixel's ray hits
  // a shell on its near and far side, so you see through it), two comet rings,
  // segmented rims, sparks and a core. uPhase is time that runs faster while
  // it speaks or thinks; uLevel is the voice's loudness.
  const FS_ORB = `precision highp float;
varying vec2 vUv;
uniform vec2 uRes, uTilt;
uniform vec3 uView;
uniform float uRot, uT, uPhase, uLevel, uGlitch, uThink, uSpeak;

float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
mat3 rotX(float a){ float c = cos(a), s = sin(a); return mat3(1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c); }
mat3 rotY(float a){ float c = cos(a), s = sin(a); return mat3(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c); }
mat3 rotZ(float a){ float c = cos(a), s = sin(a); return mat3(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0); }

// light on a shell, d = unit direction in the shell's own frame
float bandsOf(vec3 d, float seed, float bands){
  float lat = asin(clamp(d.y, -1.0, 1.0)) / 1.5708;
  float lon = atan(d.z, d.x) / 6.2832 + 0.5;
  float v = (lat * 0.5 + 0.5) * bands;
  float row = floor(v), f = abs(fract(v) - 0.5);
  float rs = hash(vec2(row, seed));
  float segs = floor(5.0 + rs * 20.0);
  float u = fract(lon + uPhase * 0.07 * (rs - 0.5)) * segs;      // every band drifts at its own pace
  float cell = floor(u), g = fract(u);
  float h = hash(vec2(cell + row * 31.0, seed * 7.0 + 1.0));
  float fl = 0.62 + 0.38 * sin(uT * (2.0 + h * 9.0) + h * 40.0);
  float ends = smoothstep(0.0, 0.05, g) * smoothstep(1.0, 0.95, g);
  float line = (1.0 - smoothstep(0.03, 0.11, f)) * step(0.40, h);
  float blk = (1.0 - smoothstep(0.30, 0.36, f)) * step(0.90, h) * 0.5;
  float tick = (1.0 - smoothstep(0.0, 0.035, g)) * step(0.72, hash(vec2(cell, row + seed)))
             * (1.0 - smoothstep(0.30, 0.45, f));
  return ((line + blk) * ends + tick) * fl * smoothstep(1.0, 0.82, abs(lat));
}

// a comet running round a great circle of the shell
float comet(vec3 d, float speed){
  float on = 1.0 - smoothstep(0.0, 0.045, abs(d.y));
  float a = fract(atan(d.z, d.x) / 6.2832 - uPhase * speed);
  return on * (pow(a, 5.0) * 1.6 + 0.10);
}

void main(){
  vec2 q = (vUv - 0.5 - uView.xy) * uRes;
  float cr = cos(uRot), sr = sin(uRot);
  q = vec2(cr * q.x + sr * q.y, -sr * q.x + cr * q.y);
  vec2 p = q / min(uRes.x, uRes.y) / uView.z;
  p.y -= 0.03;

  float row = floor(vUv.y * 46.0);
  if (hash(vec2(row, floor(uT * 24.0))) < uGlitch * 0.35) p.x += (hash(vec2(row, uT)) - 0.5) * 0.05 * uGlitch;

  float lvl = clamp(uLevel, 0.0, 1.0);
  float breathe = 1.0 + 0.012 * sin(uT * 1.3) + 0.035 * lvl;
  vec3 amber = mix(vec3(1.0, 0.56, 0.09), vec3(1.0, 0.74, 0.34), uThink * 0.6);
  vec3 hot = vec3(1.0, 0.90, 0.60);
  mat3 tilt = rotX(-uTilt.y * 0.34) * rotY(uTilt.x * 0.34);
  float r = length(p), r2 = r * r;
  float ang = atan(p.y, p.x) / 6.2832 + 0.5;
  vec3 col = vec3(0.0);

  for (int i = 0; i < 3; i++) {
    float fi = float(i);
    float R = (0.40 - 0.115 * fi) * breathe;
    float dir = mod(fi, 2.0) < 0.5 ? 1.0 : -1.0;
    if (r2 < R * R) {
      float z = sqrt(R * R - r2);
      mat3 m = rotY(uPhase * (0.20 + 0.13 * fi) * dir) * rotX(0.45 + fi * 0.95) * rotZ(fi * 0.7) * tilt;
      float nb = 16.0 - 4.5 * fi;
      float front = bandsOf(m * (vec3(p, z) / R), fi + 1.0, nb);
      float back = bandsOf(m * (vec3(p, -z) / R), fi + 1.0, nb);
      col += amber * (front + 0.42 * back) * (0.85 + 0.22 * fi + 0.7 * lvl);
    }
    // the shell's outline, in arcs that turn
    float arcs = step(0.38, hash(vec2(floor(fract(ang + uPhase * 0.03 * dir * (fi + 1.0)) * (26.0 - 6.0 * fi)), fi + 9.0)));
    col += amber * exp(-pow((r - R) / 0.0035, 2.0)) * arcs * (0.55 + 0.5 * lvl);
  }

  // two comets on their own rings
  float Rc = 0.27 * breathe;
  if (r2 < Rc * Rc) {
    float z = sqrt(Rc * Rc - r2);
    mat3 m1 = rotX(1.15 + 0.2 * sin(uPhase * 0.11)) * rotZ(0.5) * tilt;
    mat3 m2 = rotX(-0.7) * rotZ(-0.9 + 0.2 * cos(uPhase * 0.09)) * tilt;
    vec3 f = vec3(p, z) / Rc, b = vec3(p, -z) / Rc;
    float c = comet(m1 * f, 0.33) + 0.5 * comet(m1 * b, 0.33) + comet(m2 * f, -0.24) + 0.5 * comet(m2 * b, -0.24);
    col += mix(amber, hot, 0.5) * c * (0.75 + 0.9 * lvl);
  }

  // the core: flares with the voice, breathes when idle
  float pulse = 0.5 + 0.5 * sin(uT * 2.1);
  float cw = 0.0028 * (1.0 + 2.2 * lvl + 0.25 * pulse + 0.5 * uThink * sin(uT * 7.0));
  col += hot * exp(-r2 / cw) * (1.3 + 2.0 * lvl);
  col += amber * exp(-r / 0.10) * (0.30 + 0.55 * lvl + 0.10 * pulse);
  col += amber * exp(-max(r - 0.40 * breathe, 0.0) / 0.05) * 0.07 * (1.0 + lvl);

  // sparks drifting out of it
  vec2 pc = vec2(ang * 40.0, r * 22.0 - uT * (0.5 + 0.9 * uSpeak + 0.5 * uThink));
  vec2 cell = floor(pc), fr = fract(pc) - 0.5;
  float h = hash(cell);
  vec2 jit = vec2(hash(cell + 3.1), hash(cell + 7.7)) - 0.5;
  float spark = step(0.90, h) * smoothstep(0.16, 0.0, length(fr - jit * 0.6)) * (0.5 + 0.5 * sin(uT * 3.0 + h * 40.0));
  col += mix(amber, hot, 0.35) * 1.4 * spark * smoothstep(0.62, 0.30, r) * smoothstep(0.03, 0.12, r) * 0.9;

  col *= 0.90 + 0.10 * sin(vUv.y * uRes.y * 1.6);
  col *= 0.94 + 0.06 * sin(uT * 53.0) * sin(uT * 7.3) - uGlitch * 0.2;
  col += (hash(vUv * uRes + uT) - 0.5) * 0.03 * step(0.02, max(col.r, col.g));
  col = max(col, 0.0);
  gl_FragColor = vec4(col, clamp(max(col.r, max(col.g, col.b)), 0.0, 1.0));
}`;

  const CSS = `
#holo-view.orb { background:
  radial-gradient(55% 40% at 50% 46%, rgba(255,150,30,.16), transparent 72%),
  radial-gradient(120% 70% at 50% 100%, rgba(40,110,255,.16), transparent 60%), var(--bg); }
#holo-view { overflow: hidden; background:
  radial-gradient(120% 70% at 50% 100%, rgba(0,240,255,.16), transparent 60%),
  radial-gradient(90% 60% at 50% 35%, rgba(255,42,109,.10), transparent 70%), var(--bg); }
.holo-stage { position: relative; flex: 1; min-height: 0; touch-action: none; }
.holo-stage canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
.holo-grid { position: absolute; left: -20%; right: -20%; bottom: 0; height: 32%;
  background-image: linear-gradient(rgba(0,240,255,.22) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(0,240,255,.22) 1px, transparent 1px);
  background-size: 34px 34px; transform: perspective(260px) rotateX(62deg); transform-origin: 50% 100%;
  -webkit-mask-image: linear-gradient(transparent, #000 70%); mask-image: linear-gradient(transparent, #000 70%);
  animation: holo-grid 3.2s linear infinite; pointer-events: none; }
@keyframes holo-grid { to { background-position: 0 34px; } }
.holo-base { position: absolute; left: 50%; bottom: 3%; width: 62%; aspect-ratio: 4 / 1; transform: translateX(-50%);
  pointer-events: none; }
/* round rings squashed flat, so the dashed one spins without wobbling */
.holo-base > div { position: absolute; left: 0; top: 50%; width: 100%; aspect-ratio: 1; transform: translateY(-50%) scaleY(.25); }
.holo-base i { position: absolute; inset: 0; border-radius: 50%; border: 2px solid rgba(0,240,255,.55);
  box-shadow: 0 0 18px rgba(0,240,255,.5), inset 0 0 18px rgba(0,240,255,.35); }
.holo-base i:nth-child(2) { inset: 15%; border-style: dashed; border-color: rgba(255,42,109,.6);
  animation: holo-spin 9s linear infinite; }
.holo-base i:nth-child(3) { inset: 32%; background: radial-gradient(rgba(0,240,255,.7), transparent 70%); border: 0; }
.holo-beam { position: absolute; left: 50%; bottom: 8%; width: 46%; height: 78%; transform: translateX(-50%);
  background: linear-gradient(to top, rgba(0,240,255,.16), transparent 80%);
  clip-path: polygon(30% 100%, 70% 100%, 100% 0, 0 0); pointer-events: none; mix-blend-mode: screen; }
@keyframes holo-spin { to { transform: rotate(360deg); } }
.holo-stage.think .holo-base i:nth-child(2) { animation-duration: 1.6s; }
.holo-hud { position: absolute; top: 8px; font: 600 10px/1.5 var(--hud); letter-spacing: 1.6px; color: var(--accent);
  text-shadow: 0 0 8px rgba(0,240,255,.6); pointer-events: none; text-transform: uppercase; }
.holo-hud.l { left: calc(var(--gut) + var(--sal)); }
.holo-hud { z-index: 2; padding: 4px 7px; border-radius: 3px; background: rgba(4,5,12,.55);
  -webkit-backdrop-filter: blur(3px); backdrop-filter: blur(3px); }
.holo-hud.r { right: calc(var(--gut) + var(--sar)); text-align: right; color: var(--accent2); pointer-events: auto;
  border: 1px solid rgba(255,42,109,.35); cursor: pointer; min-width: var(--tap); appearance: none; -webkit-appearance: none;
  text-shadow: 0 0 8px rgba(255,42,109,.6); }
.holo-hud b { color: var(--accent3); font-weight: 700; }
.holo-setup { display: inline-block; margin-top: 3px; padding: 1px 6px; border: 1px solid currentColor; border-radius: 2px; }
#holo-diag:empty { display: none; }
.holo-sheet { position: absolute; left: 0; right: 0; bottom: 0; z-index: 6; display: flex; flex-direction: column; gap: 10px;
  padding: 12px calc(var(--gut) + var(--sar)) calc(12px + var(--sab)) calc(var(--gut) + var(--sal));
  background: var(--panel, #0d0f1e); border-top: 1px solid var(--accent); box-shadow: 0 -8px 30px rgba(0,0,0,.6); }
.holo-sheet-h { display: flex; align-items: center; justify-content: space-between; font: 700 12px var(--hud);
  letter-spacing: 2px; color: var(--accent); }
.holo-sheet-h .btn { width: var(--tap); min-height: var(--tap); padding: 0; }
.holo-field { display: flex; flex-direction: column; gap: 4px; font: 600 10px var(--hud); letter-spacing: 1.4px;
  color: var(--dim); text-transform: uppercase; }
.holo-field select, .holo-ed-row select { min-height: var(--tap); font: 16px var(--hud); text-transform: none; letter-spacing: 0;
  background: var(--code-bg, #05060c); color: var(--text); border: 1px solid var(--border); border-radius: 3px; padding: 0 10px; }
.holo-sheet-row { display: flex; gap: 8px; }
.holo-sheet-row .btn { flex: 1; width: auto; }
.holo-caption { margin: 0 calc(var(--gut) + var(--sar)) 6px calc(var(--gut) + var(--sal)); flex: none;
  max-height: calc(3 * 1.4em + 16px); box-sizing: border-box;
  -webkit-mask-image: linear-gradient(transparent 0, #000 1.2em); mask-image: linear-gradient(transparent 0, #000 1.2em);
  overflow-y: auto; -webkit-overflow-scrolling: touch;
  padding: 8px 10px; border: 1px solid var(--border); border-left: 2px solid var(--accent);
  background: var(--glass); font-size: 13px; line-height: 1.4; color: var(--text); white-space: pre-wrap; }
.holo-caption:empty { -webkit-mask-image: none; mask-image: none; }
.holo-caption:empty::before { content: "Say something."; color: var(--dim); }
.holo-composer { display: flex; gap: 8px; align-items: flex-end;
  padding: 0 calc(var(--gut) + var(--sar)) calc(8px + var(--sab)) calc(var(--gut) + var(--sal)); }
/* app.html's .btn is width:100% - that crushed the textarea to a sliver.
   16px text: iOS zooms the page into any smaller focused input. */
.holo-composer textarea { flex: 1 1 auto; min-width: 0; width: auto; resize: none; font-size: 16px;
  background: var(--code-bg, #05060c); color: var(--text); border: 1px solid var(--border); border-radius: 3px;
  font-family: -apple-system, "Segoe UI", system-ui, sans-serif; line-height: 1.35;
  min-height: var(--tap); max-height: 96px; padding: 10px 12px; }
.holo-composer .btn { flex: none; width: auto; min-width: var(--tap); min-height: var(--tap); padding: 0 14px; }
.holo-composer .holo-voice { width: var(--tap); padding: 0; }
.holo-composer .holo-voice { font-size: 20px; color: var(--dim); }
.holo-composer .holo-voice.on { color: var(--accent); border-color: var(--accent); box-shadow: 0 0 10px rgba(0,240,255,.45), inset 0 0 8px rgba(0,240,255,.25); }
@media (max-height: 500px) {
  .holo-caption { max-height: calc(1.4em + 14px); }
  .holo-hud { font-size: 9px; line-height: 1.35; }
  .holo-sheet { overflow-y: auto; max-height: 100%; }
}
@media (max-width: 340px) { .holo-hud { font-size: 9px; letter-spacing: 1px; } }
.holo-fallback { position: absolute; inset: 6% 8% 12%; background: center / contain no-repeat;
  filter: drop-shadow(0 0 14px rgba(0,240,255,.6)) saturate(1.2); opacity: .9;
  animation: holo-float 5s ease-in-out infinite; }
@keyframes holo-float { 50% { transform: translateY(-6px); } }
.holo-editor { position: absolute; inset: 0; z-index: 5; display: flex; flex-direction: column; gap: 8px;
  padding: 8px calc(var(--gut) + var(--sar)) 8px calc(var(--gut) + var(--sal)); background: var(--bg); touch-action: none; }
.holo-ed-pic { position: relative; flex: 1; min-height: 0; }
.holo-ed-pic img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; opacity: .9; user-select: none; -webkit-user-drag: none; }
.holo-dot { position: absolute; width: 22px; height: 22px; margin: -11px 0 0 -11px; border-radius: 50%; touch-action: none;
  border: 2px solid var(--accent3); background: rgba(252,238,10,.25); box-shadow: 0 0 8px rgba(252,238,10,.7); }
.holo-dot.m { border-color: var(--accent2); background: rgba(255,42,109,.25); box-shadow: 0 0 8px rgba(255,42,109,.7); }
.holo-dot::after { content: ""; position: absolute; left: 50%; top: 50%; width: 4px; height: 4px; margin: -2px 0 0 -2px;
  border-radius: 50%; background: currentColor; }
.holo-dot { color: var(--accent3); }
.holo-dot.m { color: var(--accent2); }
.holo-dot::before { content: ""; position: absolute; inset: -11px; }   /* a 44px grab area */
.holo-dot span { position: absolute; top: 22px; left: 50%; transform: translateX(-50%); font: 600 9px var(--hud);
  letter-spacing: 1px; color: var(--text); white-space: nowrap; text-shadow: 0 0 4px #000; }
/* mouth labels sit outside their corner, so close corners never collide */
.holo-dot.lft span { top: 50%; left: auto; right: 26px; transform: translateY(-50%); }
.holo-dot.rgt span { top: 50%; left: 26px; transform: translateY(-50%); }
.holo-ed-row { display: flex; gap: 8px; align-items: center; }
.holo-ed-row input, .holo-ed-row textarea { flex: 1; min-width: 0; font: 16px/1.35 -apple-system, "Segoe UI", system-ui, sans-serif;
  background: var(--code-bg, #05060c); color: var(--text); border: 1px solid var(--border); border-radius: 3px; padding: 8px 10px; }
.holo-ed-row select { flex: none; }
.holo-ed-row textarea { resize: none; height: 5.5em; }
.holo-ed-row .btn { flex: 1; width: auto; }
.holo-ed-hint { font: 11px/1.4 var(--hud); color: var(--dim); letter-spacing: .5px; }
`;

  // ── speech → mouth targets ──────────────────────────────────
  function charOpen(ch) {
    ch = ch.toLowerCase();
    if ("ao".includes(ch)) return 1.0;
    if ("eiuy".includes(ch)) return 0.62;
    if ("mbp".includes(ch)) return 0.0;
    if ("fv".includes(ch)) return 0.15;
    if (/[a-z0-9]/.test(ch)) return 0.35;
    if (/[.,!?;:\n]/.test(ch)) return -1;   // a short pause
    return 0.05;
  }

  const H = {
    el: null, stage: null, canvas: null, gl: null, prog: null, u: {},
    visible: false, raf: 0, t0: performance.now(),
    state: "idle",                 // idle | think | speak
    queue: "", qAcc: 0, mouth: 0, mouthT: 0, pauseT: 0,
    blink: 0, nextBlink: 2, blinkT: -1, glitch: 0, nextGlitch: 4,
    tilt: [0, 0], drag: null, dragOff: [0, 0], pinch: null,
    view: [0, 0, 1],               // pan x, pan y (screen fractions), zoom
    say: null,                     // {text, audio} while Piper reads
    sawText: false, caption: "",
    sp: { gen: 0, q: [], buf: "", n: 0, inCode: false, playing: false, live: false, cur: null,
          abort: new AbortController(), chain: Promise.resolve(), srcs: [], head: 0 },
    ac: null, an: null, anBuf: null,
  };

  function mount(el) {
    H.el = el;
    const st = document.createElement("style"); st.textContent = CSS; document.head.appendChild(st);
    el.innerHTML = `
      <div class="holo-stage">
        <div class="holo-grid"></div><div class="holo-beam"></div>
        <div class="holo-base"><div><i></i><i></i><i></i></div></div>
        <canvas></canvas>
        <div class="holo-hud l">Artifex // holo-link<br>state: <b id="holo-state">idle</b><br><span id="holo-diag" style="color:var(--dim)"></span></div>
        <button class="holo-hud r" id="holo-hudr" aria-label="Holo setup">sync <span id="holo-sync">98.2</span>%<br><span id="holo-vname">voice</span><br><span id="holo-model">core</span><br><span id="holo-pname">prompt</span><br><span id="holo-aname">avatar</span><br><span class="holo-setup">&#9881; setup</span></button>
        <input type="file" id="holo-file" accept="image/*" class="hidden">
        <div class="holo-editor hidden" id="holo-editor"></div>
        <div class="holo-sheet hidden" id="holo-sheet" role="dialog" aria-label="Holo setup">
          <div class="holo-sheet-h">HOLO SETUP<button class="btn" id="holo-sheet-x" aria-label="Close">&#10005;</button></div>
          <label class="holo-field"><span>Voice</span><select id="holo-vsel"></select></label>
          <label class="holo-field"><span>Brain</span><select id="holo-msel"></select></label>
          <label class="holo-field"><span>System prompt</span><select id="holo-psel"></select></label>
          <label class="holo-field"><span>Avatar</span><select id="holo-asel"></select></label>
          <div class="holo-sheet-row"><button class="btn" id="holo-add">New avatar from a photo</button><button class="btn" id="holo-edit">Edit face &amp; persona</button></div>
          <div class="holo-sheet-row"><button class="btn" id="holo-reset">Reset view</button><button class="btn" id="holo-sheet-done">Done</button></div>
        </div>
      </div>
      <div class="holo-caption" id="holo-caption"></div>
      <div class="holo-composer">
        <button class="btn holo-voice" id="holo-voice" aria-label="Read replies aloud">&#128263;</button>
        <textarea id="holo-input" rows="1" placeholder="Talk to Ember..." enterkeyhint="send"
          autocapitalize="sentences" autocorrect="on"></textarea>
        <button class="btn" id="holo-send">Send</button>
      </div>`;
    H.stage = el.querySelector(".holo-stage");
    H.canvas = el.querySelector("canvas");
    const input = H.inputEl = el.querySelector("#holo-input");
    const send = () => {
      if (speakOn()) audioCtx();   // the tap that lets her speak (iOS)
      if (typeof sendMessage !== "function") return;
      // Stop: ends the reply AND whatever she is still saying
      if (typeof isStreaming !== "undefined" && isStreaming) { speechStop(); sendOrStop(); return; }
      const txt = input.value.trim();
      // still talking: an empty box means "stop", a typed message cuts in
      if (speakingNow()) { speechStop(); if (!txt) return; }
      if (!txt) return;
      if (typeof store !== "undefined" && !store.k) { diag("set the API key in Settings first"); return; }
      input.value = ""; input.style.height = "auto";
      sendMessage(txt);      // text only: the Chat tab's draft and photos stay put
    };
    el.querySelector("#holo-send").onclick = send;
    input.addEventListener("keydown", e => {
      if (e.key === "Enter" && !e.shiftKey && !("ontouchstart" in window)) { e.preventDefault(); send(); }
    });
    input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = input.scrollHeight + "px"; });
    const vb = el.querySelector("#holo-voice");
    const vbShow = () => { vb.classList.toggle("on", speakOn()); vb.innerHTML = speakOn() ? "&#128266;" : "&#128263;"; };
    vbShow();
    vb.onclick = () => {
      audioCtx();
      try { localStorage.setItem(SPEAK_KEY, speakOn() ? "0" : "1"); } catch (e) {}
      vbShow();
      if (typeof unlockAudio === "function") unlockAudio();
      if (!speakOn()) { speechStop(); H.sp.live = false; }   // liveTurn stays: no full re-read at the end
      else speechWarm();
    };
    // The HUD labels sit over native pickers (voice, brain, prompt, avatar).
    el.querySelector(".holo-hud.r").style.pointerEvents = "auto";
    const vs = el.querySelector("#holo-vsel"), ms = el.querySelector("#holo-msel"), as = el.querySelector("#holo-asel");
    vs.onchange = () => { try { localStorage.setItem(voiceKey(), vs.value); } catch (e) {} updateVoiceLabel(); H.sp.warm = null; speechWarm(); };
    ms.onchange = () => { try { localStorage.setItem(MODEL_KEY, ms.value); } catch (e) {} updateModelLabel(); brainWarm(); };
    const ps = el.querySelector("#holo-psel");
    ps.onchange = () => { try { localStorage.setItem(PROMPT_KEY, ps.value); } catch (e) {} updatePromptLabel(); };
    as.onchange = () => {
      const a = (H.avatars || builtins()).find(x => x.id === as.value);
      if (a) chooseAvatar(a);
    };
    const sheet = el.querySelector("#holo-sheet");
    const sheetShow = on => { fillPickers(); sheet.classList.toggle("hidden", !on); };
    el.querySelector("#holo-hudr").onclick = () => sheetShow(sheet.classList.contains("hidden"));
    el.querySelector("#holo-sheet-x").onclick = () => sheetShow(false);
    el.querySelector("#holo-sheet-done").onclick = () => sheetShow(false);
    el.querySelector("#holo-reset").onclick = () => { resetView(); sheetShow(false); };
    sheet.addEventListener("pointerdown", e => e.stopPropagation());
    // a real tap opens the photo picker (iOS ignores a click from a <select>)
    el.querySelector("#holo-add").onclick = () => { sheetShow(false); el.querySelector("#holo-file").click(); };
    el.querySelector("#holo-file").onchange = e => { const f = e.target.files[0]; e.target.value = ""; if (f) uploadAvatar(f); };
    el.querySelector("#holo-edit").onclick = () => { sheetShow(false); openEditor(); };
    H.avatar = startAvatar(); H.urls = builtinUrls(H.avatar);
    el.classList.toggle("orb", H.avatar.kind === "orb");

    // One finger turns her (and stays where you leave it); two fingers pinch
    // to zoom and drag to pan; the wheel zooms; double-tap resets the view.
    const pts = new Map();
    const PAN = 1.0, ZMIN = 0.5, ZMAX = 4, TILT = 2.6;   // pan in screen widths, zoom range, 3D turn
    const pinch = () => {
      const [a, b] = [...pts.values()];
      return { d: Math.hypot(a.x - b.x, a.y - b.y) || 1, x: (a.x + b.x) / 2, y: (a.y + b.y) / 2,
               ang: Math.atan2(b.y - a.y, b.x - a.x) };
    };
    const regrab = () => {
      H.drag = null; H.pinch = null; H.pan = null;
      if (pts.size === 1) {
        const p = [...pts.values()][0];
        if (p.pan) H.pan = [p.x, p.y, H.view[0], H.view[1]];          // mouse: shift / right button
        else H.drag = [p.x, p.y, H.dragOff[0], H.dragOff[1]];
      }
      if (pts.size === 2) H.pinch = Object.assign(pinch(), { zoom: H.view[2], px: H.view[0], py: H.view[1], rot: H.rot || 0 });
    };
    H.stage.addEventListener("pointerdown", e => {
      if (speakOn()) audioCtx();
      // taps on the HUD labels / editor must stay clicks, not a captured drag
      if (e.target.closest(".holo-hud, .holo-editor, .holo-sheet")) return;
      pts.set(e.pointerId, { x: e.clientX, y: e.clientY, pan: e.pointerType === "mouse" && (e.shiftKey || e.button === 2) });
      try { H.stage.setPointerCapture(e.pointerId); } catch (err) { /* not capturable: still track it */ }
      const now = performance.now();
      if (pts.size === 1 && now - (H.lastTap || 0) < 300) resetView();
      H.lastTap = now;
      regrab();
    });
    H.stage.addEventListener("pointermove", e => {
      if (!pts.has(e.pointerId)) return;
      const prev = pts.get(e.pointerId);
      pts.set(e.pointerId, { x: e.clientX, y: e.clientY, pan: prev && prev.pan });
      const w = H.stage.clientWidth || 1, h = H.stage.clientHeight || 1;
      if (H.pinch && pts.size === 2) {
        const p = pinch();
        H.view[2] = clamp(H.pinch.zoom * p.d / H.pinch.d, ZMIN, ZMAX);
        H.view[0] = clamp(H.pinch.px + (p.x - H.pinch.x) / w, -PAN, PAN);
        H.view[1] = clamp(H.pinch.py - (p.y - H.pinch.y) / h, -PAN, PAN);
        let da = p.ang - H.pinch.ang;                       // twist, unwrapped
        if (da > Math.PI) da -= 2 * Math.PI; else if (da < -Math.PI) da += 2 * Math.PI;
        H.rot = H.pinch.rot + da;
      } else if (H.pan) {
        H.view[0] = clamp(H.pan[2] + (e.clientX - H.pan[0]) / w, -PAN, PAN);
        H.view[1] = clamp(H.pan[3] - (e.clientY - H.pan[1]) / h, -PAN, PAN);
      } else if (H.drag) {
        H.dragOff = [clamp(H.drag[2] + (e.clientX - H.drag[0]) / w * 4.2, -TILT, TILT),
                     clamp(H.drag[3] + (e.clientY - H.drag[1]) / w * 4.2, -TILT, TILT)];
      }
    });
    const up = e => { pts.delete(e.pointerId); regrab(); };
    H.stage.addEventListener("pointerup", up); H.stage.addEventListener("pointercancel", up);
    H.stage.addEventListener("wheel", e => {
      e.preventDefault();
      if (e.shiftKey && !e.ctrlKey) {
        H.rot = (H.rot || 0) + (e.deltaY || e.deltaX) * 0.003;            // shift+wheel: twist
      } else if (e.ctrlKey || Math.abs(e.deltaY) >= Math.abs(e.deltaX)) {
        H.view[2] = clamp(H.view[2] * Math.exp(-e.deltaY * 0.0015), ZMIN, ZMAX);
      } else {
        H.view[0] = clamp(H.view[0] - e.deltaX / (H.stage.clientWidth || 1), -PAN, PAN);
      }
    }, { passive: false });
    H.stage.addEventListener("contextmenu", e => e.preventDefault());   // right-drag pans

    if (!initGL()) {
      const f = document.createElement("div"); f.className = "holo-fallback";
      if (H.urls.img) f.style.backgroundImage = `url(${H.urls.img})`;
      H.canvas.replaceWith(f);
    }
    hookAudio();
    backfillCaption();
    updateVoiceLabel();
  }

  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function resetView() { H.view = [0, 0, 1]; H.rot = 0; H.dragOff = [0, 0]; H.glitch = Math.max(H.glitch, 0.5); }

  function speakOn() { try { return localStorage.getItem(SPEAK_KEY) === "1"; } catch (e) { return false; } }

  // The voices that fit the avatar: Piper ones by name, then the streamed
  // Orpheus speakers (tara & co. are female, leo/dan/zac male).
  const FEMALE_RE = /amy|kristin|female|jenny|lessac|ljspeech|kathleen|alba|cori|tara|leah|jess|mia|zoe/i;
  const MALE_RE = /joe|ryan|alan|danny|john|bryce|norman|kusal|_male|leo|:dan|zac/i;
  function femaleVoices() {
    const cat = (typeof mediaCatalog !== "undefined" && mediaCatalog) || {};
    const all = (cat.voices || []).concat((cat.orpheus_voices || []).map(v => "orpheus:" + v));
    const g = (H.avatar && H.avatar.gender) || "female";
    const fits = all.filter(v => g === "male" ? MALE_RE.test(v) && !/female/i.test(v)
                                 : g === "female" ? FEMALE_RE.test(v) : true);
    return fits.length ? fits : all;
  }
  function voiceKey() { return H.avatar && H.avatar.id !== "default" ? VOICE_KEY + ":" + H.avatar.id : VOICE_KEY; }
  function brainList() {
    // chat models only (the dropdown also lists every folder under models/)
    const sel = document.getElementById("model-select");
    let all = sel ? [...sel.options].filter(o => /\((llama_cpp|ollama)\)/.test(o.textContent)).map(o => o.value) : [];
    if (!all.length && sel) all = [...sel.options].map(o => o.value);   // other label formats
    return HOLO_BRAINS.filter(m => all.includes(m)).concat(all.filter(m => !HOLO_BRAINS.includes(m)));
  }
  function holoModel() {
    const list = brainList();
    let m = null; try { m = localStorage.getItem(MODEL_KEY); } catch (e) {}
    if (m && list.includes(m)) return m;
    const curated = HOLO_BRAINS.find(x => list.includes(x));
    const sel = document.getElementById("model-select");
    return curated || (sel && sel.value) || list[0] || null;
  }
  function voiceName(v) { return v.replace(/^orpheus:/, "").replace(/^en_[A-Z]{2}-/, "").replace(/-(medium|high|low)$/, ""); }
  function opts(sel, items, cur) {
    const sig = JSON.stringify(items);
    if (sel._sig === sig) { if (cur != null && sel.value !== cur && document.activeElement !== sel) sel.value = cur; return; }
    if (document.activeElement === sel) return;      // the user is picking right now
    sel._sig = sig;
    sel.innerHTML = "";
    for (const [value, label] of items) {
      const o = document.createElement("option"); o.value = value; o.textContent = label; sel.appendChild(o);
    }
    if (cur != null) sel.value = cur;
  }
  // (Re)build the three pickers from what the PC has right now.
  function fillPickers() {
    if (!H.el) return;
    opts(H.el.querySelector("#holo-vsel"), femaleVoices().map(v => [v, voiceName(v) + (isStreamVoice(v) ? "  (Orpheus)" : "  (Piper)")]), holoVoice());
    opts(H.el.querySelector("#holo-msel"), brainList().map(m => [m, (HOLO_BRAINS.includes(m) ? "★ " : "") + m]), holoModel());
    const av = (H.avatars || builtins()).filter(a => a.status === "ready");
    opts(H.el.querySelector("#holo-asel"), av.map(a => [a.id, a.name]), H.avatar && H.avatar.id);
    opts(H.el.querySelector("#holo-psel"), PROMPT_MODES, promptMode());
  }

  function promptMode() {
    let m = null; try { m = localStorage.getItem(PROMPT_KEY); } catch (e) {}
    return PROMPT_MODES.some(x => x[0] === m) ? m : "both";
  }
  // An uploaded avatar always has at least a name; the built-in one only
  // once it has been given a persona.
  function hasPersona(a) {
    a = a || H.meta || H.avatar;
    return !!a && (!isBuiltin(a) || !!(a.persona || "").trim());
  }
  function updatePromptLabel() {
    const n = H.el && H.el.querySelector("#holo-pname"); if (!n) return;
    const m = promptMode();
    // a mode that needs a persona says so when this avatar has none yet
    n.textContent = "prompt: " + { both: "both", system: "artifex", persona: "persona" }[m] +
      (m !== "system" && !hasPersona() ? " (no persona)" : "");
  }

  function updateModelLabel() {
    const n = H.el && H.el.querySelector("#holo-model"); if (!n) return;
    const m = holoModel();
    n.textContent = "core: " + (m || "none").replace(/-abliterated$/, "").replace(/^qwen/, "qwen ");
    fillPickers();
  }

  // ── avatars ─────────────────────────────────────────────────
  function authed() { return typeof authHeaders === "function" && typeof store !== "undefined" && store.k; }

  async function loadAvatars() {
    if (!authed()) return;
    try {
      const r = await fetch("/v1/holo/avatars", { headers: authHeaders() });
      if (!r.ok) return;
      H.avatars = builtins().concat((await r.json()).data || []);
    } catch (e) { return; }
    let want = null; try { want = localStorage.getItem(AVATAR_KEY); } catch (e) {}
    const a = H.avatars.find(x => x.id === want && x.status === "ready");
    if (a && a.id !== H.avatar.id) chooseAvatar(a);
    updateAvatarLabel();
    // an upload still processing on the PC (tab left, page reloaded): keep watching it
    for (const busy of H.avatars.filter(x => x.status === "processing")) watchAvatar(busy);
  }

  async function fetchBlobUrl(path) {
    const r = await fetch(path, { headers: authHeaders() });
    if (!r.ok) throw new Error(path + " " + r.status);
    return URL.createObjectURL(await r.blob());
  }

  async function chooseAvatar(a) {
    if (H.choosing === a.id) return;
    H.choosing = a.id;
    H.meta = a;                                          // persona/name usable right away
    const seq = H.avatarSeq = (H.avatarSeq || 0) + 1;   // a later pick wins over a slow fetch
    let urls;
    if (isBuiltin(a)) urls = builtinUrls(a);
    else {
      try {
        const base = "/v1/holo/avatars/" + a.id + "/";
        urls = { img: await fetchBlobUrl(base + "avatar.jpg"), depth: await fetchBlobUrl(base + "depth.png"), blob: true };
      } catch (e) {
        if (seq === H.avatarSeq) { H.choosing = null; H.meta = H.avatar; diag("avatar: " + e.message); }
        return;
      }
    }
    if (seq !== H.avatarSeq) { if (urls.blob) { URL.revokeObjectURL(urls.img); URL.revokeObjectURL(urls.depth); } return; }
    H.choosing = null;
    const old = H.urls;
    H.avatar = a; H.urls = urls; H.meta = a;
    try { localStorage.setItem(AVATAR_KEY, a.id); } catch (e) {}
    const fb = H.el && H.el.querySelector(".holo-fallback");     // no WebGL: show the picture flat
    if (fb) fb.style.backgroundImage = urls.img ? "url(" + urls.img + ")" : "none";
    if (H.el) H.el.classList.toggle("orb", a.kind === "orb");
    H.glitch = 1;
    applyAvatarGL();
    // the old blobs go once the new textures are in (the GPU copy is made)
    if (old && old.blob) setTimeout(() => { URL.revokeObjectURL(old.img); URL.revokeObjectURL(old.depth); }, 4000);
    updateAvatarLabel(); updateVoiceLabel(); H.sp.warm = null; speechWarm();
  }


  function updateAvatarLabel() {
    const n = H.el && H.el.querySelector("#holo-aname"); if (!n) return;
    n.textContent = "avatar: " + (H.avatar ? H.avatar.name : "?");
    const inp = H.el.querySelector("#holo-input");
    const nm = H.avatar && H.avatar.name;
    if (inp) inp.placeholder = "Talk to " + (nm || "Ember") + "...";
    fillPickers(); updatePromptLabel();
    const ed = H.el.querySelector("#holo-edit");
    if (ed) ed.style.display = "";
  }

  async function uploadAvatar(file) {
    if (!authed()) { diag("set the API key first"); return; }
    diag("avatar: uploading");
    let m;
    try {
      const r = await fetch("/v1/holo/avatars?name=", { method: "PUT", headers: authHeaders(), body: file });
      let body = null; try { body = await r.json(); } catch (e) {}
      if (!r.ok) throw new Error((body && body.detail) || ("server " + r.status));
      m = body;
    } catch (e) { diag("upload failed: " + e.message); return; }
    watchAvatar(m);
  }

  // Poll a processing avatar until it is ready, then switch to it and open
  // the dot editor. The work runs on the PC, so this can pick up again later.
  async function watchAvatar(m) {
    H.watching = H.watching || new Set();
    if (H.watching.has(m.id)) return;
    H.watching.add(m.id);
    const STEPS = { queued: "queued", depth: "making depth map", face: "finding the face", persona: "meeting them" };
    for (let i = 0; i < 400 && m.status === "processing"; i++) {
      diag("avatar: " + (STEPS[m.step] || m.step) + "…");
      await new Promise(res => setTimeout(res, 1500));
      try { m = await (await fetch("/v1/holo/avatars/" + m.id, { headers: authHeaders() })).json(); } catch (e) {}
    }
    H.watching.delete(m.id);
    if (m.status !== "ready") { diag("avatar failed: " + (m.error || m.status)); return; }
    H.avatars = (H.avatars || builtins()).filter(a => a.id !== m.id).concat([m]);
    await chooseAvatar(m);
    diag("avatar ready (" + m.landmark_source + ") - check the dots");
    openEditor();
  }

  // Dots on the flat picture: both eyes and the two mouth corners. The eye
  // sizes stay the detector's; the eye line sets both eyes' tilt.
  function openEditor() {
    const a = H.avatar; if (!a) return;
    const builtin = isBuiltin(a), orb = a.kind === "orb";
    if (H.closeEditor) H.closeEditor();
    const ed = H.el.querySelector("#holo-editor");
    const L = a.landmarks || {
      eyes: [{ c: [a.w * 0.4, a.h * 0.38], dir: [1, 0], hw: a.w * 0.07, hh: a.w * 0.03 },
             { c: [a.w * 0.6, a.h * 0.38], dir: [1, 0], hw: a.w * 0.07, hh: a.w * 0.03 }],
      mouth: { c: [a.w * 0.5, a.h * 0.5], dir: [1, 0], hw: a.w * 0.06 } };
    const mo = L.mouth, md = mo.dir;
    const pts = {
      eyeL: L.eyes[0].c.slice(), eyeR: L.eyes[1].c.slice(),
      mouthL: [mo.c[0] - md[0] * mo.hw, mo.c[1] - md[1] * mo.hw],
      mouthR: [mo.c[0] + md[0] * mo.hw, mo.c[1] + md[1] * mo.hw],
    };
    ed.innerHTML = `
      <div class="holo-ed-hint">${orb ? "Ember has no face to mark: give it a name, a voice set and a persona." : "Drag the yellow dots onto the eyes and the pink ones onto the mouth corners."} ${
        orb ? "" : { mediapipe: "Placed by face detection.", vision: "Placed by the vision model; check them.", guess: "A rough guess: place them.", user: "Your placement." }[a.landmark_source] || ""}</div>
      <div class="holo-ed-pic"><img alt=""></div>
      <div class="holo-ed-row"><input id="holo-ed-name" maxlength="40" placeholder="Name">
        <select id="holo-ed-gender" aria-label="Voices"><option value="female">female voices</option><option value="male">male voices</option><option value="neutral">all voices</option></select></div>
      <div class="holo-ed-row"><textarea id="holo-ed-persona" maxlength="2500" placeholder="Who they are and how they talk (goes in the system prompt)"></textarea></div>
      <div class="holo-ed-hint" id="holo-ed-msg"></div>
      <div class="holo-ed-row"><button class="btn" id="holo-ed-save">Save</button><button class="btn" id="holo-ed-close">Close</button><button class="btn danger" id="holo-ed-del">${builtin ? "Reset" : "Delete"}</button></div>`;
    ed.querySelector("#holo-ed-name").value = a.name || "";
    ed.querySelector("#holo-ed-persona").value = a.persona || "";
    ed.querySelector("#holo-ed-gender").value = a.gender || "neutral";
    const pic = ed.querySelector(".holo-ed-pic"), img = pic.querySelector("img");
    if (orb) pic.style.display = "none"; else img.src = H.urls.img;
    const dots = {};
    const box = () => {     // where the contained image actually sits
      const r = pic.getBoundingClientRect(), s = Math.min(r.width / a.w, r.height / a.h);
      return { r, s, ox: (r.width - a.w * s) / 2, oy: (r.height - a.h * s) / 2 };
    };
    const place = () => {
      const b = box();
      for (const k in dots) { dots[k].style.left = (b.ox + pts[k][0] * b.s) + "px"; dots[k].style.top = (b.oy + pts[k][1] * b.s) + "px"; }
    };
    for (const [k, label] of orb ? [] : [["eyeL", "eye"], ["eyeR", "eye"], ["mouthL", "mouth"], ["mouthR", "mouth"]]) {
      const d = document.createElement("div");
      d.className = "holo-dot" + (k.startsWith("mouth") ? " m" : "") + (k === "mouthL" ? " lft" : k === "mouthR" ? " rgt" : "");
      d.innerHTML = "<span>" + label + "</span>";
      d.addEventListener("pointerdown", e => { e.stopPropagation(); try { d.setPointerCapture(e.pointerId); } catch (err) {} d._drag = true; });
      d.addEventListener("pointermove", e => {
        if (!d._drag) return;
        const b = box();
        pts[k] = [clamp((e.clientX - b.r.left - b.ox) / b.s, 0, a.w), clamp((e.clientY - b.r.top - b.oy) / b.s, 0, a.h)];
        place();
      });
      d.addEventListener("pointerup", () => { d._drag = false; });
      d.addEventListener("pointercancel", () => { d._drag = false; });
      pic.appendChild(d); dots[k] = d;
    }
    ed.onpointerdown = e => e.stopPropagation();
    ed.onwheel = e => e.stopPropagation();
    img.onload = place;
    // re-place on any size change: the keyboard, rotation, or opening while
    // the tab was hidden (a zero-size box put every dot at 0,0)
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(place) : null;
    if (ro) ro.observe(pic); else window.addEventListener("resize", place);
    ed.classList.remove("hidden");
    requestAnimationFrame(place);
    const close = () => {
      ed.classList.add("hidden"); ed.innerHTML = "";
      if (ro) ro.disconnect(); else window.removeEventListener("resize", place);
      H.closeEditor = null;
    };
    H.closeEditor = close;
    ed.querySelector("#holo-ed-close").onclick = close;
    const msg = t => { const n = ed.querySelector("#holo-ed-msg"); if (n) n.textContent = t; };
    let delArmed = false;
    ed.querySelector("#holo-ed-del").onclick = async e => {
      if (!delArmed) { delArmed = true; e.target.textContent = "Tap again"; return; }
      if (builtin) {         // back to the original name, persona and dots
        try { localStorage.removeItem(orb ? ORB_EDIT_KEY : DEFAULT_EDIT_KEY); } catch (err) {}
        const d = builtinById(a.id);
        H.avatars = (H.avatars || []).map(x => (x.id === a.id ? d : x));
        close(); H.choosing = null; chooseAvatar(d); diag("built-in avatar reset");
        return;
      }
      const r = await fetch("/v1/holo/avatars/" + a.id, { method: "DELETE", headers: authHeaders() }).catch(() => null);
      if (!r || !r.ok) { e.target.textContent = "Delete"; delArmed = false; msg("Delete failed" + (r ? ": " + r.status : "")); return; }
      H.avatars = (H.avatars || []).filter(x => x.id !== a.id);
      close(); chooseAvatar(builtins()[0]);
    };
    ed.querySelector("#holo-ed-save").onclick = async e => {
      const btn = e.currentTarget;
      if (btn.disabled) return;
      btn.disabled = true; btn.textContent = "Saving..."; msg("");
      const swap = pts.eyeL[0] > pts.eyeR[0];
      const [l, r] = swap ? [pts.eyeR, pts.eyeL] : [pts.eyeL, pts.eyeR];
      const [el, er] = swap ? [L.eyes[1], L.eyes[0]] : [L.eyes[0], L.eyes[1]];
      const eyeDir = [r[0] - l[0], r[1] - l[1]], mDir = [pts.mouthR[0] - pts.mouthL[0], pts.mouthR[1] - pts.mouthL[1]];
      const landmarks = {
        eyes: [Object.assign({}, el, { c: l, dir: eyeDir }), Object.assign({}, er, { c: r, dir: eyeDir })],
        mouth: { c: [(pts.mouthL[0] + pts.mouthR[0]) / 2, (pts.mouthL[1] + pts.mouthR[1]) / 2], dir: mDir,
                 hw: Math.hypot(mDir[0], mDir[1]) / 2 },
      };
      const body = { landmarks, name: ed.querySelector("#holo-ed-name").value, persona: ed.querySelector("#holo-ed-persona").value,
                     gender: ed.querySelector("#holo-ed-gender").value };
      try {
        if (builtin) {       // no server manifest: this phone keeps the edits
          if (orb) delete body.landmarks;
          try { localStorage.setItem(orb ? ORB_EDIT_KEY : DEFAULT_EDIT_KEY, JSON.stringify(body)); } catch (err) { throw new Error("couldn't store on this phone"); }
          const d = builtinById(a.id);
          H.avatar = H.meta = d; H.avatars = (H.avatars || []).map(x => (x.id === a.id ? d : x));
          applyAvatarGL(); updateAvatarLabel(); updateVoiceLabel();
          btn.textContent = "Saved ✓"; msg(orb ? "Saved on this phone." : "Saved on this phone. Watch the blink.");
          H.blinkT = 0;
          setTimeout(() => { if (H.closeEditor === close) close(); diag("saved " + d.name); }, 700);
          return;
        }
        const res = await fetch("/v1/holo/avatars/" + a.id, { method: "PATCH",
          headers: authHeaders({ "Content-Type": "application/json" }), body: JSON.stringify(body) });
        const m = await res.json();
        if (!res.ok) throw new Error(m.detail || res.status);
        H.avatar = m; H.avatars = (H.avatars || []).map(x => (x.id === m.id ? m : x));
        applyAvatarGL(); updateAvatarLabel(); updateVoiceLabel();
        btn.textContent = "Saved ✓"; msg("Saved. Watch the blink.");
        H.blinkT = 0;
        setTimeout(() => { if (H.closeEditor === close) close(); diag("saved " + m.name); }, 700);
      } catch (err) {
        btn.disabled = false; btn.textContent = "Save";
        msg("Save failed: " + err.message);
      }
    };
  }

  function isStreamVoice(v) { return !!v && v.startsWith("orpheus:"); }
  function holoVoice() {
    const fem = femaleVoices();
    let v = null; try { v = localStorage.getItem(voiceKey()); } catch (e) {}
    if (v && fem.includes(v)) return v;
    // defaults: Piper Jenny / Joe (instant, no GPU); Orpheus voices are opt-in
    const pref = H.avatar && H.avatar.voice_pref;
    const liked = pref && fem.find(x => pref.test(x) && !isStreamVoice(x));
    if (liked) return liked;
    return fem.find(x => /jenny|joe/i.test(x)) || fem.find(x => !isStreamVoice(x)) || fem[0] || null;
  }
  function updateVoiceLabel() {
    const n = H.el && H.el.querySelector("#holo-vname"); if (!n) return;
    const v = holoVoice();
    n.textContent = v ? "voice: " + voiceName(v) : "voice: none";
    fillPickers();
  }

  function compile(gl, type, src) {
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) { diag("shader: " + gl.getShaderInfoLog(s)); return null; }
    return s;
  }

  function initGL() {
    const gl = H.canvas.getContext("webgl", { premultipliedAlpha: true, alpha: true, antialias: false });
    if (!gl) { diag("no webgl"); return false; }
    // iOS drops WebGL contexts (backgrounding, memory pressure): rebuild on
    // restore instead of staying blank.
    H.canvas.addEventListener("webglcontextlost", e => { e.preventDefault(); H.gl = null; H.ready = 0; diag("gl lost"); });
    H.canvas.addEventListener("webglcontextrestored", () => { setupGL(gl); });
    return setupGL(gl);
  }

  function setupGL(gl) {
    const vs = compile(gl, gl.VERTEX_SHADER, VS), fs = compile(gl, gl.FRAGMENT_SHADER, FS);
    if (!vs || !fs) return false;
    const p = gl.createProgram(); gl.attachShader(p, vs); gl.attachShader(p, fs);
    gl.bindAttribLocation(p, 0, "aPos"); gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) { diag("link: " + gl.getProgramInfoLog(p)); return false; }
    // the orb's own program, on the same quad (attribute 0 in both)
    H.progOrb = null; H.uo = {};
    const fo = compile(gl, gl.FRAGMENT_SHADER, FS_ORB);
    if (fo) {
      const po = gl.createProgram(); gl.attachShader(po, vs); gl.attachShader(po, fo);
      gl.bindAttribLocation(po, 0, "aPos"); gl.linkProgram(po);
      if (gl.getProgramParameter(po, gl.LINK_STATUS)) {
        H.progOrb = po;
        for (const n of ["uRes", "uTilt", "uView", "uRot", "uT", "uPhase", "uLevel", "uGlitch", "uThink", "uSpeak"]) H.uo[n] = gl.getUniformLocation(po, n);
      } else diag("orb link: " + gl.getProgramInfoLog(po));
    }
    gl.useProgram(p);
    const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(p, "aPos"); gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    for (const n of ["IMG", "uImg", "uDepth", "uRes", "uTilt", "uView", "uRot", "uT", "uMouth", "uBlink", "uGlitch", "uThink", "uSpeak",
                     "uMC", "uMD", "uMW", "uE0", "uE1", "uE0d", "uE1d"]) H.u[n] = gl.getUniformLocation(p, n);
    gl.uniform1i(H.u.uImg, 0); gl.uniform1i(H.u.uDepth, 1);
    H.gl = gl; H.prog = p; H.tex = [null, null]; H.texFor = null;
    applyAvatarGL();
    return true;
  }

  // Face uniforms + both textures for H.avatar (H.urls: its image + depth).
  function applyAvatarGL() {
    const gl = H.gl; if (!gl || !H.urls) return;
    if (H.urls.orb) { H.ready = 2; return; }       // drawn by its shader: nothing to load
    const a = H.avatar, L = a.landmarks || DEFAULT_AVATAR.landmarks;
    const norm = v => { const l = Math.hypot(v[0], v[1]) || 1; return [v[0] / l, v[1] / l]; };
    const [e0, e1] = L.eyes, mo = L.mouth;
    gl.useProgram(H.prog);
    gl.uniform2f(H.u.IMG, a.w, a.h);
    gl.uniform2fv(H.u.uMC, mo.c); gl.uniform2fv(H.u.uMD, norm(mo.dir)); gl.uniform1f(H.u.uMW, mo.hw);
    gl.uniform4f(H.u.uE0, e0.c[0], e0.c[1], e0.hw, e0.hh); gl.uniform2fv(H.u.uE0d, norm(e0.dir));
    gl.uniform4f(H.u.uE1, e1.c[0], e1.c[1], e1.hw, e1.hh); gl.uniform2fv(H.u.uE1d, norm(e1.dir));
    if (H.texFor !== H.urls) {
      H.texFor = H.urls;
      if (!H.tex[0]) H.ready = 0;                  // coming from the orb: nothing to show until these load
      loadPair(H.urls);
    }
  }

  // Both textures of an avatar are decoded first and swapped in together:
  // the old avatar stays on screen meanwhile, and a stale load (switching
  // fast) is dropped instead of landing on the new avatar's depth map.
  function loadPair(urls) {
    const gl = H.gl, imgs = [new Image(), new Image()];
    let left = 2, failed = false;
    const done = () => {
      if (--left || failed || gl !== H.gl || gl.isContextLost() || H.texFor !== urls) return;
      imgs.forEach((img, unit) => {
        gl.activeTexture(gl.TEXTURE0 + unit);
        if (H.tex[unit]) gl.deleteTexture(H.tex[unit]);
        const t = H.tex[unit] = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      });
      H.ready = 2;
    };
    imgs.forEach((img, i) => {
      img.onload = done;
      img.onerror = () => { failed = true; diag("texture failed to load"); };
      img.src = i ? urls.depth : urls.img;
    });
  }

  // Piper playback drives the mouth: the spoken text is spread across the
  // clip's duration, so syllables roughly line up with the audio.
  function hookAudio() {
    if (typeof ttsAudio === "undefined") return;
    const end = () => { if (!H.sp.owns) H.say = null; };
    ttsAudio.addEventListener("pause", end);
    ttsAudio.addEventListener("ended", end);
  }

  // ── live speech ─────────────────────────────────────────────
  // The streaming-voice trick: don't wait for the whole reply. Each clause goes to
  // Piper (~0.1 s) or Orpheus the moment it is complete, the next one is made
  // while this one plays, and the first piece may be short so she starts at
  // once. One voice per turn (S.voice, fixed at turnStart), so Piper clips and
  // streamed Orpheus audio never overlap.
  function speechOn() { return H.visible && speakOn() && typeof authHeaders === "function" && typeof ttsAudio !== "undefined"; }

  // The first Piper call per voice loads it (~1.5 s); pay that up front.
  // (Tara: this also starts the Orpheus server, ~2 s, and the SNAC decoder.)
  function speechWarm() {
    const v = holoVoice();
    if (!speechOn() || !v || H.sp.warm === v || !(typeof store !== "undefined" && store.k)) return;
    H.sp.warm = v;
    const stream = isStreamVoice(v);
    // Orpheus: ?warm=1 only starts the server + decoder (a spoken warm-up
    // would cut off a line she is saying)
    fetch(stream ? "/v1/voice/tts/stream?warm=1" : "/v1/voice/tts", {
      method: "POST", headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ text: "Hi.", voice: stream ? v.slice(8) : v }),
    }).then(r => { if (!r.ok) { diag("voice warm-up: " + r.status); H.sp.warm = null; } return r.arrayBuffer(); })
      .catch(() => { H.sp.warm = null; });
  }

  // ── Web Audio for streamed voices ──────────────────────────
  // Tara's PCM arrives in ~0.3-0.7 s chunks; each is scheduled right after
  // the previous one on the AudioContext clock, so playback starts with the
  // first chunk. The analyser feeds her mouth the real loudness.
  const TARA_SPEED = 1.08;   // 8% quicker, a touch higher (tuned by ear)
  const LEAD = 0.15;         // cushion when (re)starting, absorbs chunk jitter

  // Must first run inside a tap (iOS keeps a context suspended otherwise).
  // iOS also parks a running context as "interrupted" (a call, the lock
  // screen); any later tap or return to the page resumes it.
  function audioCtx() {
    if (!H.ac) {
      const C = window.AudioContext || window.webkitAudioContext;
      if (!C) return null;
      try { if (navigator.audioSession) navigator.audioSession.type = "playback"; } catch (e) {}  // play through the silent switch
      H.ac = new C();
      H.an = H.ac.createAnalyser(); H.an.fftSize = 512; H.an.connect(H.ac.destination);
      H.anBuf = new Float32Array(H.an.fftSize);
    }
    if (H.ac.state !== "running") H.ac.resume().catch(() => {});
    return H.ac;
  }

  // She is audibly talking (Piper clip, streamed Tara audio, or clips queued).
  function speakingNow() {
    const S = H.sp;
    return streamSpeaking() || S.playing || S.q.length > 0;
  }

  function streamSpeaking() { return !!H.ac && H.ac.state === "running" && H.sp.head > H.ac.currentTime; }

  function schedulePcm(f, rate) {
    const S = H.sp, ac = H.ac;
    const buf = ac.createBuffer(1, f.length, rate);
    buf.getChannelData(0).set(f);
    const src = ac.createBufferSource();
    src.buffer = buf; src.connect(H.an);
    if (S.head < ac.currentTime + 0.02) S.head = ac.currentTime + LEAD;
    src.start(S.head);
    S.head += buf.duration;
    S.srcs.push(src);
    src.onended = () => { const i = S.srcs.indexOf(src); if (i >= 0) S.srcs.splice(i, 1); };
  }

  // Say this clause with Piper instead (Tara off, failed or unreachable);
  // the rest of the turn follows.
  function toPiper(text, gen, why) {
    const S = H.sp;
    if (gen !== S.gen) return;
    if (!S.noStream && why) diag("tara off: " + why);
    S.noStream = true;
    speechEnqueue(text);
  }

  async function streamClause(text, voice, gen) {
    const S = H.sp;
    if (gen !== S.gen) return;
    if (S.noStream || !audioCtx()) { toPiper(text, gen); return; }
    let r;
    try {
      r = await fetch("/v1/voice/tts/stream", {
        method: "POST", headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({ text, voice }), signal: S.abort.signal,
      });
    } catch (e) {
      if (e.name !== "AbortError") toPiper(text, gen, "network");
      return;
    }
    if (gen !== S.gen) return;
    if (!r.ok || !r.body) {
      // e.g. the chosen brain needs both GPUs: say it with Piper instead
      const why = (await r.text().catch(() => "")).slice(0, 70) || String(r.status);
      toPiper(text, gen, why);
      return;
    }
    const rate = (+r.headers.get("X-Sample-Rate") || 24000) * TARA_SPEED;
    const reader = r.body.getReader();
    let odd = null, got = 0;
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done || gen !== S.gen) break;
        let b = value;
        if (odd !== null) { const m = new Uint8Array(b.length + 1); m[0] = odd; m.set(b, 1); b = m; odd = null; }
        const n = b.length >> 1;
        if (b.length & 1) odd = b[b.length - 1];
        if (!n) continue;
        const dv = new DataView(b.buffer, b.byteOffset, n * 2), f = new Float32Array(n);
        for (let i = 0; i < n; i++) f[i] = dv.getInt16(i * 2, true) / 32768;
        schedulePcm(f, rate);
        got += n;
      }
    } catch (e) { if (e.name === "AbortError") return; }
    // a 200 that carried no audio (the server failed after answering)
    if (!got && gen === S.gen) toPiper(text, gen, "no audio");
  }

  // Opening the tab loads her brain now, so the first message isn't stuck
  // behind a model swap (a cold load is ~10-37 s) - unless it is already the
  // loaded model, a chat is streaming, or an agent run owns the engine.
  async function brainWarm() {
    const m = holoModel();
    if (!m || H.warming || !(typeof store !== "undefined" && store.k)) return;
    if (typeof isStreaming !== "undefined" && isStreaming) return;
    if (typeof agentRunId !== "undefined" && agentRunId && !(typeof agentDone !== "undefined" && agentDone)) return;
    H.warming = true;
    try {
      const e = await (await fetch("/v1/engine", { headers: authHeaders() })).json();
      if (e.busy || (e.loaded && e.model === m)) return;
      await fetch("/v1/chat/completions", { method: "POST", headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({ model: m, stream: false, max_tokens: 1, messages: [{ role: "user", content: "hi" }] }) });
    } catch (e) { /* best effort */ } finally { H.warming = false; }
  }

  // Silence her and drop everything queued. Only touches the shared <audio>
  // element when Holo is the one playing it (chat read-aloud stays alone).
  function speechStop() {
    const S = H.sp;
    S.gen++;
    for (const it of S.q) if (it.p) it.p.then(u => { if (u) URL.revokeObjectURL(u); });
    S.q = []; S.buf = ""; S.n = 0; S.inCode = false; S.playing = false; S.skip = false;
    if (S.cur) { S.cur(); S.cur = null; }
    if (S.owns && typeof ttsAudio !== "undefined") {
      S.owns = false;
      if (!ttsAudio.paused) ttsAudio.pause();
      H.say = null;
    }
    S.abort.abort(); S.abort = new AbortController(); S.chain = Promise.resolve();
    S.srcs.splice(0).forEach(s => { try { s.stop(); } catch (e) {} });
    S.head = 0;
  }

  // Where the next speakable piece ends in buf, or -1 to wait for more text.
  const ABBREV = /(?:^|\s)(?:e\.g|i\.e|etc|vs|dr|mr|mrs|ms|st|no|jr|sr|approx|fig)\.$/i;
  function clauseEnd(buf, first, done) {
    const re = /[.!?…]+["')\]]*(?=\s)|[,;:—–](?=\s)|\n/g;
    let m;
    while ((m = re.exec(buf))) {
      const end = m.index + m[0].length;
      const head = buf.slice(0, end).trim();
      const strong = /[.!?…\n]/.test(m[0]);
      if (strong && m[0] === "." && ABBREV.test(head)) continue;     // "e.g." is not a sentence
      if (head.length >= (strong ? 6 : first ? 10 : 28)) return end;  // no tiny "1." clips
    }
    if (buf.length > 180) { const sp = buf.lastIndexOf(" ", 160); if (sp > 40) return sp; }
    return done ? buf.length : -1;
  }

  function cleanSpeech(piece) {
    const S = H.sp;
    let out = "";
    piece.split("```").forEach((p, i) => { if (i) S.inCode = !S.inCode; if (!S.inCode) out += p; });
    return out.replace(/\[([^\]]+)\]\([^)]*\)/g, "$1").replace(/https?:\/\/\S+/g, " link ")
              .replace(/[*_`#>|~]/g, "").replace(/\s+/g, " ").trim();
  }

  function speechFeed(text, done) {
    const S = H.sp;
    S.buf += text;
    if (S.skip) {
      // after a reattach: the sentence in progress was partly said already
      const m = /[.!?…]["')\]]*\s|\n/.exec(S.buf);
      if (!m && !done && S.buf.length < 400) return;
      S.buf = m ? S.buf.slice(m.index + m[0].length) : "";
      S.skip = false;
    }
    let end;
    while (S.buf && (end = clauseEnd(S.buf, S.n === 0, done)) > 0) {
      const piece = cleanSpeech(S.buf.slice(0, end));
      S.buf = S.buf.slice(end);
      if (/[a-z0-9]/i.test(piece)) { S.n++; speechEnqueue(piece); }
    }
  }

  function piperFallback() {
    const fem = femaleVoices().filter(v => !isStreamVoice(v));
    return fem.find(x => /jenny|joe/i.test(x)) || fem[0] || null;
  }

  function speechEnqueue(text) {
    const S = H.sp, gen = S.gen;
    let v = S.voice || holoVoice();
    if (isStreamVoice(v) && S.noStream) v = piperFallback();
    if (isStreamVoice(v)) {
      // One Orpheus slot: clause N+1 starts generating when N's stream ends
      // (it is faster than realtime, so the timeline stays ahead of playback).
      S.chain = S.chain.then(() => streamClause(text, v.slice(8), gen)).catch(() => {});
      return;
    }
    S.q.push({ text, voice: v, p: null });
    prefetch();
    if (!S.playing) speechNext(gen);
  }

  // Piper clips are made two ahead of playback, not all at once: a long reply
  // no longer queues 30 requests on the PC, and Stop cancels what is pending.
  function piperFetch(it) {
    it.p = fetch("/v1/voice/tts", {
      method: "POST", headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ text: it.text, voice: it.voice }), signal: H.sp.abort.signal,
    }).then(r => (r.ok ? r.blob() : null)).then(b => (b ? URL.createObjectURL(b) : null)).catch(() => null);
  }
  function prefetch() {
    for (const it of H.sp.q.slice(0, 2)) if (!it.p) piperFetch(it);
  }

  async function speechNext(gen) {
    const S = H.sp;
    if (gen !== S.gen) return;
    S.playing = true;
    if (streamSpeaking()) { setTimeout(() => speechNext(gen), 120); return; }   // let Tara finish first
    const it = S.q.shift();
    if (!it) { S.playing = false; return; }
    if (!it.p) piperFetch(it);
    prefetch();
    const url = await it.p;
    if (gen !== S.gen) { if (url) URL.revokeObjectURL(url); return; }
    if (!url) { speechNext(gen); return; }
    let fired = false;
    const off = () => {
      ttsAudio.removeEventListener("ended", next); ttsAudio.removeEventListener("error", next);
      ttsAudio.removeEventListener("pause", paused);
    };
    const next = () => {
      if (fired) return; fired = true; S.cur = null; off();
      URL.revokeObjectURL(url);
      S.owns = false;
      speechNext(gen);
    };
    // Paused by someone else (a chat speaker tap, an iOS call or Siri):
    // that is a hand-over, not a gap - stop cleanly instead of hanging.
    const paused = () => { if (!fired && !ttsAudio.ended) { fired = true; S.cur = null; off(); URL.revokeObjectURL(url); S.owns = false; speechStop(); } };
    S.cur = () => { fired = true; off(); URL.revokeObjectURL(url); };
    ttsAudio.addEventListener("ended", next); ttsAudio.addEventListener("error", next);
    ttsAudio.src = url;
    S.owns = true;
    H.say = it.text;
    try {
      await ttsAudio.play();
      if (!fired) ttsAudio.addEventListener("pause", paused);
    } catch (e) { next(); }
  }

  function setState(s) {
    if (s === H.state) return;
    if (s === "idle" || H.state === "idle") H.glitch = Math.max(H.glitch, 0.55);
    H.state = s;
    const n = document.getElementById("holo-state"); if (n) n.textContent = { idle: "idle", think: "processing", speak: "speaking" }[s];
    if (H.stage) H.stage.classList.toggle("think", s === "think");
  }

  // Opening the tab with nothing said yet shows the last reply from history.
  function backfillCaption() {
    if (H.caption || (typeof isStreaming !== "undefined" && isStreaming)) return;
    if (typeof conversation !== "undefined") {
      for (let i = conversation.length - 1; i >= 0; i--) {
        const m = conversation[i];
        if (m && m.role === "assistant" && typeof m.content === "string") { H.caption = m.content; break; }
      }
    }
    refreshCaption();
  }

  function refreshCaption() {
    const c = document.getElementById("holo-caption"); if (!c) return;
    // plain text, newest at the bottom: markdown symbols are noise here
    const t = H.caption.replace(/```[\s\S]*?(```|$)/g, " [code] ").replace(/[*_`#>]/g, "").replace(/\n{3,}/g, "\n\n");
    c.textContent = t.length > 600 ? "…" + t.slice(-600) : t;
    c.scrollTop = c.scrollHeight;
  }

  // On-screen status for phone debugging (no console on the phone).
  function diag(msg) {
    H.diagHold = performance.now() + 5000;
    const n = document.getElementById("holo-diag");
    if (n) n.textContent = String(msg).slice(0, 160);
    else if (H.el) H.el.textContent = "holo: " + msg;
  }

  function frame(now) {
    H.raf = 0;
    if (!H.canvas) return;
    if (!H.visible || document.hidden) return;
    H.raf = requestAnimationFrame(frame);
    try { step(now); } catch (e) { diag("frame: " + e.message); }
    H.frames = (H.frames || 0) + 1;
    if (now - (H.lblAt || 0) > 2000) {      // catalogs may arrive after the tab opened
      H.lblAt = now;
      // audio still queued but the context got parked (iOS interruption,
      // browser power saving): wake it so the rest is not stuck
      if (H.ac && H.ac.state !== "running" && H.sp.srcs.length && speakOn()) H.ac.resume().catch(() => {});
      const v = H.el && H.el.querySelector("#holo-vname"), mm = H.el && H.el.querySelector("#holo-model");
      if (v && /none/.test(v.textContent)) updateVoiceLabel();
      if (mm && /none/.test(mm.textContent)) updateModelLabel();
    }
    if (now - (H.diagAt || 0) > 1000 && now > (H.diagHold || 0)) {
      H.diagAt = now;
      const keep = H.diagHold;
      if (!/holodebug/.test(location.search)) { const n = document.getElementById("holo-diag"); if (n) n.textContent = H.gl ? "" : "no 3D (WebGL off)"; H.diagHold = keep; }
      else diag((H.gl ? (H.ready >= 2 ? "gl ok " : "gl tex " + H.ready + "/2 ") : "gl none ") + H.canvas.width + "x" + H.canvas.height + " f" + H.frames); H.diagHold = keep;
    }
  }

  function step(now) {
    const t = (now - H.t0) / 1000;
    const dt = Math.min(0.05, (now - (H.last || now)) / 1000); H.last = now;

    // chat state (read from app.html's globals; no hook needed)
    const streaming = typeof isStreaming !== "undefined" && isStreaming;
    const btn = document.getElementById("holo-send");
    if (btn) {
      const typed = H.inputEl && H.inputEl.value.trim();
      const t = streaming || (speakingNow() && !typed) ? "Stop" : "Send";
      if (btn.textContent !== t) btn.textContent = t;
    }

    // mouth target: Piper audio first, else the streamed text queue
    let target = 0;
    if (streamSpeaking()) {
      // streamed voice: the mouth follows the actual loudness
      H.an.getFloatTimeDomainData(H.anBuf);
      let e = 0; for (let i = 0; i < H.anBuf.length; i++) e += H.anBuf[i] * H.anBuf[i];
      target = Math.min(1, Math.sqrt(e / H.anBuf.length) * 7);
      setState("speak");
    } else if (H.say && !ttsAudio.paused && ttsAudio.duration > 0.3) {
      const k = Math.min(H.say.length - 1, Math.floor(ttsAudio.currentTime / ttsAudio.duration * H.say.length));
      const o = charOpen(H.say[k] || " ");
      target = o < 0 ? 0 : o;
      setState("speak");
    } else if (H.queue.length || H.pauseT > 0) {
      H.pauseT -= dt;
      H.qAcc += dt * 15;                         // ~15 chars/s, a talking pace
      if (H.queue.length > 60) H.queue = H.queue.slice(-60);   // never lag far behind the text
      while (H.qAcc >= 1 && H.queue.length) {
        H.qAcc -= 1;
        const o = charOpen(H.queue[0]); H.queue = H.queue.slice(1);
        if (o < 0) { H.pauseT = 0.12; H.mouthT = 0; } else H.mouthT = o;
      }
      target = H.pauseT > 0 ? 0 : H.mouthT;
      setState("speak");
    } else {
      setState(streaming ? (H.sawText && !H.quiet ? "speak" : "think") : "idle");
    }
    target *= 0.85 + 0.15 * Math.sin(t * 31);
    H.mouth += (target - H.mouth) * Math.min(1, dt * 22);

    // blinks (sometimes a double blink)
    if (H.blinkT < 0 && t > H.nextBlink) { H.blinkT = 0; }
    if (H.blinkT >= 0) {
      H.blinkT += dt;
      const x = H.blinkT / 0.17;
      H.blink = x < 0.4 ? x / 0.4 : Math.max(0, 1 - (x - 0.4) / 0.6);
      if (x >= 1) { H.blinkT = -1; H.blink = 0; H.nextBlink = t + (Math.random() < 0.18 ? 0.22 : 2.2 + Math.random() * 4); }
    }
    // glitches
    if (t > H.nextGlitch) { H.glitch = 0.4 + Math.random() * 0.6; H.nextGlitch = t + 4 + Math.random() * 9; }
    H.glitch = Math.max(0, H.glitch - dt * 3.5);

    // head: idle drift plus wherever you dragged her
    const think = H.state === "think" ? 1 : 0;
    H.thinkS = (H.thinkS || 0) + (think - (H.thinkS || 0)) * Math.min(1, dt * 3);
    H.speakS = (H.speakS || 0) + ((H.state === "speak" ? 1 : 0) - (H.speakS || 0)) * Math.min(1, dt * 4);
    const idle = [Math.sin(t * 0.37) * 0.55 + Math.sin(t * 0.13) * 0.3, Math.sin(t * 0.23) * 0.3 - H.thinkS * 0.35];
    H.tilt = [idle[0] + H.dragOff[0], idle[1] + H.dragOff[1]];

    if (Math.random() < dt * 2) { const s = document.getElementById("holo-sync"); if (s) s.textContent = (96 + Math.random() * 3.9).toFixed(1); }

    // the orb's clock: turns faster while it speaks or thinks
    H.phase = (H.phase || 0) + dt * (1 + 2.2 * H.speakS + 1.1 * H.thinkS);
    const gl = H.gl; if (!gl || H.ready < 2) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.round(H.canvas.clientWidth * dpr), h = Math.round(H.canvas.clientHeight * dpr);
    if (!w || !h) return;
    if (H.canvas.width !== w || H.canvas.height !== h) { H.canvas.width = w; H.canvas.height = h; }
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    if (H.urls && H.urls.orb && H.progOrb) {
      const u = H.uo;
      gl.useProgram(H.progOrb);
      gl.uniform2f(u.uRes, w, h); gl.uniform2fv(u.uTilt, H.tilt); gl.uniform3fv(u.uView, H.view); gl.uniform1f(u.uRot, H.rot || 0);
      gl.uniform1f(u.uT, t); gl.uniform1f(u.uPhase, H.phase); gl.uniform1f(u.uLevel, Math.max(0, H.mouth));
      gl.uniform1f(u.uGlitch, H.glitch); gl.uniform1f(u.uThink, H.thinkS); gl.uniform1f(u.uSpeak, H.speakS);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      return;
    }
    gl.useProgram(H.prog);
    gl.uniform2f(H.u.uRes, w, h); gl.uniform2fv(H.u.uTilt, H.tilt); gl.uniform3fv(H.u.uView, H.view); gl.uniform1f(H.u.uRot, H.rot || 0);
    gl.uniform1f(H.u.uT, t); gl.uniform1f(H.u.uMouth, Math.max(0, H.mouth));
    gl.uniform1f(H.u.uBlink, H.blink); gl.uniform1f(H.u.uGlitch, H.glitch);
    gl.uniform1f(H.u.uThink, H.thinkS); gl.uniform1f(H.u.uSpeak, H.speakS);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  // The prompt for one turn, from choices captured when it was sent.
  function buildPrompt(build, a, mode) {
    if (!a || mode === "system" || !hasPersona(a)) return build();
    const p = (a.persona || "").trim();
    if (mode === "persona") return build(/^you are\b/i.test(p) ? p : "You are " + a.name + "." + (p ? " " + p : ""));
    return build() + "\n\nYour name is " + a.name + "." + (p ? " " + p : "");
  }

  window.Holo = {
    // showTab() → start/stop the render loop with the tab
    visible(on) {
      H.visible = on;
      if (on) {
        loadAvatars(); updateAvatarLabel(); updateVoiceLabel(); updateModelLabel(); backfillCaption();
        speechWarm(); brainWarm();
        if (H.ac && speakOn()) audioCtx();
        if (!H.raf) { H.last = 0; H.raf = requestAnimationFrame(frame); }
      }
    },
    // When a message is sent: the Holo tab's model + prompt choices for that
    // turn, or null when it is an ordinary chat turn (tab not open).
    capture() {
      if (!H.visible) return null;
      const model = holoModel();
      if (!model) return null;
      const a = H.meta || H.avatar, mode = promptMode();
      return { model, system: build => buildPrompt(build, a, mode) };
    },
    // a new assistant turn begins (also a reattach): fresh caption, and
    // whatever she was still saying stops. The voice is fixed for the turn.
    turnStart() {
      H.caption = ""; H.sawText = false; H.queue = ""; refreshCaption();
      const S = H.sp;
      if (S.live || S.playing || S.q.length || S.srcs.length || S.owns) speechStop();
      S.noStream = false;
      S.live = S.liveTurn = speechOn();
      S.voice = S.live ? holoVoice() : null;
    },
    // each streamed answer chunk; `quiet` = it will be read aloud after, so
    // don't lip-flap now (she'd say it twice)
    feed(text, quiet) {
      H.sawText = true;
      H.caption += text; if (H.visible) refreshCaption();
      if (H.sp.liveTurn) {
        H.quiet = true;
        if (H.sp.live) speechFeed(text, false);
        return;
      }
      H.quiet = !!quiet;
      if (!quiet) H.queue += text;
    },
    // A reattach after a dropped connection: the reply so far, all at once.
    // The caption catches up; speech picks up at the next sentence (the one
    // in progress was partly said before the drop).
    resume(text) {
      H.caption = text || ""; refreshCaption();
      if (H.sp.live) { H.sp.buf = ""; H.sp.skip = true; }
    },
    // The reply ended. A normal end speaks the tail that never reached a
    // boundary; a dropped turn waits for its reattach; a stopped one is silent.
    turnEnd(how) {
      const S = H.sp;
      if (how && how.stopped) { speechStop(); return; }
      if (how && how.dropped) return;
      if (S.live) speechFeed("", true);
    },
    // a finished reply loaded from the server (it completed while away)
    historyChanged() { H.caption = ""; backfillCaption(); },
    // speakBubble() is about to play this text through ttsAudio
    say(text) { H.say = text || null; H.queue = ""; },
    // true for a turn Holo reads aloud as it streams (even if muted part way),
    // so the end-of-turn auto-read must not say it all again
    speaksLive() { return !!H.sp.liveTurn; },
    // Stop / a chat speaker tap / New chat: silence her (and, for a new or
    // opened conversation, drop the old caption)
    hush(clear) {
      speechStop();
      H.sp.live = false;
      if (clear) { H.caption = ""; refreshCaption(); }
    },
    // the Piper voice to read a whole reply with (auto-read while the tab is
    // open); Orpheus voices can't go through that endpoint
    voice() {
      if (!H.visible) return null;
      const v = holoVoice();
      return isStreamVoice(v) ? piperFallback() : v;
    },
    // Her own fast brain while the tab is open, if the PC lists it
    model() {
      if (!H.visible) return null;
      return holoModel();
    },
    // Debug from the console: Holo._h.blink = 1 etc.
    _h: H,
  };

  const el = document.getElementById("holo-view");
  if (el) { try { mount(el); } catch (e) { diag("mount: " + e.message); } }
  // Coming back to the app (or a bfcache restore) restarts the render loop.
  const wake = () => {
    if (document.hidden || !H.visible) return;
    if (H.ac && H.ac.state !== "running") H.ac.resume().catch(() => {});
    if (!H.raf) { H.last = 0; H.raf = requestAnimationFrame(frame); }
  };
  document.addEventListener("visibilitychange", wake);
  window.addEventListener("pageshow", wake);
})();
