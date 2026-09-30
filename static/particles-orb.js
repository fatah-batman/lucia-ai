/**
 * particles-orb.js - Port of the Particles Orb AI assistant visualization
 * Pure vanilla JavaScript implementation using 2D Canvas.
 */

export const PARTICLE_COUNT = 720;
export const TWO_PI = Math.PI * 2;
export const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));
export const TIME_OFFSET = 1.7;
export const TONE_BUCKETS = 6;
export const ALPHA_BUCKETS = 10;
export const BUCKETS = TONE_BUCKETS * ALPHA_BUCKETS;
export const ANGLE_X = 0.32;

export const ERROR_COLOR_FROM = '#fb7185';
export const ERROR_COLOR_TO = '#f43f5e';

export const hexToRgb = (hex) => {
  const clean = hex.replace('#', '');
  const full =
    clean.length === 3
      ? clean
          .split('')
          .map((c) => c + c)
          .join('')
      : clean;
  const n = Number.parseInt(full, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
};

const ERROR_FROM_RGB = hexToRgb(ERROR_COLOR_FROM);
const ERROR_TO_RGB = hexToRgb(ERROR_COLOR_TO);

const toLinear = (c) => {
  const v = c / 255;
  return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
};

const toSrgb = (v) => {
  const c = v <= 0.0031308 ? v * 12.92 : 1.055 * v ** (1 / 2.4) - 0.055;
  return Math.min(255, Math.max(0, c * 255));
};

export const mixRgb = (a, b, t) => [
  toSrgb(toLinear(a[0]) + (toLinear(b[0]) - toLinear(a[0])) * t),
  toSrgb(toLinear(a[1]) + (toLinear(b[1]) - toLinear(a[1])) * t),
  toSrgb(toLinear(a[2]) + (toLinear(b[2]) - toLinear(a[2])) * t),
];

export const rgba = ([r, g, b], alpha) =>
  `rgba(${Math.round(r)},${Math.round(g)},${Math.round(b)},${Math.min(1, Math.max(0, alpha)).toFixed(3)})`;

export const clamp01 = (value) => Math.min(1, Math.max(0, value));

export const approach = (current, target, rate, dt) =>
  current + (target - current) * (1 - Math.exp(-rate * dt));

export const wave = (x) => 0.5 - 0.5 * Math.cos(x);

export const stateEnergy = (state, t) => {
  switch (state) {
    case 'listening':
      return 0.4 + 0.32 * wave(t * 17) + 0.18 * wave(t * 8.2 + 3);
    case 'speaking':
      return 0.3 + 0.24 * wave(t * 12.4) + 0.16 * wave(t * 6 + 1.2);
    case 'thinking':
      return 0.24 + 0.2 * wave(t * 4.8);
    case 'connecting':
      return 0.12 + 0.1 * wave(t * 3.2);
    case 'error':
      return 0.2;
    default:
      return 0;
  }
};

export const ENTER_RATE = 14;
export const SETTLE_RATE = 5;
export const ERROR_RATE = 10;

export const stateRate = (state) => {
  if (state === 'idle' || state === 'disabled') return SETTLE_RATE;
  if (state === 'error') return ERROR_RATE;
  return ENTER_RATE;
};

export const createStateMix = (initial = 'idle') => {
  const weights = {
    idle: 0,
    connecting: 0,
    listening: 0,
    thinking: 0,
    speaking: 0,
    error: 0,
    disabled: 0,
  };
  weights[initial] = 1;
  const keys = Object.keys(weights);
  const update = (state, dt, rate = stateRate(state)) => {
    let total = 0;
    for (const key of keys) {
      const target = key === state ? 1 : 0;
      const next = approach(weights[key], target, rate, dt);
      weights[key] = target === 0 && next < 0.001 ? 0 : next;
      total += weights[key];
    }
    if (total > 0) {
      for (const key of keys) weights[key] /= total;
    }
    return weights;
  };
  return { weights, update };
};

export const blendStates = (weights, table) => {
  const out = {};
  for (const key of Object.keys(weights)) {
    const w = weights[key];
    if (w === 0) continue;
    const row = table[key];
    for (const param of Object.keys(row)) {
      out[param] = (out[param] ?? 0) + row[param] * w;
    }
  }
  return out;
};

export const blendEnergy = (weights, t) => {
  let energy = 0;
  for (const key of Object.keys(weights)) {
    if (weights[key] > 0) energy += weights[key] * stateEnergy(key, t);
  }
  return energy;
};

export const LEVEL_ATTACK_RATE = 14;
export const LEVEL_RELEASE_RATE = 4;
export const MAX_FRAME_DT = 0.1;

export const smoothLevel = (current, target, dt) =>
  approach(current, target, target > current ? LEVEL_ATTACK_RATE : LEVEL_RELEASE_RATE, dt);

export const STATES = {
  idle: { tempo: 1, spin: 0.14, breathe: 0.05, drift: 1, ripple: 0, swell: 0.04, flow: 0, swirl: 0, pulse: 0, pulseRate: 1, ring: 0, jitter: 0, shake: 0, alpha: 0.72, rest: 0 },
  connecting: { tempo: 1, spin: 0.3, breathe: 0.02, drift: 0.2, ripple: 0, swell: 0, flow: 0, swirl: 0, pulse: 0.06, pulseRate: 0.5, ring: 1, jitter: 0, shake: 0, alpha: 0.8, rest: 0.12 },
  listening: { tempo: 1, spin: 0.55, breathe: 0.012, drift: 0, ripple: 1, swell: 0.12, flow: 0, swirl: 0, pulse: 0, pulseRate: 1, ring: 0, jitter: 0, shake: 0, alpha: 0.92, rest: 0.55 },
  thinking: { tempo: 1, spin: 0.32, breathe: 0.01, drift: 0, ripple: 0, swell: 0, flow: 0, swirl: 0, pulse: 1, pulseRate: 1, ring: 0, jitter: 0, shake: 0, alpha: 0.78, rest: 0.3 },
  speaking: { tempo: 1, spin: 0.24, breathe: 0.01, drift: 0, ripple: 0, swell: 0.06, flow: 1, swirl: 1, pulse: 0, pulseRate: 1, ring: 0, jitter: 0.6, shake: 0, alpha: 0.94, rest: 0.55 },
  error: { tempo: 1, spin: 0.08, breathe: 0, drift: 0, ripple: 0, swell: 0, flow: 0, swirl: 0, pulse: 0, pulseRate: 1, ring: 0, jitter: 0.7, shake: 1, alpha: 0.85, rest: 0.2 },
  disabled: { tempo: 0.04, spin: 0, breathe: 0, drift: 0, ripple: 0, swell: 0, flow: 0, swirl: 0, pulse: 0, pulseRate: 1, ring: 0, jitter: 0, shake: 0, alpha: 0.45, rest: 0 },
};

const buildSphere = (count) => {
  const sphere = {
    x: new Float32Array(count),
    y: new Float32Array(count),
    z: new Float32Array(count),
    ringFrac: new Float32Array(count),
    seed: new Float32Array(count),
    toneBucket: new Uint8Array(count),
    tone: new Float32Array(count),
  };
  for (let i = 0; i < count; i += 1) {
    const y = 1 - (i / (count - 1)) * 2;
    const radiusAtY = Math.sqrt(1 - y * y);
    const theta = GOLDEN_ANGLE * i;
    const tone = (i * 0.5436890126) % 1;
    sphere.x[i] = Math.cos(theta) * radiusAtY;
    sphere.y[i] = y;
    sphere.z[i] = Math.sin(theta) * radiusAtY;
    sphere.ringFrac[i] = (i * 0.61803398875) % 1;
    sphere.seed[i] = ((i * 0.7548776662) % 1) * TWO_PI;
    sphere.tone[i] = tone;
    sphere.toneBucket[i] = Math.min(TONE_BUCKETS - 1, Math.floor(tone * TONE_BUCKETS));
  }
  return sphere;
};

const SPHERE = buildSphere(PARTICLE_COUNT);

/**
 * ParticlesOrb class: encapsulates state transitions, audio levels, and rendering.
 */
export class ParticlesOrb {
  constructor({
    canvas,
    size = 280,
    speed = 1,
    colorFrom = '#fbbf24',
    colorTo = '#f43f5e',
    initialState = 'idle'
  }) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.size = size;
    this.speed = speed;
    this.colorFrom = colorFrom;
    this.colorTo = colorTo;
    this.state = initialState;

    this.liveLevel = -1; // -1 means no live audio (use procedural energy)
    this.mix = createStateMix(initialState);

    this.frame = {
      dt: 0,
      phase: 0,
      level: 0,
      time: 0
    };

    this.clock = 0;
    this.pulseClock = 0;
    this.angleY = 0;
    this.ringPhase = 0;
    this.lastPhase = 0;
    this.lastTime = null;
    this.rafId = null;

    // Buffer arrays
    const n = PARTICLE_COUNT;
    this.px = new Float32Array(n);
    this.py = new Float32Array(n);
    this.pr = new Float32Array(n);
    this.bucketOf = new Uint8Array(n);
    this.order = new Uint16Array(n);
    this.counts = new Uint16Array(BUCKETS);
    this.starts = new Uint16Array(BUCKETS);
    this.toneStyles = new Array(TONE_BUCKETS).fill('#fff');
    this.paletteKey = '';

    this.resize(size);
    this.start();
  }

  resize(size) {
    this.size = size;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = size * dpr;
    this.canvas.height = size * dpr;
    this.canvas.style.width = `${size}px`;
    this.canvas.style.height = `${size}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  setState(newState) {
    if (this.state !== newState && STATES[newState]) {
      this.state = newState;
    }
  }

  setLiveLevel(level) {
    this.liveLevel = typeof level === 'number' ? level : -1;
  }

  setColors(colorFrom, colorTo) {
    this.colorFrom = colorFrom;
    this.colorTo = colorTo;
    this.paletteKey = '';
  }

  ensurePalette(errorW) {
    const errKey = Math.round(errorW * 64);
    const key = `${this.colorFrom}|${this.colorTo}|${errKey}`;
    if (key === this.paletteKey) return;
    this.paletteKey = key;

    const baseFrom = hexToRgb(this.colorFrom);
    const baseTo = hexToRgb(this.colorTo);
    const e = errKey / 64;
    const f = mixRgb(baseFrom, ERROR_FROM_RGB, e);
    const t = mixRgb(baseTo, ERROR_TO_RGB, e);
    for (let b = 0; b < TONE_BUCKETS; b += 1) {
      this.toneStyles[b] = rgba(mixRgb(f, t, (b + 0.5) / TONE_BUCKETS), 1);
    }
  }

  tick(now) {
    const dt = this.lastTime === null ? 0 : Math.min((now - this.lastTime) / 1000, MAX_FRAME_DT);
    this.lastTime = now;

    this.mix.update(this.state, dt);
    const dPhase = dt * Math.max(0, this.speed);
    this.frame.time += dt;
    this.frame.phase += dPhase;

    const hasLive = typeof this.liveLevel === 'number' && this.liveLevel >= 0;
    const target = hasLive ? this.liveLevel : blendEnergy(this.mix.weights, this.frame.phase);
    this.frame.level = smoothLevel(this.frame.level, target, dt);
    this.frame.dt = dt;

    this.draw();
    this.rafId = requestAnimationFrame((t) => this.tick(t));
  }

  draw() {
    const ctx = this.ctx;
    const size = this.size;
    const center = size / 2;
    const smallness = Math.min(1, size / 140);
    const dotScale = 0.55 + 0.45 * smallness;
    const alphaScale = smallness * smallness;
    const baseRadius = center * 0.62;
    const n = PARTICLE_COUNT;

    const w = this.mix.weights;
    const p = blendStates(w, STATES);
    const dPhase = Math.max(0, this.frame.phase - this.lastPhase);
    this.lastPhase = this.frame.phase;
    const level = this.frame.level;

    this.clock += dPhase * p.tempo;
    this.pulseClock += dPhase * p.tempo * p.pulseRate;
    this.angleY += dPhase * p.spin * (1 + p.ripple * level * 1.8);
    this.ringPhase = (this.ringPhase + dPhase * 0.7) % TWO_PI;
    const t = this.clock + TIME_OFFSET;
    const pt = this.pulseClock + TIME_OFFSET;

    const beat = Math.sin(pt * 2.6) * 0.5 + 0.5;
    const beatSharp = beat * beat * beat;
    const breathe = p.breathe * Math.sin(t * 1.1);
    const conv = p.pulse * (0.06 + 0.12 * beatSharp);
    const radius = baseRadius * (1 + breathe + level * p.swell - conv);

    const shakeAmp = p.shake * radius * 0.05;
    const shakeX = shakeAmp * (Math.sin(t * 26) + 0.5 * Math.sin(t * 15.7));
    const shakeY = shakeAmp * (Math.cos(t * 22.5) + 0.5 * Math.sin(t * 13.1));

    const driftAmp = p.drift * radius * 0.055;
    const jitterAmp = p.jitter * radius * (0.012 + level * 0.07);
    const rippleAmp = p.ripple * (0.04 + level * 0.22);
    const pulseAmp = p.pulse * 0.16 * (0.4 + 0.6 * beat);
    const flowAmp = p.flow * (0.18 + level * 0.4);
    const swirlAmp = p.swirl * (0.35 + level * 0.9);
    const ringW = clamp01(p.ring);
    const ringBreath = 1 + p.pulse * 0.4 * Math.sin(pt * 2.6);

    const cosX = Math.cos(ANGLE_X);
    const sinX = Math.sin(ANGLE_X);
    this.counts.fill(0);

    for (let i = 0; i < n; i += 1) {
      const sx = SPHERE.x[i];
      const sy = SPHERE.y[i];
      const sz = SPHERE.z[i];
      const seed = SPHERE.seed[i];
      const ringFrac = SPHERE.ringFrac[i];

      const twist = swirlAmp > 0.002 ? this.angleY + swirlAmp * Math.sin(sy * 2.4 + t * 1.6) : this.angleY;
      const cy = Math.cos(twist);
      const sny = Math.sin(twist);
      const x1 = sx * cy - sz * sny;
      const z1 = sx * sny + sz * cy;
      const y1 = sy * cosX - z1 * sinX;
      const z2 = sy * sinX + z1 * cosX;

      const depth = (z2 + 1) / 2;
      const perspective = 0.65 + depth * 0.45;

      let pointRadius = radius;
      if (rippleAmp > 0.002) {
        pointRadius *= 1 + rippleAmp * (0.5 + 0.5 * Math.sin(sy * 4.5 - t * 6.5));
      }
      if (pulseAmp > 0.002) {
        pointRadius *= 1 - pulseAmp * (0.5 + 0.5 * Math.sin(ringFrac * TWO_PI + pt * 3.1));
      }
      if (flowAmp > 0.002) {
        const stream = 0.5 + 0.5 * Math.sin(seed * 3 - t * 3.4);
        pointRadius *= 1 - flowAmp * stream * stream;
      }

      let ox = shakeX;
      let oy = shakeY;
      if (driftAmp > 0.01) {
        ox += driftAmp * (Math.sin(t * 0.55 + seed * 3.7) + 0.5 * Math.sin(t * 1.3 + seed * 1.3));
        oy += driftAmp * (Math.cos(t * 0.62 + seed * 2.9) + 0.5 * Math.sin(t * 1.05 + seed * 5.1));
      }
      if (jitterAmp > 0.01) {
        ox += jitterAmp * Math.sin(t * 14 + seed * 9.3);
        oy += jitterAmp * Math.cos(t * 17 + seed * 6.1);
      }

      let screenX = center + x1 * pointRadius * perspective + ox;
      let screenY = center + y1 * pointRadius * perspective + oy;
      let alpha = 0.12 + depth * depth * 0.78;
      let dot = 0.6 + depth * 1.5;

      if (ringW > 0.004) {
        const tone = SPHERE.tone[i];
        const ringAngle = (i / n) * TWO_PI + this.ringPhase + 0.05 * Math.sin(t * 1.3 + seed);
        const ringR =
          center * (0.58 + 0.13 * ringFrac) * (1 + 0.05 * Math.sin(t + seed * 1.7)) * ringBreath;
        const circleX = center + Math.cos(ringAngle) * ringR;
        const circleY = center + Math.sin(ringAngle) * ringR;
        screenX += (circleX - screenX) * ringW;
        screenY += (circleY - screenY) * ringW;
        alpha += (0.35 + tone * 0.5 - alpha) * ringW;
        dot += (0.75 + tone * 0.9 - dot) * ringW;
      }

      this.px[i] = screenX;
      this.py[i] = screenY;
      this.pr[i] = dot * dotScale;
      alpha *= alphaScale;
      const ab = Math.min(ALPHA_BUCKETS - 1, Math.floor(alpha * ALPHA_BUCKETS));
      const b = SPHERE.toneBucket[i] * ALPHA_BUCKETS + ab;
      this.bucketOf[i] = b;
      this.counts[b] += 1;
    }

    let acc = 0;
    for (let b = 0; b < BUCKETS; b += 1) {
      this.starts[b] = acc;
      acc += this.counts[b];
    }
    for (let i = 0; i < n; i += 1) {
      const b = this.bucketOf[i];
      this.order[this.starts[b]] = i;
      this.starts[b] += 1;
    }

    this.ensurePalette(w.error);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);
    ctx.globalCompositeOperation = 'lighter';

    let cursor = 0;
    for (let b = 0; b < BUCKETS; b += 1) {
      const count = this.counts[b];
      if (count === 0) continue;
      const ab = b % ALPHA_BUCKETS;
      ctx.globalAlpha = clamp01(((ab + 0.5) / ALPHA_BUCKETS) * p.alpha);
      ctx.fillStyle = this.toneStyles[(b - ab) / ALPHA_BUCKETS];
      ctx.beginPath();
      for (let k = cursor; k < cursor + count; k += 1) {
        const i = this.order[k];
        ctx.moveTo(this.px[i] + this.pr[i], this.py[i]);
        ctx.arc(this.px[i], this.py[i], this.pr[i], 0, TWO_PI);
      }
      ctx.fill();
      cursor += count;
    }

    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
  }

  start() {
    if (!this.rafId) {
      this.lastTime = null;
      this.rafId = requestAnimationFrame((t) => this.tick(t));
    }
  }

  stop() {
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
  }
}
