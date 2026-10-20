// stats.js
// Throttled CPU readback of the boid buffer to show live statistics. Reading a
// GPU buffer stalls the pipeline, so this runs only every ~250ms, not per frame.

import { AGENT_STRIDE } from "../sim/bindings.js";

const FLOATS_PER_AGENT = AGENT_STRIDE / 4; // 12
const FLOATS_PER_POINT = 8;                // posR(4) + extra(4)

function countActive(flat, count) {
  let n = 0;
  for (let i = 0; i < count; i++) {
    if (flat[i * FLOATS_PER_POINT + 4] > 0.5) n++; // extra.x = active flag
  }
  return n;
}

export class Stats {
  constructor(world) {
    this.world = world;
    this.last = { alive: 0, avgHealth: 0, predators: 0, food: 0, fps: 0 };
    this._pending = false;
    this._lastRun = 0;
    this._intervalMs = 250;
  }

  // Call once per frame with the current params + measured fps.
  async update(params, fps, now = performance.now()) {
    if (this._pending) return;
    if (now - this._lastRun < this._intervalMs) return;
    this._lastRun = now;
    this._pending = true;
    try {
      const data = await this.world.readAgents();
      const n = Math.min(params.boidCount, this.world.capacity.boids);
      let alive = 0, hp = 0;
      for (let i = 0; i < n; i++) {
        const infoX = data[i * FLOATS_PER_AGENT + 8]; // info.x = alive flag
        if (infoX > 0.5) {
          alive++;
          hp += data[i * FLOATS_PER_AGENT + 3]; // posH.w = health
        }
      }
      // Food and predators are read back too, so the counters reflect what the
      // simulation is actually doing (eaten clusters regrow, dead predators
      // stay dead) instead of echoing the static params.
      const foodFlat = await this.world.readPoints(this.world.buffers.food, this.world.capacity.food);
      const predRaw = await this.world.readBuffer(
        this.world.buffers.predators, this.world.capacity.predators * AGENT_STRIDE);
      const predData = new Float32Array(predRaw.buffer, predRaw.byteOffset, predRaw.byteLength / 4);
      let predators = 0;
      const pn = Math.min(params.predatorCount, this.world.capacity.predators);
      for (let i = 0; i < pn; i++) {
        if (predData[i * FLOATS_PER_AGENT + 8] > 0.5) predators++;
      }
      this.last = {
        alive,
        avgHealth: alive ? hp / alive : 0,
        predators,
        food: countActive(foodFlat, Math.min(params.foodCount, this.world.capacity.food)),
        fps,
      };
    } finally {
      this._pending = false;
    }
  }

  get() { return this.last; }
}

export function renderStats(el, stats) {
  el.innerHTML =
    `<div class="stat"><span>Boids alive</span><b>${stats.alive}</b></div>` +
    `<div class="stat"><span>Avg health</span><b>${stats.avgHealth.toFixed(0)}%</b></div>` +
    `<div class="stat"><span>Predators</span><b>${stats.predators}</b></div>` +
    `<div class="stat"><span>Food clusters</span><b>${stats.food}</b></div>` +
    `<div class="stat"><span>FPS</span><b>${stats.fps.toFixed(0)}</b></div>`;
}
