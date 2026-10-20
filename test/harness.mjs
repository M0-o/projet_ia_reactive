// test/harness.mjs
// Headless verification of the WebGPU simulation using the Dawn bindings.
// This is the only place a GPU toolchain is needed; the app itself is plain
// static ES modules. Run with: npm test
//
// It checks:
//   * every WGSL module compiles (compute + all render modules)
//   * the spatial-hash counting sort produces a correct prefix sum + buckets
//   * boids stay inside the box, lose health over time, and refill it on food
//   * reaching food lifts the whole flock's health, not just the eater
//   * deaths spread out (per-boid vitality) instead of wiping the flock at once
//   * predators accumulate hits on a boid and eventually kill it
//   * GLB models parse, bake their animations, and render visibly

import { create, globals } from "webgpu";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

Object.assign(globalThis, globals);

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");

const { COMMON, COMPUTE_BINDINGS, SIM, RENDER_MODULES, allShaderFiles, composeCompute, composeRender, loadShaderParts } =
  await import(join(root, "src/sim/compose.js"));
const { createWorld, CAPACITY, AGENT_STRIDE, GRID_WORDS, MAX_CELLS } = await import(join(root, "src/sim/world.js"));
const { makeParams, computeGridDims } = await import(join(root, "src/sim/params.js"));

// ---- tiny test runner ------------------------------------------------------
let passed = 0;
const failures = [];
async function test(name, fn) {
  try {
    await fn();
    passed++;
    console.log(`  ok   ${name}`);
  } catch (err) {
    failures.push({ name, err });
    console.log(`  FAIL ${name}\n       ${err.message}`);
  }
}
function assert(cond, msg) {
  if (!cond) throw new Error(msg || "assertion failed");
}
function approx(a, b, tol, msg) {
  if (Math.abs(a - b) > tol) throw new Error(`${msg || "approx"}: ${a} != ${b} (tol ${tol})`);
}

// ---- GPU setup -------------------------------------------------------------
const gpu = create([]);
const adapter = await gpu.requestAdapter();
if (!adapter) {
  console.error("No WebGPU adapter available; skipping headless tests.");
  process.exit(0);
}
const device = await adapter.requestDevice();
device.addEventListener?.("uncapturederror", (e) => {
  console.error("GPU ERROR:", e.error?.message ?? e);
});

const parts = await loadShaderParts((p) => readFile(join(root, p), "utf8"));
const params0 = makeParams();

// ---- shader compilation ----------------------------------------------------
async function compile(label, code) {
  const mod = device.createShaderModule({ code });
  const info = await mod.getCompilationInfo();
  const errs = info.messages.filter((m) => m.type === "error");
  if (errs.length) {
    throw new Error(`${label}: ${errs.map((m) => `line ${m.lineNum}: ${m.message}`).join("\n       ")}`);
  }
  return mod;
}

console.log("shader compilation");
await test("compute module compiles", () => compile("sim", composeCompute(parts)));
for (const name of Object.keys(RENDER_MODULES)) {
  await test(`render module '${name}' compiles`, () => compile(name, composeRender(parts, name)));
}

// ---- helpers ---------------------------------------------------------------
function agentAt(f32, i) {
  const o = i * 12;
  return {
    pos: [f32[o + 0], f32[o + 1], f32[o + 2]],
    health: f32[o + 3],
    vel: [f32[o + 4], f32[o + 5], f32[o + 6]],
    hits: f32[o + 7],
    alive: f32[o + 8] > 0.5,
    theta: f32[o + 10],
    radius: f32[o + 11],
  };
}
function pointAt(f32, i) {
  const o = i * 8;
  return {
    pos: [f32[o + 0], f32[o + 1], f32[o + 2]],
    radius: f32[o + 3],
    active: f32[o + 4] > 0.5,
    timer: f32[o + 5],
  };
}
function cellIndexJS(pos, params, dims) {
  const minB = [-params.worldX / 2, -params.worldY / 2, -params.worldZ / 2];
  const c = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    const rel = (pos[k] - minB[k]) / params.gridCellSize;
    c[k] = Math.min(Math.max(Math.floor(rel), 0), [dims.dx, dims.dy, dims.dz][k] - 1);
  }
  return c[0] + c[1] * dims.dx + c[2] * dims.dx * dims.dy;
}
async function writeAgent(buffer, i, a) {
  const arr = new Float32Array(12);
  arr[0] = a.pos[0]; arr[1] = a.pos[1]; arr[2] = a.pos[2]; arr[3] = a.health ?? 100;
  arr[4] = a.vel?.[0] ?? 0; arr[5] = a.vel?.[1] ?? 0; arr[6] = a.vel?.[2] ?? 0;
  arr[7] = a.hits ?? 0; arr[8] = a.alive === false ? 0 : 1; arr[9] = 0;
  arr[10] = a.theta ?? 0; arr[11] = a.radius ?? 2.6;
  device.queue.writeBuffer(buffer, i * AGENT_STRIDE, arr);
}
async function writePoint(buffer, i, p) {
  const arr = new Float32Array(8);
  arr[0] = p.pos[0]; arr[1] = p.pos[1]; arr[2] = p.pos[2]; arr[3] = p.radius ?? 3;
  arr[4] = p.active === false ? 0 : 1; arr[5] = p.timer ?? 0; arr[6] = i; arr[7] = 100;
  device.queue.writeBuffer(buffer, i * 32, arr);
}
function nextFrame() {
  return new Promise((r) => setTimeout(r, 0));
}

// ---- spatial hash ----------------------------------------------------------
console.log("spatial hash");
await test("counting sort buckets match boid cells", async () => {
  const params = { ...params0, boidCount: 500, predatorCount: 0, foodCount: 0, obstacleCount: 0,
    maxSpeed: 0, maxForce: 0, alignWeight: 0, cohesionWeight: 0, separationWeight: 0,
    boundariesWeight: 0, avoidWeight: 0, seekFoodWeight: 0, fleePredatorWeight: 0, wanderWeight: 0,
    healthDecayPerSec: 0, gridCellSize: 9 };
  const world = createWorld({ device, shaderParts: parts, params });
  world.setParams(params);
  world.reset(7);
  world.step(1 / 60, 0, 1);
  await nextFrame();

  const dims = computeGridDims(params);
  const agents = await world.readAgents();
  const grid = new Uint32Array((await world.readBuffer(world.buffers.grid, GRID_WORDS * 4)).buffer);
  const O_STARTS = MAX_CELLS, O_ITEMS = 3 * MAX_CELLS + 256;

  // prefix sum is consistent with counts
  const counts = grid.subarray(0, dims.cells);
  const starts = grid.subarray(O_STARTS, O_STARTS + dims.cells);
  let running = 0;
  for (let c = 0; c < dims.cells; c++) {
    running += counts[c];
    if (starts[c] !== running) throw new Error(`prefix mismatch at cell ${c}: ${starts[c]} != ${running}`);
  }
  assert(running === params.boidCount, `total ${running} != ${params.boidCount}`);

  // every item is in the bucket its position maps to, and buckets are complete
  for (let c = 0; c < dims.cells; c++) {
    if (counts[c] === 0) continue;
    const start = c === 0 ? 0 : starts[c - 1];
    const end = starts[c];
    if (end - start !== counts[c]) throw new Error(`bucket ${c} size mismatch`);
    for (let k = start; k < end; k++) {
      const idx = grid[O_ITEMS + k];
      const a = agentAt(agents, idx);
      assert(cellIndexJS(a.pos, params, dims) === c, `item ${idx} in wrong cell`);
    }
  }
});

// Regression guard: the neighbour search must cover the whole perception radius
// regardless of cell size. With the old fixed 3x3x3 neighborhood, any cell
// smaller than `perception` silently hid most neighbours and flocking stopped
// working (it only "worked" once the cell size was pushed to the max).
await test("flocking aligns boids when cells are smaller than perception", async () => {
  const params = { ...params0,
    boidCount: 300, predatorCount: 0, foodCount: 0, obstacleCount: 0,
    // A deliberately small cell (perception is 26) that the old 3x3x3 scan
    // would have failed to cover; reach clamps to 3 rings here (9x9x9 cells).
    gridCellSize: 10,
    perception: 26, alignWeight: 2.0, cohesionWeight: 1.2, separationWeight: 0.2,
    wanderWeight: 0, seekFoodWeight: 0, fleePredatorWeight: 0,
    boundariesWeight: 0, avoidWeight: 0, maxSpeed: 55, maxForce: 40,
    healthDecayPerSec: 0, boidRespawnDelay: 1000,
    worldX: 120, worldY: 80, worldZ: 120 };
  const world = createWorld({ device, shaderParts: parts, params });
  world.setParams(params);
  world.reset(21);
  // Seeded uniformly, so any alignment must come from neighbour interaction.
  const t0 = await world.readAgents();
  const alignAt = (agents) => {
    let sum = 0, n = 0;
    for (let i = 0; i < params.boidCount; i++) {
      const a = agentAt(agents, i);
      if (!a.alive) continue;
      const l = Math.hypot(a.vel[0], a.vel[1], a.vel[2]) || 1;
      sum += a.vel[0] / l; n++;           // mean x-component of unit velocity
    }
    return n ? sum / n : 0;
  };
  const before = alignAt(t0);
  for (let f = 0; f < 600; f++) world.step(1 / 60, f / 60, f);
  await nextFrame();
  const after = alignAt(await world.readAgents());
  assert(Math.abs(after) > Math.abs(before) + 0.1,
    `boids did not align despite small cells (|mean dir| ${Math.abs(before).toFixed(3)} -> ${Math.abs(after).toFixed(3)})`);
});

// Spec conformance with boids.js: `cohesion()` uses 2x the perception radius
// while `align()`/`separation()` use the plain radius. Two boids sitting between
// the two radii must therefore be pulled together by cohesion (and not by
// alignment), so the gap between them must shrink.
await test("cohesion uses twice the perception radius (matches boids.js)", async () => {
  const params = { ...params0,
    boidCount: 2, predatorCount: 0, foodCount: 0, obstacleCount: 0,
    perception: 20, alignWeight: 0, separationWeight: 0, cohesionWeight: 4.0,
    wanderWeight: 0, seekFoodWeight: 0, fleePredatorWeight: 0,
    boundariesWeight: 0, avoidWeight: 0, maxSpeed: 5, maxForce: 30,
    healthDecayPerSec: 0, boidRespawnDelay: 1000, gridCellSize: 100,
    worldX: 400, worldY: 400, worldZ: 400 };
  const world = createWorld({ device, shaderParts: parts, params });
  world.setParams(params);
  world.reset(77);
  await nextFrame();
  // Separation is 30 units: beyond perception (20) but inside 2x perception (40).
  await writeAgent(world.buffers.boids, 0, { pos: [-15, 0, 0], health: 100, vel: [0, 0, 0], alive: true });
  await writeAgent(world.buffers.boids, 1, { pos: [15, 0, 0], health: 100, vel: [0, 0, 0], alive: true });
  for (let f = 0; f < 30; f++) world.step(1 / 60, f / 60, f);
  await nextFrame();
  const a = await world.readAgents();
  const gap = Math.abs(agentAt(a, 0).pos[0] - agentAt(a, 1).pos[0]);
  assert(gap < 30 - 1,
    `cohesion did not act at 2x perception (gap stayed ${gap.toFixed(2)}, expected < 29)`);
});

await test("two-level scan works at max grid size", async () => {
  // 256 * 16 * 16 = 65536 cells = exactly 256 scan blocks (the maximum).
  const params = { ...params0, boidCount: 300, predatorCount: 0, foodCount: 0, obstacleCount: 0,
    maxSpeed: 0, maxForce: 0, alignWeight: 0, cohesionWeight: 0, separationWeight: 0,
    boundariesWeight: 0, avoidWeight: 0, seekFoodWeight: 0, fleePredatorWeight: 0, wanderWeight: 0,
    healthDecayPerSec: 0, gridCellSize: 1, worldX: 256, worldY: 16, worldZ: 16 };
  const world = createWorld({ device, shaderParts: parts, params });
  world.setParams(params);
  world.reset(11);
  world.step(1 / 60, 0, 2);
  await nextFrame();
  const dims = computeGridDims(params);
  assert(dims.cells === MAX_CELLS, `expected ${MAX_CELLS} cells, got ${dims.cells}`);
  const grid = new Uint32Array((await world.readBuffer(world.buffers.grid, GRID_WORDS * 4)).buffer);
  const starts = grid.subarray(MAX_CELLS, MAX_CELLS + dims.cells);
  const counts = grid.subarray(0, dims.cells);
  let running = 0;
  for (let c = 0; c < dims.cells; c++) {
    running += counts[c];
    if (starts[c] !== running) throw new Error(`prefix mismatch at ${c}`);
  }
  assert(running === params.boidCount, `total ${running}`);
});

// ---- behaviour -------------------------------------------------------------
console.log("behaviour");

await test("boids stay inside the world box", async () => {
  const params = { ...params0, boidCount: 400, predatorCount: 4, foodCount: 10, obstacleCount: 8,
    healthDecayPerSec: 0, boidRespawnDelay: 1000 };
  const world = createWorld({ device, shaderParts: parts, params });
  world.setParams(params);
  world.reset(3);
  for (let f = 0; f < 240; f++) world.step(1 / 60, f / 60, f);
  await nextFrame();
  const agents = await world.readAgents();
  const eps = 0.001;
  for (let i = 0; i < params.boidCount; i++) {
    const a = agentAt(agents, i);
    if (!a.alive) continue;
    for (let k = 0; k < 3; k++) {
      const half = [params.worldX, params.worldY, params.worldZ][k] / 2;
      assert(Math.abs(a.pos[k]) <= half + eps, `boid ${i} escaped axis ${k}: ${a.pos[k]}`);
    }
  }
});

await test("health decays without food", async () => {
  const params = { ...params0, boidCount: 40, predatorCount: 0, foodCount: 0, obstacleCount: 0,
    healthDecayPerSec: 5, boidRespawnDelay: 1000 };
  const world = createWorld({ device, shaderParts: parts, params });
  world.setParams(params);
  world.reset(5);
  await nextFrame();
  const before = await world.readAgents();
  for (let f = 0; f < 120; f++) world.step(1 / 60, f / 60, f);
  await nextFrame();
  const after = await world.readAgents();
  for (let i = 0; i < params.boidCount; i++) {
    if (!agentAt(before, i).alive) continue;
    const a = agentAt(after, i);
    if (!a.alive) continue;
    assert(a.health < agentAt(before, i).health, `boid ${i} health did not decay`);
  }
});

// The population must thin out a *fraction* at a time rather than everyone dying
// and returning together. With per-boid vitality (decay) and resilience (hits)
// the deaths are spread out, so while a long starvation run is in progress some
// boids are already dead and others are still alive.
await test("boid deaths spread out instead of wiping the flock at once", async () => {
  const params = { ...params0, boidCount: 400, predatorCount: 0, foodCount: 0, obstacleCount: 0,
    healthDecayPerSec: 9, boidRespawnDelay: 1000, gridCellSize: 150 };
  const world = createWorld({ device, shaderParts: parts, params });
  world.setParams(params);
  world.reset(21);
  await nextFrame();

  let sawPartial = 0; // frames where the flock is a mix of alive + dead
  let sawAliveFrac = 0; // frames where at least one boid is alive
  const aliveCounts = [];
  for (let f = 0; f < 480; f++) {
    world.step(1 / 60, f / 60, f);
    if (f % 20 !== 0) continue;
    await nextFrame();
    const a = await world.readAgents();
    let alive = 0;
    for (let i = 0; i < params.boidCount; i++) if (agentAt(a, i).alive) alive++;
    aliveCounts.push(alive);
    if (alive > 0) sawAliveFrac++;
    if (alive > 0 && alive < params.boidCount * 0.98) sawPartial++;
  }
  // At some point the flock must be genuinely partial (some dead, some alive),
  // not a clean 100% -> 0% wipe.
  assert(sawPartial >= 2, `flock never thinned partially (partial frames: ${sawPartial}, samples: ${aliveCounts.join(",")})`);
  // And not every boid may die on the very first sample together: the alive
  // count must take at least two distinct values across the run.
  const distinct = new Set(aliveCounts);
  assert(distinct.size >= 2, `population never varied (counts: ${[...distinct].join(",")})`);
});

await test("eating food lifts the whole flock's health and consumes the cluster", async () => {
  const params = { ...params0, boidCount: 3, predatorCount: 0, foodCount: 1, obstacleCount: 0,
    healthDecayPerSec: 0, foodHeal: 5, foodRespawnDelay: 1000, maxSpeed: 0, maxForce: 0,
    alignWeight: 0, cohesionWeight: 0, separationWeight: 0, boundariesWeight: 0,
    avoidWeight: 0, seekFoodWeight: 0, fleePredatorWeight: 0, wanderWeight: 0 };
  const world = createWorld({ device, shaderParts: parts, params });
  world.setParams(params);
  world.reset(9);
  await nextFrame();
  // Boid 0 sits on the cluster; boids 1 and 2 are far away and healthy.
  await writeAgent(world.buffers.boids, 0, { pos: [0, 0, 0], health: 12, vel: [0, 0, 0] });
  await writeAgent(world.buffers.boids, 1, { pos: [200, 0, 0], health: 40, vel: [0, 0, 0] });
  await writeAgent(world.buffers.boids, 2, { pos: [0, 200, 0], health: 70, vel: [0, 0, 0] });
  await writePoint(world.buffers.food, 0, { pos: [0, 0, 0], radius: 3, active: true });
  const before = await world.readAgents();
  world.step(1 / 60, 0, 1);
  await nextFrame();
  const agents = await world.readAgents();
  const food = new Float32Array((await world.readBuffer(world.buffers.food, CAPACITY.food * 32)).buffer);
  assert(food[0 * 8 + 4] < 0.5, "food cluster was not consumed");
  // The average rises by foodHeal (5) for the whole population, not just the eater.
  const avg = (a) => (agentAt(a, 0).health + agentAt(a, 1).health + agentAt(a, 2).health) / 3;
  const rise = avg(agents) - avg(before);
  assert(Math.abs(rise - 5) < 0.5, `population average rose by ${rise.toFixed(2)}, expected ~5`);
  // Boid 0 ate, so it dropped nothing; boids 1 and 2 also gained the 5.
  assert(Math.abs(agentAt(agents, 1).health - 45) < 0.5, `far boid should also heal: ${agentAt(agents, 1).health}`);
  assert(Math.abs(agentAt(agents, 2).health - 75) < 0.5, `far boid should also heal: ${agentAt(agents, 2).health}`);
});

// Regression guard for the food respawn timer: eaten clusters must count down
// and come back. The old foodUpdate decremented a local copy but never wrote it
// back, so the timer restarted every frame and food was gone forever (the food
// counter only ever fell).
await test("eaten food regrows after the respawn delay", async () => {
  const params = { ...params0, boidCount: 1, predatorCount: 0, foodCount: 1, obstacleCount: 0,
    healthDecayPerSec: 0, foodRespawnDelay: 0.2, boidRespawnDelay: 1000,
    maxSpeed: 0, maxForce: 0, alignWeight: 0, cohesionWeight: 0, separationWeight: 0,
    boundariesWeight: 0, avoidWeight: 0, seekFoodWeight: 0, fleePredatorWeight: 0, wanderWeight: 0 };
  const world = createWorld({ device, shaderParts: parts, params });
  world.setParams(params);
  world.reset(31);
  await nextFrame();
  await writePoint(world.buffers.food, 0, { pos: [0, 0, 0], radius: 3, active: false, timer: params.foodRespawnDelay });
  const foodAt = async () => {
    const f = new Float32Array((await world.readBuffer(world.buffers.food, CAPACITY.food * 32)).buffer);
    return { active: f[4], timer: f[5] };
  };
  // Run well past the respawn delay.
  for (let f = 0; f < 40; f++) world.step(1 / 60, f / 60, f);
  await nextFrame();
  const after = await foodAt();
  assert(after.active > 0.5, "eaten food never regrew");
});

await test("predator hits kill a boid after hitsToKill contacts", async () => {
  const params = { ...params0, boidCount: 1, predatorCount: 1, foodCount: 0, obstacleCount: 0,
    healthDecayPerSec: 0, hitsToKill: 3, predatorDamage: 20, attackCooldown: 0,
    boidRespawnDelay: 1000, maxSpeed: 0, maxForce: 0, alignWeight: 0, cohesionWeight: 0,
    separationWeight: 0, boundariesWeight: 0, avoidWeight: 0, seekFoodWeight: 0,
    fleePredatorWeight: 0, wanderWeight: 0 };
  const world = createWorld({ device, shaderParts: parts, params });
  world.setParams(params);
  world.reset(13);
  await nextFrame();
  await writeAgent(world.buffers.boids, 0, { pos: [0, 0, 0], health: 100 });
  await writeAgent(world.buffers.predators, 0, { pos: [0, 0, 0], health: 100 });
  for (let f = 0; f < 5; f++) world.step(1 / 60, f / 60, f);
  await nextFrame();
  const agents = await world.readAgents();
  const a = agentAt(agents, 0);
  assert(!a.alive || a.hits >= params.hitsToKill, `boid should be dead by hits; hits=${a.hits} alive=${a.alive}`);
});

await test("boids avoid obstacles (never overlap an obstacle core)", async () => {
  const params = { ...params0, boidCount: 300, predatorCount: 0, foodCount: 0, obstacleCount: 1,
    healthDecayPerSec: 0, obstacleRadius: 22, boidRespawnDelay: 1000, wanderWeight: 0.6 };
  const world = createWorld({ device, shaderParts: parts, params });
  world.setParams(params);
  world.reset(17);
  await nextFrame();
  await writePoint(world.buffers.obstacles, 0, { pos: [0, 0, 0], radius: 22, active: true });
  for (let f = 0; f < 420; f++) world.step(1 / 60, f / 60, f);
  await nextFrame();
  const agents = await world.readAgents();
  let inside = 0;
  for (let i = 0; i < params.boidCount; i++) {
    const a = agentAt(agents, i);
    if (!a.alive) continue;
    const d = Math.hypot(a.pos[0], a.pos[1], a.pos[2]);
    if (d < params.obstacleRadius) inside++;
  }
  // A few frames of overshoot are acceptable, but boids must not sit inside.
  assert(inside <= params.boidCount * 0.02, `${inside}/${params.boidCount} boids inside the obstacle`);
});

// ---- GLB models ------------------------------------------------------------
console.log("glb models");
const { bakeModel, buildModelPool, collectModelImages, applyAtlas } =
  await import(join(root, "src/render/models.js"));
const { buildAtlas } = await import(join(root, "src/render/textures.js"));
const modelBuf = async (n) => (await readFile(join(root, `3D-models/${n}.glb`))).buffer;
// Bake with a unique key prefix (as loadModel does from the URL), so texture
// keys from different GLBs never collide in the atlas.
const bake = async (n) => bakeModel(await modelBuf(n), { keyPrefix: n });

// Node has no image decoder, so tests inject a stub that returns a solid-colour
// bitmap. The atlas layout logic (grid packing, UV rewriting) is what matters.
const stubDecode = async () => ({
  width: 4, height: 4,
  data: new Uint8ClampedArray(4 * 4 * 4).fill(200),
});

await test("boid GLB bakes a skinned, animated mesh", async () => {
  const m = bakeModel(await modelBuf("boid"));
  assert(m.vertexCount > 100, `expected a real mesh, got ${m.vertexCount} verts`);
  assert(m.indexCount > 0, "mesh must have indices");
  assert(m.hasSkin, "boid must be skinned");
  assert(m.info.jointCount > 0, "boid must have joints");
  assert(m.info.frameCount > 1, "boid animation must bake multiple frames");
  assert(m.info.animLength > 0, "boid animation must have a duration");
  assert(m.info.scale > 0 && Number.isFinite(m.info.scale), "scale must be finite");
});

await test("predator GLB bakes a skinned, animated mesh", async () => {
  const m = bakeModel(await modelBuf("predator"));
  assert(m.hasSkin && m.info.jointCount > 0, "predator must be skinned");
  assert(m.info.frameCount > 1, "predator animation must bake multiple frames");
});

await test("obstacle GLB bakes a static mesh", async () => {
  const m = bakeModel(await modelBuf("obstacle"));
  assert(m.vertexCount > 0 && m.indexCount > 0, "obstacle mesh must exist");
  assert(!m.hasSkin, "obstacle should be static");
});

// The phoenix declares alphaMode BLEND and stores the gaps between its feathers
// in the base-colour alpha. That mode must reach the vertex pool (so the shader
// can cut out), and the GLB's material factors must be carried through so the
// glow and albedo match the asset instead of a hard-coded tint.
await test("predator BLEND alpha mode is carried into the vertex pool", async () => {
  const m = bakeModel(await modelBuf("predator"));
  let blend = 0;
  for (let v = 0; v < m.vertexCount; v++) {
    if (m.pool[v * 12 + 3] === 2) blend++; // 2 = BLEND ordinal
    assert(m.pool[v * 12 + 7] > 0, `vertex ${v} lost its alpha cutoff`);
  }
  assert(blend === m.vertexCount,
    `expected every predator vertex to be BLEND, got ${blend}/${m.vertexCount}`);
  const ef = m.emissiveFactor;
  assert(ef && ef[0] === 1 && ef[1] === 1 && ef[2] === 1,
    `predator emissiveFactor should be (1,1,1), got ${ef}`);
  // Opaque models must NOT cut out, or the boid/obstacle would get holes.
  const boid = bakeModel(await modelBuf("boid"));
  for (let v = 0; v < boid.vertexCount; v++) {
    assert(boid.pool[v * 12 + 3] === 0, `boid vertex ${v} should be OPAQUE`);
  }
});

await test("baked joint frames actually differ across the clip", async () => {
  const m = bakeModel(await modelBuf("boid"));
  const jc = m.info.jointCount;
  const frameWords = jc * 16;
  let maxDelta = 0;
  let scale = 0;
  for (let j = 0; j < jc; j++) {
    const first = m.words - m.info.frameCount * frameWords + j * 16;
    const last = m.words - frameWords + j * 16;
    // Compare every matrix element: the clip may animate rotation and leave the
    // translation column nearly constant.
    for (let k = 0; k < 16; k++) {
      maxDelta = Math.max(maxDelta, Math.abs(m.pool[first + k] - m.pool[last + k]));
      scale = Math.max(scale, Math.abs(m.pool[first + k]));
    }
  }
  // Joint matrices are normalised to unit-radius space, so compare the delta
  // against the matrix magnitude rather than an absolute threshold.
  assert(maxDelta > 1e-4 * Math.max(scale, 1e-6),
    `animation appears static (max joint delta ${maxDelta}, scale ${scale})`);
});

await test("every GLB references at least one base-colour image", async () => {
  for (const name of ["boid", "predator", "obstacle"]) {
    const m = await bake(name);
    const images = collectModelImages(m, "base");
    assert(images.length > 0, `${name} should reference a texture`);
    for (const im of images) {
      assert(im.bytes && im.bytes.byteLength > 8, `${name} image ${im.key} must carry bytes`);
      // PNG signature, so we know we grabbed a real image payload.
      assert(im.bytes[0] === 0x89 && im.bytes[1] === 0x50, `${name} image must be a PNG`);
    }
  }
});

await test("predator GLB carries emissive (glow) images", async () => {
  const m = await bake("predator");
  const glow = collectModelImages(m, "emissive");
  assert(glow.length > 0, "predator should have an emissive map to glow");
  for (const prim of m.primitives) {
    assert(prim.emissiveKey, "every predator primitive must reference an emissive map");
  }
  // Non-glowing models must not accidentally report an emissive map.
  const obstacle = await bake("obstacle");
  assert(collectModelImages(obstacle, "emissive").length === 0,
    "obstacle should not glow");
});

await test("atlases pack images into cells and rewrite both UV sets", async () => {
  const baked = [await bake("boid"), await bake("predator"), await bake("obstacle")];
  const baseImages = baked.flatMap((m) => collectModelImages(m, "base"));
  const emImages = baked.flatMap((m) => collectModelImages(m, "emissive"));
  const baseAtlas = await buildAtlas(baseImages, stubDecode, 32);
  const emAtlas = await buildAtlas(emImages, stubDecode, 32);

  assert(baseAtlas.rgba.length === baseAtlas.width * baseAtlas.height * 4,
    "base atlas pixel count must match its size");
  // Keys are unique across GLBs, so a shared source index must not collapse.
  assert(baseAtlas.placements.size === new Set(baseImages.map((i) => i.key)).size,
    "every distinct base image key needs a cell");
  assert(emAtlas.placements.size === new Set(emImages.map((i) => i.key)).size,
    "every distinct emissive image key needs a cell");

  for (const m of baked) {
    applyAtlas(m, baseAtlas.placements, emAtlas.placements, baseAtlas.cell);
    const base = m.mesh.vertexOffset;
    for (const prim of m.primitives) {
      const check = (place, uWord, vWord, what) => {
        if (!place) return;
        for (let v = prim.vertexStart; v < prim.vertexStart + prim.vertexCount; v++) {
          const u = m.pool[base + v * 12 + uWord];
          const w = m.pool[base + v * 12 + vWord];
          assert(u >= place.u0 - 1e-5 && u <= place.u1 + 1e-5,
            `${what} u ${u} escapes cell [${place.u0}, ${place.u1}]`);
          assert(w >= place.v0 - 1e-5 && w <= place.v1 + 1e-5,
            `${what} v ${w} escapes cell [${place.v0}, ${place.v1}]`);
        }
      };
      // Base UV lives at words +8/+9, emissive UV at +10/+11.
      const brim = prim.textureKey != null ? baseAtlas.placements.get(prim.textureKey) : null;
      if (prim.textureKey != null) assert(brim, `no base placement for ${prim.textureKey}`);
      check(brim, 8, 9, "base");
      const emir = prim.emissiveKey != null ? emAtlas.placements.get(prim.emissiveKey) : null;
      if (prim.emissiveKey != null) assert(emir, `no emissive placement for ${prim.emissiveKey}`);
      check(emir, 10, 11, "emissive");
    }
  }

  // Distinct materials must occupy distinct cells, or two parts would share art.
  for (const atlas of [baseAtlas, emAtlas]) {
    const cells = new Set([...atlas.placements.values()].map((p) => `${p.u0},${p.v0}`));
    assert(cells.size === atlas.placements.size, "each image must get its own cell");
  }
});

await test("buildModelPool packs models with correct offsets", async () => {
  const models = [bakeModel(await modelBuf("boid")), bakeModel(await modelBuf("predator")), bakeModel(await modelBuf("obstacle"))];
  const pool = buildModelPool(models);
  assert(models[0].mesh.vertexOffset === 0, "first model at offset 0");
  assert(models[1].mesh.vertexOffset > 0, "second model offset must advance");
  assert(pool.length >= models.reduce((s, m) => s + m.words, 0), "pool must hold every model");
});

// Mirrors the shader's accessors in shaders/render/models.wgsl. If the pool
// layout and the shader ever disagree, indices or joints resolve to garbage
// and the models silently vanish — exactly the failure this guards against.
await test("shader pool layout resolves indices and joints for every model", async () => {
  const models = [bakeModel(await modelBuf("boid")), bakeModel(await modelBuf("predator")), bakeModel(await modelBuf("obstacle"))];
  const pool = buildModelPool(models);
  const f32 = pool;
  const u32 = new Uint32Array(pool.buffer);

  for (const m of models) {
    const { vertexOffset, vertexCount, indexOffset, jointOffset } = m.mesh;
    // Every index the shader reads must point at a vertex of this model.
    let maxIndex = -1;
    for (let k = 0; k < m.indexCount; k++) {
      maxIndex = Math.max(maxIndex, u32[indexOffset + k]);
    }
    assert(maxIndex >= 0 && maxIndex < vertexCount,
      `index ${maxIndex} out of range for ${vertexCount} verts`);

    // The first vertex must be at vertexOffset and be a sane, finite position.
    // (Raw positions are in bind space, so only skinned models come out near
    // the unit sphere; the check is just "not garbage".)
    const vx = f32[vertexOffset], vy = f32[vertexOffset + 1], vz = f32[vertexOffset + 2];
    assert(Number.isFinite(vx) && Number.isFinite(vy) && Number.isFinite(vz), "vertex 0 must be finite");
    assert(Math.hypot(vx, vy, vz) < 1e4, `vertex 0 looks like garbage: (${vx}, ${vy}, ${vz})`);

    // Joint block: check the offset lands on a valid index into the frame data
    // and (for skinned models) that it holds real matrices, not index bits.
    if (m.info.jointCount > 0) {
      const jc = m.info.jointCount;
      const stride = jc * 16;
      const last = jointOffset + (m.info.frameCount - 1) * stride + (jc - 1) * 16 + 15;
      assert(last < pool.length, `joint block overruns the pool (${last} >= ${pool.length})`);
      // An identity-ish bottom row (0,0,0,1) is the signature of a real mat4.
      assert(Math.abs(f32[jointOffset + 3]) < 1e-3 &&
             Math.abs(f32[jointOffset + 7]) < 1e-3 &&
             Math.abs(f32[jointOffset + 11]) < 1e-3 &&
             Math.abs(f32[jointOffset + 15] - 1) < 1e-3,
        "joint 0 frame 0 does not look like a mat4x4 (wrong jointOffset?)");
    }
  }
});

// ---- JS 3D reference (the Boid3D subclasses) -------------------------------
console.log("js 3d reference (Boid3D subclasses)");
const vm = await import("node:vm");
const legacyFiles = [
  "src/legacy/p5shim.js",
  "boids.js",
  "src/legacy/boid3d.js",
  "src/legacy/vehicle3d.js",
  "src/legacy/predator3d.js",
  "src/legacy/food3d.js",
  "src/legacy/obstacle3d.js",
];
const legacySrc = (await Promise.all(legacyFiles.map((f) => readFile(join(root, f), "utf8")))).join("\n;\n");
const legacyCtx = vm.createContext({});
vm.runInContext(legacySrc, legacyCtx, { filename: "legacy-bundle.js" });
const AL = vm.runInContext("AL", legacyCtx);

function lockBoid(b) {
  b.alignWeight = 0; b.cohesionWeight = 0; b.separationWeight = 0;
  b.boundariesWeight = 0; b.avoidWeight = 0; b.wanderWeight = 0;
  b.seekFoodWeight = 0; b.fleePredatorWeight = 0;
  b.healthDecayPerSec = 0;
}
const baseWorld = () => ({ boids: [], obstacles: [], food: [], predators: [], hitsToKill: 4, predatorVision: 72 });

await test("original Boid (2D) still loads unmodified", () => {
  const b = new AL.Boid3D(0, 0, 0);
  assert(typeof AL.Boid3D === "function" && AL.Boid3D.prototype instanceof AL.Boid, "Boid3D must extend Boid");
  assert(typeof b.align === "function" && typeof b.avoid === "function", "inherited behaviours missing");
});

await test("Boid3D generalises position + velocity to 3D", () => {
  const b = new AL.Boid3D(10, 20, 30);
  assert(b.pos.z === 30, "position must carry a z");
  assert(Math.abs(b.vel.z) > 0, "velocity must not stay in the z=0 plane");
});

await test("Boid3D seeks food", () => {
  const b = new AL.Boid3D(0, 0, 0);
  lockBoid(b);
  b.seekFoodWeight = 1;
  const world = baseWorld();
  world.boids = [b];
  const food = new AL.Food3D([60, 0, 0], 3);
  world.food = [food];
  const d0 = b.pos.dist(food.pos);
  let closest = d0;
  for (let i = 0; i < 60; i++) {
    b.tick(1 / 60, world);
    closest = Math.min(closest, b.pos.dist(food.pos));
  }
  assert(closest < 20, `boid never approached the food (closest ${closest}, started ${d0})`);
});

await test("Boid3D flees a predator", () => {
  const b = new AL.Boid3D(0, 0, 0);
  lockBoid(b);
  b.fleePredatorWeight = 1;
  const predator = new AL.Predator3D(40, 0, 0); // predator is static here
  const world = baseWorld();
  world.boids = [b];
  world.predators = [predator];
  const d0 = b.pos.dist(predator.pos);
  let far = d0;
  for (let i = 0; i < 60; i++) {
    b.tick(1 / 60, world);
    far = Math.max(far, b.pos.dist(predator.pos));
  }
  assert(far > d0 + 5, `boid never fled the predator (start ${d0}, max ${far})`);
});

await test("Boid3D avoids a spherical obstacle", () => {
  const b = new AL.Boid3D(0, 0, 0);
  lockBoid(b);
  // Heading +x at full speed; an obstacle sits right on that path.
  b.vel.x = b.maxSpeed; b.vel.y = 0; b.vel.z = 0;
  b.avoidWidth = 40;
  const obs = new AL.Obstacle3D([30, 0, 0], 15);
  const force = b.avoid([obs]);
  assert(force.mag() > 0, "avoid force should be non-zero for an obstacle ahead");
  assert(force.x < 0, `avoid force must push back away from the obstacle (got ${force.x})`);
  // ...and no force when the obstacle is behind / far away.
  const far = b.avoid([new AL.Obstacle3D([-300, 0, 0], 15)]);
  assert(far.mag() === 0, "no avoid force when the obstacle is behind");
});

await test("Predator3D kills a boid after hitsToKill contacts", () => {
  const b = new AL.Boid3D(0, 0, 0);
  b.healthDecayPerSec = 0;
  const p = new AL.Predator3D(0, 0, 0);
  // Keep the pair coincident so every tick is a contact (isolates the hit
  // counting from movement); the GPU test covers the moving case.
  p.maxSpeed = 0; p.maxForce = 0;
  const world = baseWorld();
  world.boids = [b];
  world.hitsToKill = 4;
  p.hitsToKill = 4;
  p.predatorDamage = 20;
  p.attackCooldown = 1;
  for (let i = 0; i < 8 && b.alive; i++) p.tick(1, world);
  assert(!b.alive, `boid should be dead by hits (hits=${b.hits})`);
  assert(b.hits >= 4, `hits=${b.hits} should be >= 4`);
});

await test("Boid3D respects the world box and refills health on food", () => {
  const b = new AL.Boid3D(0, 0, 0);
  b.healthDecayPerSec = 4;
  const world = baseWorld();
  world.boids = [b];
  for (let i = 0; i < 600; i++) {
    // Food is respawned in front of the boid every few frames: with the +5%
    // heal (not a full refill) the boid must keep eating to stay alive.
    if (i % 5 === 0) world.food = [new AL.Food3D(b.pos.copy(), 3)];
    b.tick(1 / 60, world);
  }
  const minB = b.boundsMin;
  const maxB = AL.Vec3.add(b.boundsMin, b.boundsSize);
  assert(b.pos.x >= minB.x && b.pos.x <= maxB.x, "x out of bounds");
  assert(b.pos.y >= minB.y && b.pos.y <= maxB.y, "y out of bounds");
  assert(b.pos.z >= minB.z && b.pos.z <= maxB.z, "z out of bounds");
  assert(b.alive, "boid should survive while food is available");
  assert(b.health <= 100, "health must be capped at 100");
});

// ---- render pipelines (offscreen) -----------------------------------------
// Builds every render pipeline and renders the scene to an offscreen texture,
// then reads the pixels back to prove the vertex + fragment stages actually
// produce non-background output. This is the closest we can get to a visual
// check without a real browser (the container has no display).
console.log("render pipelines (offscreen)");

const RENDER_FORMAT = "bgra8unorm";
const W = 256, H = 192;

await test("every render pipeline builds without error", async () => {
  const { renderLayout } = await import(join(root, "src/sim/bindings.js"));
  const layout = renderLayout(device);
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const specs = [
    ["sky", "vsSky", "fsSky", "triangle-list"],
    ["models", "vsBoid", "fsModel", "triangle-list"],
    ["models", "vsPredator", "fsModel", "triangle-list"],
    ["models", "vsObstacle", "fsModel", "triangle-list"],
    ["box", "vsBoxEdges", "fsBoxEdges", "line-list"],
    ["food", "vsFood", "fsFood", "triangle-list"],
    ["hashcells", "vsHashCells", "fsHashCells", "line-list"],
    ["debug", "vsHeadingBoids", "fsLine", "line-list"],
    ["debug", "vsVisionBoids", "fsLine", "line-list"],
    ["debug", "vsHitboxBoids", "fsLine", "line-list"],
    ["debug", "vsHitboxPredators", "fsLine", "line-list"],
    ["debug", "vsHitboxObstacles", "fsLine", "line-list"],
    ["debug", "vsBBoxBoids", "fsLine", "line-list"],
  ];
  for (const [mod, vs, fs, topo] of specs) {
    const module = device.createShaderModule({ code: composeRender(parts, mod) });
    device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: { module, entryPoint: vs },
      fragment: { module, entryPoint: fs, targets: [{ format: RENDER_FORMAT }] },
      primitive: { topology: topo },
      depthStencil: { format: "depth24plus", depthWriteEnabled: topo === "triangle-list", depthCompare: "less" },
    });
  }
});

await test("offscreen render produces visible pixels", async () => {
  const params = { ...params0, boidCount: 200, predatorCount: 4, foodCount: 8, obstacleCount: 4 };
  const world = createWorld({ device, shaderParts: parts, params });
  world.setParams(params);
  world.reset(31);
  for (let f = 0; f < 30; f++) world.step(1 / 60, f / 60, f);
  await nextFrame();

  const { createRenderer } = await import(join(root, "src/render/renderer.js"));
  const { OrbitCamera } = await import(join(root, "src/render/camera.js"));

  // Bake + pack the real models, exactly as the app does.
  const boid = bakeModel(await modelBuf("boid"));
  const predator = bakeModel(await modelBuf("predator"));
  const obstacle = bakeModel(await modelBuf("obstacle"));
  const pool = buildModelPool([boid, predator, obstacle]);
  const poolBuffer = device.createBuffer({
    size: pool.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(poolBuffer, 0, pool);

  // A fake canvas/context pair that renders into an offscreen texture instead
  // of a swap chain.
  const tex = device.createTexture({
    size: [W, H],
    format: RENDER_FORMAT,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const fakeContext = { getCurrentTexture: () => tex };
  const camera = new OrbitCamera({ addEventListener() {}, setPointerCapture() {}, releasePointerCapture() {} },
    [params.worldX, params.worldY, params.worldZ]);
  const renderer = createRenderer({
    device, context: fakeContext, format: RENDER_FORMAT, shaderParts: parts, world,
    models: { poolBuffer, boid, predator, obstacle },
  });

  renderer.updateCamera(camera.update(W, H));
  renderer.updateModels({ boidScale: 1, predatorScale: 1, boidYaw: 0, predatorYaw: 0 });
  renderer.render(
    { boids: params.boidCount, predators: params.predatorCount, food: params.foodCount, obstacles: params.obstacleCount },
    { heading: true, vision: true, hitbox: true, bbox: true, hashCells: true },
    W, H,
  );

  // Read back the colour attachment (bgra8unorm -> 4 bytes/pixel, 256-aligned rows).
  const bytesPerRow = Math.ceil(W * 4 / 256) * 256;
  const read = device.createBuffer({ size: bytesPerRow * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture: tex }, { buffer: read, bytesPerRow }, [W, H]);
  device.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const pixels = new Uint8Array(read.getMappedRange().slice(0));
  read.unmap();
  read.destroy();

  // The night-sky background is dark but not black (stars + gradient), so we
  // look for pixels that are clearly brighter — the lit boid/predator/obstacle
  // models and the debug overlays. bgra8unorm: byte 0 = blue, 1 = green, 2 = red.
  let bright = 0;
  let maxLuma = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const o = y * bytesPerRow + x * 4;
      const luma = pixels[o + 2] * 0.3 + pixels[o + 1] * 0.6 + pixels[o] * 0.1;
      maxLuma = Math.max(maxLuma, luma);
      if (luma > 70) bright++;
    }
  }
  assert(maxLuma > 120, `nothing bright in the frame (max luma ${maxLuma.toFixed(0)})`);
  assert(bright > 150, `expected visible models/overlays, only ${bright} bright pixels`);
});

// Models-only render: no food, no debug overlays, no box. Every non-background
// pixel must therefore come from the skinned/static GLB draw path. This is the
// strongest headless proof that the models are actually visible.
await test("models-only render shows the GLB meshes", async () => {
  const params = { ...params0, boidCount: 300, predatorCount: 6, foodCount: 0, obstacleCount: 6 };
  const world = createWorld({ device, shaderParts: parts, params });
  world.setParams(params);
  world.reset(7);
  for (let f = 0; f < 10; f++) world.step(1 / 60, f / 60, f);
  await nextFrame();

  const { createRenderer } = await import(join(root, "src/render/renderer.js"));
  const { OrbitCamera } = await import(join(root, "src/render/camera.js"));
  const boid = bakeModel(await modelBuf("boid"));
  const predator = bakeModel(await modelBuf("predator"));
  const obstacle = bakeModel(await modelBuf("obstacle"));
  const pool = buildModelPool([boid, predator, obstacle]);
  const poolBuffer = device.createBuffer({
    size: pool.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(poolBuffer, 0, pool);

  const tex = device.createTexture({
    size: [W, H], format: RENDER_FORMAT,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const camera = new OrbitCamera({ addEventListener() {}, setPointerCapture() {}, releasePointerCapture() {} },
    [params.worldX, params.worldY, params.worldZ]);
  const renderer = createRenderer({
    device, context: { getCurrentTexture: () => tex }, format: RENDER_FORMAT,
    shaderParts: parts, world, models: { poolBuffer, boid, predator, obstacle },
  });

  renderer.updateCamera(camera.update(W, H));
  renderer.updateModels({ boidScale: 1, predatorScale: 1, boidYaw: 0, predatorYaw: 0 });
  // Every model type, and nothing else.
  renderer.render(
    { boids: params.boidCount, predators: params.predatorCount, food: 0, obstacles: params.obstacleCount },
    { heading: false, vision: false, hitbox: false, bbox: false, hashCells: false },
    W, H,
  );

  const bytesPerRow = Math.ceil(W * 4 / 256) * 256;
  const read = device.createBuffer({ size: bytesPerRow * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture: tex }, { buffer: read, bytesPerRow }, [W, H]);
  device.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const pixels = new Uint8Array(read.getMappedRange().slice(0));
  read.unmap();
  read.destroy();

  // The sky is very dark; count pixels clearly lit by a model.
  let lit = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const o = y * bytesPerRow + x * 4;
      const luma = pixels[o + 2] * 0.3 + pixels[o + 1] * 0.6 + pixels[o] * 0.1;
      if (luma > 60) lit++;
    }
  }
  assert(lit > 300, `models-only frame shows almost nothing (${lit} lit pixels)`);
});
console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) {
  for (const f of failures) console.error(`\n${f.name}\n${f.err.stack}`);
  process.exit(1);
}
