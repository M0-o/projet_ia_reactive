// world.js
// Owns all simulation buffers, the compute pipelines and one step() call.
// Deliberately environment-agnostic: it only needs a GPUDevice and already
// composed shader source, so the browser app and the Node test harness share
// exactly the same code path.

import { CAPACITY, GRID_WORDS, AGENT_STRIDE, POINT_STRIDE,
         MAX_CELLS, computeLayout, STORAGE, UNIFORM } from "./bindings.js";
import { ParamPacker, computeGridDims } from "./params.js";
import { composeCompute } from "./compose.js";

const COMPUTE_ENTRIES = [
  "gridClear", "gridCount", "gridScanLocal", "gridScanPartial", "gridScanAdd",
  "gridCursorInit", "gridScatter", "boidsUpdate", "predatorsUpdate", "foodUpdate",
  "flockHealUpdate",
];

function mulberry32(a) {
  let s = a >>> 0;
  return function () {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomDir(rng) {
  const u = rng() * 2 - 1;
  const t = rng() * Math.PI * 2;
  const r = Math.sqrt(Math.max(0, 1 - u * u));
  return [r * Math.cos(t), r * Math.sin(t), u];
}

export function createWorld({ device, shaderParts, params, seed = 1337 }) {
  const code = composeCompute(shaderParts);
  const module = device.createShaderModule({ code, label: "sim-module" });
  const layout = computeLayout(device);
  const pipelineLayout = device.createPipelineLayout({ label: "sim-pipeline-layout", bindGroupLayouts: [layout] });

  const buffers = {
    params: device.createBuffer({ label: "params", size: 15 * 16, usage: UNIFORM }),
    boids: device.createBuffer({ label: "boids", size: CAPACITY.boids * AGENT_STRIDE, usage: STORAGE }),
    predators: device.createBuffer({ label: "predators", size: CAPACITY.predators * AGENT_STRIDE, usage: STORAGE }),
    food: device.createBuffer({ label: "food", size: CAPACITY.food * POINT_STRIDE, usage: STORAGE }),
    obstacles: device.createBuffer({ label: "obstacles", size: CAPACITY.obstacles * POINT_STRIDE, usage: STORAGE }),
    grid: device.createBuffer({ label: "grid", size: GRID_WORDS * 4, usage: STORAGE | GPUBufferUsage.COPY_SRC }),
  };

  const pipelines = {};
  for (const entryPoint of COMPUTE_ENTRIES) {
    pipelines[entryPoint] = device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint } });
  }

  const bindGroup = device.createBindGroup({
    label: "compute-bind-group",
    layout,
    entries: [
      { binding: 0, resource: { buffer: buffers.params } },
      { binding: 1, resource: { buffer: buffers.boids } },
      { binding: 2, resource: { buffer: buffers.predators } },
      { binding: 3, resource: { buffer: buffers.food } },
      { binding: 4, resource: { buffer: buffers.obstacles } },
      { binding: 5, resource: { buffer: buffers.grid } },
    ],
  });

  const packer = new ParamPacker();
  let state = { ...params };
  let dims = computeGridDims(state);

  // ---- initial data ----------------------------------------------------
  function genAgents(buffer, capacity, count, radius, speed, rng) {
    const data = new Float32Array(capacity * 12);
    const [sx, sy, sz] = [state.worldX, state.worldY, state.worldZ];
    for (let k = 0; k < capacity; k++) {
      const o = k * 12;
      data[o + 0] = (rng() - 0.5) * sx * 0.92;
      data[o + 1] = (rng() - 0.5) * sy * 0.92;
      data[o + 2] = (rng() - 0.5) * sz * 0.92;
      // Spread initial health over [35, 100] so the flock does not all hit zero
      // (and respawn) at the same instant; the population count then drifts.
      data[o + 3] = (0.35 + 0.65 * rng()) * 100;
      const d = randomDir(rng);
      data[o + 4] = d[0] * speed;
      data[o + 5] = d[1] * speed;
      data[o + 6] = d[2] * speed;
      data[o + 7] = 0;
      data[o + 8] = 1;
      data[o + 9] = 0;
      data[o + 10] = rng() * Math.PI * 2;
      data[o + 11] = radius;
    }
    device.queue.writeBuffer(buffer, 0, data);
  }

  function genFood(rng) {
    const data = new Float32Array(CAPACITY.food * 8);
    const [sx, sy, sz] = [state.worldX, state.worldY, state.worldZ];
    const centers = [];
    const count = Math.min(state.foodCount, CAPACITY.food);
    // Keep clusters out of a central dead-zone so they do not all pile up in
    // the middle of the world; they scatter across the whole volume instead.
    const minFromCentre = Math.min(sx, sy, sz) * 0.18;
    for (let k = 0; k < CAPACITY.food; k++) {
      const o = k * 8;
      if (k < count) {
        let cx = 0, cy = 0, cz = 0, tries = 0;
        do {
          cx = (rng() - 0.5) * sx * 0.92;
          cy = (rng() - 0.5) * sy * 0.92;
          cz = (rng() - 0.5) * sz * 0.92;
          tries++;
          const tooClose =
            Math.hypot(cx, cy, cz) < minFromCentre ||
            centers.some((c) => Math.hypot(c[0] - cx, c[1] - cy, c[2] - cz) < state.eatRadius * 3);
          if (!tooClose) break;
        } while (tries < 48);
        centers.push([cx, cy, cz]);
        data[o + 0] = cx; data[o + 1] = cy; data[o + 2] = cz;
        data[o + 3] = state.foodRadius;
        data[o + 4] = 1;  // active
        data[o + 5] = 0;  // respawn timer
        data[o + 6] = k;  // seed
        data[o + 7] = 100;
      } else {
        data[o + 3] = state.foodRadius;
        data[o + 4] = 0;
      }
    }
    device.queue.writeBuffer(buffers.food, 0, data);
  }

  function genObstacles(rng) {
    const data = new Float32Array(CAPACITY.obstacles * 8);
    const [sx, sy, sz] = [state.worldX, state.worldY, state.worldZ];
    const spheres = [];
    const count = Math.min(state.obstacleCount, CAPACITY.obstacles);
    for (let k = 0; k < CAPACITY.obstacles; k++) {
      const o = k * 8;
      if (k < count) {
        // Vary each obstacle's size so the field is not a grid of clones: the
        // "Obstacle radius" slider is the mean, and each rock is a random
        // 0.4x-1.8x multiple of it.
        const r = state.obstacleRadius * (0.4 + rng() * 1.4);
        let cx = 0, cy = 0, cz = 0, tries = 0;
        do {
          cx = (rng() - 0.5) * sx * 0.9;
          cy = (rng() - 0.5) * sy * 0.9;
          cz = (rng() - 0.5) * sz * 0.9;
          tries++;
        } while (tries < 48 && spheres.some((s) => Math.hypot(s[0] - cx, s[1] - cy, s[2] - cz) < (s[3] + r) * 1.4));
        spheres.push([cx, cy, cz, r]);
        data[o + 0] = cx; data[o + 1] = cy; data[o + 2] = cz;
        data[o + 3] = r;
        data[o + 4] = 1;
        data[o + 5] = 0;
        data[o + 6] = k;
        data[o + 7] = r;
      } else {
        data[o + 3] = state.obstacleRadius;
        data[o + 4] = 0;
      }
    }
    device.queue.writeBuffer(buffers.obstacles, 0, data);
  }

  function reset(newSeed = seed) {
    const rng = mulberry32(newSeed);
    genAgents(buffers.boids, CAPACITY.boids, state.boidCount, state.boidRadius, state.maxSpeed * 0.25, rng);
    genAgents(buffers.predators, CAPACITY.predators, state.predatorCount, state.predatorRadius, state.predatorSpeed * 0.3, rng);
    genFood(rng);
    genObstacles(rng);
  }

  function setParams(p) {
    state = { ...p };
    dims = computeGridDims(state);
  }

  function step(dt, time, frame) {
    const view = packer.write(state, dims, dt, time, frame);
    device.queue.writeBuffer(buffers.params, 0, view);

    const nB = Math.max(1, Math.ceil(state.boidCount / 64));
    const nP = Math.max(1, Math.ceil(state.predatorCount / 64));
    const nF = Math.max(1, Math.ceil(state.foodCount / 64));
    const nCells64 = Math.max(1, Math.ceil(dims.cells / 64));
    const nCells256 = Math.max(1, Math.ceil(dims.cells / 256));

    const enc = device.createCommandEncoder({ label: "sim-step" });
    const pass = enc.beginComputePass({ label: "sim-compute" });
    pass.setBindGroup(0, bindGroup);

    const run = (name, groups) => {
      pass.setPipeline(pipelines[name]);
      pass.dispatchWorkgroups(groups);
    };

    run("gridClear", nCells64);
    run("gridCount", nB);
    run("gridScanLocal", nCells256);
    run("gridScanPartial", 1);
    run("gridScanAdd", nCells256);
    run("gridCursorInit", nCells64);
    run("gridScatter", nB);
    run("boidsUpdate", nB);
    run("predatorsUpdate", nP);
    run("foodUpdate", nF);
    run("flockHealUpdate", nB);

    pass.end();
    device.queue.submit([enc.finish()]);
  }

  // Copy a region of a buffer back to the CPU. Used by the UI stats readout
  // (throttled) and by the Node test harness.
  async function readBuffer(src, bytes) {
    const read = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, read, 0, bytes);
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const out = new Uint8Array(read.getMappedRange().slice(0));
    read.unmap();
    read.destroy();
    return out;
  }

  async function readAgents() {
    const bytes = CAPACITY.boids * AGENT_STRIDE;
    const raw = await readBuffer(buffers.boids, bytes);
    return new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
  }

  // Read a point buffer (food / obstacles) as flat floats. Point layout is
  // posR(4) + extra(4); extra.x is the active flag.
  async function readPoints(buffer, capacity) {
    const bytes = capacity * POINT_STRIDE;
    const raw = await readBuffer(buffer, bytes);
    return new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
  }

  function getDims() {
    return dims;
  }

  return {
    buffers,
    bindGroup,
    pipelines,
    layout,
    reset,
    setParams,
    step,
    readBuffer,
    readAgents,
    readPoints,
    getDims,
    capacity: CAPACITY,
    gridWords: GRID_WORDS,
  };
}

export { CAPACITY, GRID_WORDS, AGENT_STRIDE, POINT_STRIDE, MAX_CELLS };
