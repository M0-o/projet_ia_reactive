// main.js
// Browser entry point. Loads shaders + GLB models, boots WebGPU, wires up the
// world, renderer, camera, control panel and stats, then runs the frame loop.

import { loadShaderParts } from "./sim/compose.js";
import { makeParams } from "./sim/params.js";
import { createWorld } from "./sim/world.js";
import { initWebGPU, onDeviceLost } from "./gpu/device.js";
import { loadModel, buildModelPool, collectModelImages, applyAtlas } from "./render/models.js";
import { buildAtlas, browserDecode } from "./render/textures.js";
import { createRenderer, createAtlasTexture } from "./render/renderer.js";
import { OrbitCamera } from "./render/camera.js";
import { buildPanel } from "./ui/panel.js";
import { Stats, renderStats } from "./ui/stats.js";

const canvas = document.getElementById("view");
const panelEl = document.getElementById("panel");
const statsEl = document.getElementById("stats");
const errorEl = document.getElementById("error");

// Boid draw cap: the simulation runs for every boid, but only this many are
// rendered with the (skinned) model, for performance. Nearest-to-camera win.
const BOID_DRAW_CAP = 800;

function showError(msg) {
  errorEl.textContent = msg;
  errorEl.classList.remove("hidden");
}

// Non-fatal GPU problems (e.g. a pipeline that fails validation in this
// browser) are shown in a dismissible banner so the cause is visible on screen.
function showGpuWarning(msg) {
  let el = document.getElementById("gpu-warning");
  if (!el) {
    el = document.createElement("div");
    el.id = "gpu-warning";
    el.className = "gpu-warning";
    document.body.appendChild(el);
  }
  el.textContent = "GPU: " + msg;
}
if (typeof window !== "undefined") window.__reportGpuError = showGpuWarning;

async function readShaderFile(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load shader ${path}: ${res.status}`);
  return res.text();
}

async function main() {
  const params = makeParams();
  const debug = { heading: false, vision: false, hitbox: false, bbox: false, hashCells: false };

  // The control panel is built up-front so it is usable even if the GPU is
  // unavailable.
  let paused = false;
  let seed = 1337;
  let world = null;
  let camera = null;
  const actions = {
    pause: () => { paused = !paused; },
    reset: () => world?.reset(seed),
    reseed: () => { seed = (Math.random() * 1e9) | 0; world?.reset(seed); },
  };

  buildPanel(panelEl, params, {
    onChange(key, value) {
      params[key] = value;
      world?.setParams(params);
      if (key === "worldX" || key === "worldY" || key === "worldZ") {
        camera?.frame([params.worldX, params.worldY, params.worldZ]);
      }
    },
    onDebug(key, value) { debug[key] = value; },
    onAction(name) { actions[name]?.(); },
    debug,
  });

  let shaderParts;
  try {
    shaderParts = await loadShaderParts(readShaderFile);
  } catch (e) {
    showError("Could not load shaders. " + e.message);
    return;
  }

  let gpu;
  try {
    gpu = await initWebGPU(canvas);
  } catch (e) {
    showError(e.userMessage || e.message);
    return;
  }
  const { device, context, format } = gpu;
  onDeviceLost(device, (msg) => showError("WebGPU device lost: " + msg));

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
  }
  resize();
  window.addEventListener("resize", resize);

  // ---- Load + bake the GLB models, then pack them into one GPU buffer ----
  let modelPack;
  try {
    const [boid, predator, obstacle] = await Promise.all([
      loadModel("3D-models/boid.glb"),
      loadModel("3D-models/predator.glb"),
      loadModel("3D-models/obstacle.glb"),
    ]);
    const baked = [boid, predator, obstacle];

    // Build the texture atlases from every image the GLBs reference, then
    // rewrite each vertex's UVs to sample its material's cell. Two atlases:
    // base colour and emissive (the predators' glowing parts). This keeps one
    // draw call per model type even when a mesh uses several materials.
    let atlasView, emissiveView, emissiveStrength;
    try {
      const baseImages = baked.flatMap((m) => collectModelImages(m, "base"));
      const emImages = baked.flatMap((m) => collectModelImages(m, "emissive"));
      const baseAtlas = await buildAtlas(baseImages, browserDecode);
      const emAtlas = emImages.length > 0
        ? await buildAtlas(emImages, browserDecode)
        : null;
      for (const m of baked) {
        applyAtlas(m, baseAtlas.placements, emAtlas ? emAtlas.placements : new Map(), baseAtlas.cell);
      }
      atlasView = createAtlasTexture(device, baseAtlas);
      if (emAtlas) {
        emissiveView = createAtlasTexture(device, emAtlas);
        emissiveStrength = true;
      }
    } catch (e) {
      // Textures are cosmetic: fall back to untextured tinted models.
      console.warn("[models] texture atlas unavailable:", e.message);
    }

    const pool = buildModelPool(baked);
    const poolBuffer = device.createBuffer({
      label: "model-pool",
      size: pool.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(poolBuffer, 0, pool);
    modelPack = { poolBuffer, boid, predator, obstacle, atlasView, emissiveView, emissiveStrength };
  } catch (e) {
    showError("Could not load 3D models. " + e.message);
    return;
  }

  world = createWorld({ device, shaderParts, params });
  world.reset(seed);

  camera = new OrbitCamera(canvas, [params.worldX, params.worldY, params.worldZ]);
  camera.frame([params.worldX, params.worldY, params.worldZ]);

  const renderer = createRenderer({
    device, context, format, shaderParts, world, models: modelPack,
  });
  const stats = new Stats(world);

  let last = performance.now();
  let simTime = 0;
  let frame = 0;
  let fps = 60;
  let fpsAccum = 0;
  let fpsCount = 0;

  function loop(now) {
    const dtRaw = (now - last) / 1000;
    last = now;
    const dt = Math.min(dtRaw, 1 / 20);

    if (dtRaw > 0) {
      fpsAccum = fpsAccum ? fpsAccum * 0.9 + (1 / dtRaw) * 0.1 : 1 / dtRaw;
      fpsCount++;
      if (fpsCount % 10 === 0) fps = fpsAccum;
    }

    resize();
    const widthPx = canvas.width;
    const heightPx = canvas.height;

    if (!paused) {
      simTime += dt * params.timeScale;
      world.step(dt, simTime, frame);
      frame++;
    }

    renderer.updateCamera(camera.update(widthPx, heightPx));
    renderer.updateModels({
      boidScale: params.boidModelScale,
      predatorScale: params.predatorModelScale,
      boidYaw: params.boidModelYaw,
      predatorYaw: params.predatorModelYaw,
      glow: params.modelGlow,
    });
    renderer.render(
      {
        // Cap the skinned boid draws; the simulation still updates them all.
        boids: Math.min(params.boidCount, world.capacity.boids, BOID_DRAW_CAP),
        predators: Math.min(params.predatorCount, world.capacity.predators),
        food: Math.min(params.foodCount, world.capacity.food),
        obstacles: Math.min(params.obstacleCount, world.capacity.obstacles),
      },
      debug,
      widthPx,
      heightPx,
    );

    stats.update(params, fps, now).then(() => renderStats(statsEl, stats.get()));
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}

main();
