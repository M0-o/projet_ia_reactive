// Temporary diagnostic: measure the pure bloom layer's shape for ONE radius
// (the container's software Dawn is flaky across many renders, so each radius
// runs in its own process).
//   peak  : strongest single-pixel addition
//   iso   : sigmaMin/sigmaMax of the glow (1.0 = round mist, low = streak)
//   wide  : pixels lifted by more than 8 luma (the visible haze)
import { create, globals } from "webgpu";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { decodePNG } from "./_png.mjs";
Object.assign(globalThis, globals);
const root = process.cwd();
const { createWorld } = await import(join(root, "src/sim/world.js"));
const { makeParams } = await import(join(root, "src/sim/params.js"));
const { loadShaderParts } = await import(join(root, "src/sim/compose.js"));
const { createRenderer, createAtlasTexture } = await import(join(root, "src/render/renderer.js"));
const { OrbitCamera } = await import(join(root, "src/render/camera.js"));
const { bakeModel, buildModelPool, collectModelImages, applyAtlas } = await import(join(root, "src/render/models.js"));
const { buildAtlas } = await import(join(root, "src/render/textures.js"));

const W = 400, H = 300, FMT = "bgra8unorm";
const gpu = create([]);
const device = await (await gpu.requestAdapter()).requestDevice();
const parts = await loadShaderParts((p) => readFile(join(root, p), "utf8"));
const params = { ...makeParams(), boidCount: 0, predatorCount: 3, foodCount: 0, obstacleCount: 0 };
const world = createWorld({ device, shaderParts: parts, params });
world.setParams(params); world.reset(7);
for (let f = 0; f < 5; f++) world.step(1 / 60, f / 60, f);
const mb = async (n) => { const b = await readFile(join(root, "3D-models", n + ".glb")); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const decode = async (b) => decodePNG(b);
const baked = [await bakeModel(await mb("boid"), { keyPrefix: "boid" }), await bakeModel(await mb("predator"), { keyPrefix: "predator" }), await bakeModel(await mb("obstacle"), { keyPrefix: "obstacle" })];
const baseAtlas = await buildAtlas(baked.flatMap((m) => collectModelImages(m, "base")), decode, 64);
const emAtlas = await buildAtlas(baked.flatMap((m) => collectModelImages(m, "emissive")), decode, 64);
for (const m of baked) applyAtlas(m, baseAtlas.placements, emAtlas.placements, baseAtlas.cell);
const pool = buildModelPool(baked);
const poolBuffer = device.createBuffer({ size: pool.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
device.queue.writeBuffer(poolBuffer, 0, pool);
const tex = device.createTexture({ size: [W, H], format: FMT, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
const camera = new OrbitCamera({ addEventListener() {}, setPointerCapture() {}, releasePointerCapture() {} }, [params.worldX, params.worldY, params.worldZ]);
const renderer = createRenderer({ device, context: { getCurrentTexture: () => tex }, format: FMT, shaderParts: parts, world, models: { poolBuffer, boid: baked[0], predator: baked[1], obstacle: baked[2], atlasView: createAtlasTexture(device, baseAtlas), emissiveView: createAtlasTexture(device, emAtlas), emissiveStrength: true } });
const bpr = Math.ceil(W * 4 / 256) * 256;
const rb = device.createBuffer({ size: bpr * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
async function grab(post) {
  renderer.updateCamera(camera.update(W, H));
  renderer.updateModels({ boidScale: params.boidModelScale, predatorScale: params.predatorModelScale, boidYaw: 0, boidPitch: params.boidModelPitch, predatorYaw: params.predatorModelYaw, predatorPitch: params.predatorModelPitch, glow: params.modelGlow });
  renderer.render({ boids: 0, predators: 3, food: 0, obstacles: 0 }, { heading: false, vision: false, hitbox: false, bbox: false, hashCells: false }, W, H, post);
  const e = device.createCommandEncoder();
  e.copyTextureToBuffer({ texture: tex }, { buffer: rb, bytesPerRow: bpr }, [W, H]);
  device.queue.submit([e.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const px = new Uint8Array(rb.getMappedRange().slice(0));
  rb.unmap();
  return px;
}
const base = await grab({ bloom: false });
const radius = Number(process.argv[2] ?? 1);
const b = await grab({ bloom: true, threshold: params.bloomThreshold, strength: params.bloomStrength, radius });
let total = 0, cx = 0, cy = 0, peak = 0, ppx = 0, ppy = 0, wide = 0;
const d = new Float32Array(W * H);
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
  const o = y * bpr + x * 4;
  const dv = Math.max(0, (b[o + 2] - base[o + 2]) * 0.3 + (b[o + 1] - base[o + 1]) * 0.6 + (b[o] - base[o]) * 0.1);
  d[y * W + x] = dv;
  total += dv; cx += dv * x; cy += dv * y;
  if (dv > peak) { peak = dv; ppx = x; ppy = y; }
  if (dv > 8) wide++;
}
if (total <= 0) { console.log(`radius ${radius}: NO BLOOM`); process.exit(0); }
cx /= total; cy /= total;
let sx = 0, sy = 0;
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const dv = d[y * W + x]; sx += dv * (x - cx) ** 2; sy += dv * (y - cy) ** 2; }
sx = Math.sqrt(sx / total); sy = Math.sqrt(sy / total);
// Radial profile of the bloom layer around the peak.
const prof = [];
for (const r of [0, 4, 8, 16, 24, 32, 48, 64]) {
  let s = 0, n = 0;
  for (let a = 0; a < 48; a++) {
    const th = (a / 48) * Math.PI * 2;
    const x = Math.round(ppx + Math.cos(th) * r), y = Math.round(ppy + Math.sin(th) * r);
    if (x < 0 || y < 0 || x >= W || y >= H) continue;
    s += d[y * W + x]; n++;
  }
  prof.push(n ? +(s / n).toFixed(1) : null);
}
console.log(`radius ${radius}: peak ${peak.toFixed(0)} @(${ppx},${ppy}) sigmaX ${sx.toFixed(1)} sigmaY ${sy.toFixed(1)} iso ${(Math.min(sx, sy) / Math.max(sx, sy)).toFixed(3)} wide ${wide} energy ${(total / 1000).toFixed(0)}k radial ${JSON.stringify(prof)}`);
