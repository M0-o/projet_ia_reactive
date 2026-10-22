// Temporary diagnostic: isolate the pure bloom layer (scene held constant) and
// measure its radial profile, to see whether it is a haze or a streak.
import { create, globals } from "webgpu";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
Object.assign(globalThis, globals);
const root = process.cwd();
const { createWorld } = await import(join(root, "src/sim/world.js"));
const { makeParams } = await import(join(root, "src/sim/params.js"));
const { loadShaderParts } = await import(join(root, "src/sim/compose.js"));
const { createRenderer, createAtlasTexture } = await import(join(root, "src/render/renderer.js"));
const { OrbitCamera } = await import(join(root, "src/render/camera.js"));
const { bakeModel, buildModelPool, collectModelImages, applyAtlas } = await import(join(root, "src/render/models.js"));
const { buildAtlas } = await import(join(root, "src/render/textures.js"));

const W = 320, H = 240;
const gpu = create([]);
const device = await (await gpu.requestAdapter()).requestDevice();
const parts = await loadShaderParts((p) => readFile(join(root, p), "utf8"));
const params = { ...makeParams(), boidCount: 30, predatorCount: 2, foodCount: 0, obstacleCount: 1 };
const world = createWorld({ device, shaderParts: parts, params });
world.setParams(params); world.reset(7);
for (let f = 0; f < 20; f++) world.step(1 / 60, f / 60, f);
const mb = async (n) => { const b = await readFile(join(root, "3D-models", n + ".glb")); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const stubDecode = async () => ({ width: 4, height: 4, data: new Uint8Array(64).fill(200) });
const baked = [await bakeModel(await mb("boid"), { keyPrefix: "boid" }), await bakeModel(await mb("predator"), { keyPrefix: "predator" }), await bakeModel(await mb("obstacle"), { keyPrefix: "obstacle" })];
const baseAtlas = await buildAtlas(baked.flatMap((m) => collectModelImages(m, "base")), stubDecode, 64);
const emAtlas = await buildAtlas(baked.flatMap((m) => collectModelImages(m, "emissive")), stubDecode, 64);
for (const m of baked) applyAtlas(m, baseAtlas.placements, emAtlas.placements, baseAtlas.cell);
const pool = buildModelPool(baked);
const poolBuffer = device.createBuffer({ size: pool.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
device.queue.writeBuffer(poolBuffer, 0, pool);
const FMT = "bgra8unorm";
const tex = device.createTexture({ size: [W, H], format: FMT, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
const camera = new OrbitCamera({ addEventListener() {}, setPointerCapture() {}, releasePointerCapture() {} }, [params.worldX, params.worldY, params.worldZ]);
const renderer = createRenderer({ device, context: { getCurrentTexture: () => tex }, format: FMT, shaderParts: parts, world, models: { poolBuffer, boid: baked[0], predator: baked[1], obstacle: baked[2], atlasView: createAtlasTexture(device, baseAtlas), emissiveView: createAtlasTexture(device, emAtlas), emissiveStrength: true } });
const bpr = Math.ceil(W * 4 / 256) * 256;
async function grab(post) {
  renderer.updateCamera(camera.update(W, H));
  renderer.updateModels({ boidScale: params.boidModelScale, predatorScale: params.predatorModelScale, boidYaw: 0, boidPitch: params.boidModelPitch, predatorYaw: params.predatorModelYaw, predatorPitch: params.predatorModelPitch, glow: params.modelGlow });
  // Bloom OFF in the composite so we get the raw scene; bloom passes still run.
  renderer.render({ boids: 30, predators: 2, food: 0, obstacles: 1 }, { heading: false, vision: false, hitbox: false, bbox: false, hashCells: false }, W, H, post);
  const rb = device.createBuffer({ size: bpr * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const e = device.createCommandEncoder();
  e.copyTextureToBuffer({ texture: tex }, { buffer: rb, bytesPerRow: bpr }, [W, H]);
  device.queue.submit([e.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const px = new Uint8Array(rb.getMappedRange().slice(0)); rb.unmap(); rb.destroy();
  return px;
}
const scene = await grab({ bloom: false, threshold: params.bloomThreshold });
const withB = await grab({ bloom: true, threshold: params.bloomThreshold, strength: params.bloomStrength, radius: 4 });
// Difference image = pure bloom contribution.
let bx = 0, by = 0, best = -1;
const diff = new Float32Array(W * H);
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
  const o = y * bpr + x * 4;
  const d = (withB[o + 2] - scene[o + 2]) * 0.3 + (withB[o + 1] - scene[o + 1]) * 0.6 + (withB[o] - scene[o]) * 0.1;
  diff[y * W + x] = d;
  if (d > best) { best = d; bx = x; by = y; }
}
console.log(`pure bloom layer: peak ${best.toFixed(1)} at (${bx},${by})`);
let cardinal = [], diag = [];
for (const r of [0, 2, 4, 8, 16, 24, 32, 48, 64]) {
  let s = 0, n = 0;
  for (let a = 0; a < 72; a++) {
    const th = (a / 72) * Math.PI * 2;
    const x = Math.round(bx + Math.cos(th) * r), y = Math.round(by + Math.sin(th) * r);
    if (x < 0 || y < 0 || x >= W || y >= H) continue;
    s += diff[y * W + x]; n++;
  }
  cardinal.push([r, n ? +(s / n).toFixed(2) : null]);
}
console.log("radial mean bloom:", JSON.stringify(cardinal));
// Anisotropy: compare the horizontal vs vertical spread of the glow.
function extent(axis) {
  let c = 0;
  for (let k = 1; k < (axis === "x" ? W : H); k++) {
    const x = axis === "x" ? bx + k : bx, y = axis === "y" ? by + k : by;
    if (x >= W || y >= H) break;
    if (diff[y * W + x] < 5) break;
    c = k;
  }
  return c;
}
console.log(`glow extent: +X ${extent("x")}, +Y ${extent("y")} px`);
