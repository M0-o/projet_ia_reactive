// Temporary diagnostic: a single small bright emitter, to measure the glow's
// spatial width and isotropy (haze vs streak) cleanly.
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

const W = 320, H = 240, bpr = Math.ceil(W * 4 / 256) * 256, FMT = "bgra8unorm";
const gpu = create([]);
const device = await (await gpu.requestAdapter()).requestDevice();
const parts = await loadShaderParts((p) => readFile(join(root, p), "utf8"));
// A single predator, centred, is the only emitter.
const params = { ...makeParams(), boidCount: 0, predatorCount: 1, foodCount: 0, obstacleCount: 0, modelGlow: 3.0 };
const world = createWorld({ device, shaderParts: parts, params });
world.setParams(params); world.reset(7);
for (let f = 0; f < 5; f++) world.step(1 / 60, f / 60, f);
const mb = async (n) => { const b = await readFile(join(root, "3D-models", n + ".glb")); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const stubDecode = async () => ({ width: 4, height: 4, data: new Uint8Array(64).fill(255) });
const baked = [await bakeModel(await mb("boid"), { keyPrefix: "boid" }), await bakeModel(await mb("predator"), { keyPrefix: "predator" }), await bakeModel(await mb("obstacle"), { keyPrefix: "obstacle" })];
const baseAtlas = await buildAtlas(baked.flatMap((m) => collectModelImages(m, "base")), stubDecode, 64);
const emAtlas = await buildAtlas(baked.flatMap((m) => collectModelImages(m, "emissive")), stubDecode, 64);
for (const m of baked) applyAtlas(m, baseAtlas.placements, emAtlas.placements, baseAtlas.cell);
const pool = buildModelPool(baked);
const poolBuffer = device.createBuffer({ size: pool.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
device.queue.writeBuffer(poolBuffer, 0, pool);
const tex = device.createTexture({ size: [W, H], format: FMT, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
const camera = new OrbitCamera({ addEventListener() {}, setPointerCapture() {}, releasePointerCapture() {} }, [params.worldX, params.worldY, params.worldZ]);
const renderer = createRenderer({ device, context: { getCurrentTexture: () => tex }, format: FMT, shaderParts: parts, world, models: { poolBuffer, boid: baked[0], predator: baked[1], obstacle: baked[2], atlasView: createAtlasTexture(device, baseAtlas), emissiveView: createAtlasTexture(device, emAtlas), emissiveStrength: true } });
async function grab(post) {
  renderer.updateCamera(camera.update(W, H));
  renderer.updateModels({ boidScale: params.boidModelScale, predatorScale: params.predatorModelScale, boidYaw: 0, boidPitch: params.boidModelPitch, predatorYaw: params.predatorModelYaw, predatorPitch: params.predatorModelPitch, glow: params.modelGlow });
  renderer.render({ boids: 0, predators: 1, food: 0, obstacles: 0 }, { heading: false, vision: false, hitbox: false, bbox: false, hashCells: false }, W, H, post);
  const rb = device.createBuffer({ size: bpr * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const e = device.createCommandEncoder();
  e.copyTextureToBuffer({ texture: tex }, { buffer: rb, bytesPerRow: bpr }, [W, H]);
  device.queue.submit([e.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const px = new Uint8Array(rb.getMappedRange().slice(0)); rb.unmap(); rb.destroy();
  return px;
}
function luma(px, x, y) { const o = y * bpr + x * 4; return px[o + 2] * 0.3 + px[o + 1] * 0.6 + px[o] * 0.1; }
// Horizontal and vertical profiles through the image centre.
function profile(px) {
  const cy = Math.floor(H / 2), cx = Math.floor(W / 2);
  const hx = [], vy = [];
  for (let d = 0; d <= 80; d += 4) {
    hx.push(+(luma(px, Math.min(W - 1, cx + d), cy)).toFixed(1));
    vy.push(+(luma(px, cx, Math.min(H - 1, cy + d))).toFixed(1));
  }
  return { hx, vy };
}
for (const r of [0.5, 1, 2, 4]) {
  const px = await grab({ bloom: true, threshold: params.bloomThreshold, strength: params.bloomStrength, radius: r });
  const p = profile(px);
  console.log(`r=${r} | H: ${p.hx.join(",")}`);
  console.log(`     | V: ${p.vy.join(",")}`);
}
const nb = await grab({ bloom: false });
const p = profile(nb);
console.log(`no-bloom | H: ${p.hx.join(",")}`);
console.log(`         | V: ${p.vy.join(",")}`);
