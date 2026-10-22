// Temporary diagnostic: time a single bloomed render at 400x300 to see whether
// the blur tap count is what makes Dawn's software backend hang.
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
device.lost.then((i) => console.log("!! DEVICE LOST:", i.reason, i.message));
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
async function grab(post) {
  const t0 = Date.now();
  renderer.updateCamera(camera.update(W, H));
  renderer.updateModels({ boidScale: params.boidModelScale, predatorScale: params.predatorModelScale, boidYaw: 0, boidPitch: params.boidModelPitch, predatorYaw: params.predatorModelYaw, predatorPitch: params.predatorModelPitch, glow: params.modelGlow });
  renderer.render({ boids: 0, predators: 3, food: 0, obstacles: 0 }, { heading: false, vision: false, hitbox: false, bbox: false, hashCells: false }, W, H, post);
  const rb = device.createBuffer({ size: bpr * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const e = device.createCommandEncoder();
  e.copyTextureToBuffer({ texture: tex }, { buffer: rb, bytesPerRow: bpr }, [W, H]);
  device.queue.submit([e.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const px = new Uint8Array(rb.getMappedRange().slice(0));
  rb.unmap(); rb.destroy();
  console.log(`  render took ${Date.now() - t0} ms`);
  return px;
}
for (let i = 0; i < 4; i++) {
  console.log("iteration", i);
  await grab({ bloom: true, threshold: params.bloomThreshold, strength: params.bloomStrength, radius: [0.5, 1, 2, 4][i] });
}
console.log("done");
