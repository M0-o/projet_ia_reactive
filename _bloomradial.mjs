// Temporary bloom diagnostic: radial falloff around the brightest pixel, to
// confirm the glow is a wide haze rather than a tight streak.
import { create, globals } from "webgpu";
import { readFile, writeFile } from "node:fs/promises";
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
const params = { ...makeParams(), boidCount: 40, predatorCount: 3, foodCount: 0, obstacleCount: 2 };
const world = createWorld({ device, shaderParts: parts, params });
world.setParams(params); world.reset(99);
for (let f = 0; f < 20; f++) world.step(1 / 60, f / 60, f);

const modelBuf = async (n) => {
  const b = await readFile(join(root, "3D-models", n + ".glb"));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};
const stubDecode = async () => ({ width: 4, height: 4, data: new Uint8Array(64).fill(180) });
const boid = await bakeModel(await modelBuf("boid"), { keyPrefix: "boid" });
const predator = await bakeModel(await modelBuf("predator"), { keyPrefix: "predator" });
const obstacle = await bakeModel(await modelBuf("obstacle"), { keyPrefix: "obstacle" });
const baked = [boid, predator, obstacle];
const baseAtlas = await buildAtlas(baked.flatMap((m) => collectModelImages(m, "base")), stubDecode, 64);
const emAtlas = await buildAtlas(baked.flatMap((m) => collectModelImages(m, "emissive")), stubDecode, 64);
for (const m of baked) applyAtlas(m, baseAtlas.placements, emAtlas.placements, baseAtlas.cell);
const pool = buildModelPool(baked);
const poolBuffer = device.createBuffer({ size: pool.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
device.queue.writeBuffer(poolBuffer, 0, pool);
const RENDER_FORMAT = "bgra8unorm";
const tex = device.createTexture({ size: [W, H], format: RENDER_FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
const camera = new OrbitCamera({ addEventListener() {}, setPointerCapture() {}, releasePointerCapture() {} }, [params.worldX, params.worldY, params.worldZ]);
const renderer = createRenderer({ device, context: { getCurrentTexture: () => tex }, format: RENDER_FORMAT, shaderParts: parts, world, models: { poolBuffer, boid, predator, obstacle, atlasView: createAtlasTexture(device, baseAtlas), emissiveView: createAtlasTexture(device, emAtlas), emissiveStrength: true } });

const bytesPerRow = Math.ceil(W * 4 / 256) * 256;
async function grab(post) {
  renderer.updateCamera(camera.update(W, H));
  renderer.updateModels({ boidScale: params.boidModelScale, predatorScale: params.predatorModelScale, boidYaw: 0, boidPitch: params.boidModelPitch, predatorYaw: params.predatorModelYaw, predatorPitch: params.predatorModelPitch, glow: params.modelGlow });
  renderer.render({ boids: 40, predators: 3, food: 0, obstacles: 2 }, { heading: false, vision: false, hitbox: false, bbox: false, hashCells: false }, W, H, post);
  const rb = device.createBuffer({ size: bytesPerRow * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const e = device.createCommandEncoder();
  e.copyTextureToBuffer({ texture: tex }, { buffer: rb, bytesPerRow }, [W, H]);
  device.queue.submit([e.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const px = new Uint8Array(rb.getMappedRange().slice(0)); rb.unmap(); rb.destroy();
  return { px, bytesPerRow };
}
function lumaAt(px, bpr, x, y) {
  const o = y * bpr + x * 4;
  return px[o + 2] * 0.3 + px[o + 1] * 0.6 + px[o] * 0.1;
}
function brightest(px, bpr) {
  let bx = 0, by = 0, best = -1;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const l = lumaAt(px, bpr, x, y);
    if (l > best) { best = l; bx = x; by = y; }
  }
  return { bx, by, best };
}
function radial(px, bpr, cx, cy) {
  const out = [];
  for (const r of [0, 2, 4, 8, 16, 24, 32, 48, 64]) {
    let s = 0, n = 0;
    for (let a = 0; a < 72; a++) {
      const th = (a / 72) * Math.PI * 2;
      const x = Math.round(cx + Math.cos(th) * r), y = Math.round(cy + Math.sin(th) * r);
      if (x < 0 || y < 0 || x >= W || y >= H) continue;
      s += lumaAt(px, bpr, x, y); n++;
    }
    out.push([r, n ? +(s / n).toFixed(1) : null]);
  }
  return out;
}
for (const r of [0.5, 4, 8]) {
  const { px, bytesPerRow: bpr } = await grab({ bloom: true, threshold: params.bloomThreshold, strength: params.bloomStrength, radius: r });
  const b = brightest(px, bpr);
  console.log(`radius ${r}: brightest luma ${b.best.toFixed(0)} at (${b.bx},${b.by}); radial`, JSON.stringify(radial(px, bpr, b.bx, b.by)));
  await writeFile(`/tmp/opencode/bloom-mist-r${r}.png.raw`, Buffer.from(px));
}
const { px, bytesPerRow: bpr } = await grab({ bloom: false });
const b = brightest(px, bpr);
console.log(`no bloom: brightest luma ${b.best.toFixed(0)}; radial`, JSON.stringify(radial(px, bpr, b.bx, b.by)));
