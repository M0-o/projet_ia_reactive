// Temporary diagnostic: dump the intermediate bloom textures as raw and inspect
// their variance, to isolate which blur pass loses the emitter.
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

const W = 320, H = 240, FMT = "rgba16float";
const gpu = create([]);
const device = await (await gpu.requestAdapter()).requestDevice();
const parts = await loadShaderParts((p) => readFile(join(root, p), "utf8"));
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
const CANVAS_FMT = "bgra8unorm";
const tex = device.createTexture({ size: [W, H], format: CANVAS_FMT, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
const camera = new OrbitCamera({ addEventListener() {}, setPointerCapture() {}, releasePointerCapture() {} }, [params.worldX, params.worldY, params.worldZ]);
const renderer = createRenderer({ device, context: { getCurrentTexture: () => tex }, format: CANVAS_FMT, shaderParts: parts, world, models: { poolBuffer, boid: baked[0], predator: baked[1], obstacle: baked[2], atlasView: createAtlasTexture(device, baseAtlas), emissiveView: createAtlasTexture(device, emAtlas), emissiveStrength: true } });
renderer.updateCamera(camera.update(W, H));
renderer.updateModels({ boidScale: params.boidModelScale, predatorScale: params.predatorModelScale, boidYaw: 0, boidPitch: params.boidModelPitch, predatorYaw: params.predatorModelYaw, predatorPitch: params.predatorModelPitch, glow: params.modelGlow });
renderer.render({ boids: 0, predators: 1, food: 0, obstacles: 0 }, { heading: false, vision: false, hitbox: false, bbox: false, hashCells: false }, W, H, { bloom: true, threshold: params.bloomThreshold, strength: params.bloomStrength, radius: 4 });
// Read back the HDR scene and both bloom ping-pong textures.
const bw = Math.ceil(W / 2), bh = Math.ceil(H / 2);
async function readTex(t, w, h) {
  const bpr = Math.ceil(w * 8 / 256) * 256;
  const rb = device.createBuffer({ size: bpr * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const e = device.createCommandEncoder();
  e.copyTextureToBuffer({ texture: t }, { buffer: rb, bytesPerRow: bpr }, [w, h]);
  device.queue.submit([e.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const u16 = new Uint16Array(rb.getMappedRange().slice(0));
  rb.unmap(); rb.destroy();
  const out = new Float32Array(w * h * 3);
  for (let i = 0; i < w * h; i++) {
    for (let c = 0; c < 3; c++) {
      const hh = u16[i * 4 + c];
      const s = (hh >> 15) & 1, ex = (hh >> 10) & 0x1f, m = hh & 0x3ff;
      let v;
      if (ex === 0) v = (m / 1024) * Math.pow(2, -14);
      else if (ex === 31) v = m ? NaN : Infinity;
      else v = (1 + m / 1024) * Math.pow(2, ex - 15);
      out[i * 3 + c] = s ? -v : v;
    }
  }
  return { w, h, data: out };
}
const hdr = renderer.hdrTexture();
const { a, b } = renderer.bloomTextures();
const H_ = await readTex(hdr, W, H);
const A = await readTex(a, bw, bh);
const B = await readTex(b, bw, bh);
function summarize(name, img) {
  let sum = 0, max = 0, nz = 0;
  for (let i = 0; i < img.w * img.h; i++) {
    const l = img.data[i * 3] * 0.2126 + img.data[i * 3 + 1] * 0.7152 + img.data[i * 3 + 2] * 0.0722;
    sum += l; if (l > max) max = l; if (l > 0.05) nz++;
  }
  console.log(`${name}: mean ${(sum / (img.w * img.h)).toFixed(4)} max ${max.toFixed(3)} pixels>0.05 ${nz}/${img.w * img.h}`);
  // Row/column profile through the brightest pixel.
  let bx = 0, by = 0, best = -1;
  for (let y = 0; y < img.h; y++) for (let x = 0; x < img.w; x++) {
    const l = img.data[(y * img.w + x) * 3] * 0.2126 + img.data[(y * img.w + x) * 3 + 1] * 0.7152 + img.data[(y * img.w + x) * 3 + 2] * 0.0722;
    if (l > best) { best = l; bx = x; by = y; }
  }
  const row = [], col = [];
  for (let d = 0; d <= 60; d += 5) {
    const lr = img.data[(by * img.w + Math.min(img.w - 1, bx + d)) * 3] * 0.2126 + img.data[(by * img.w + Math.min(img.w - 1, bx + d)) * 3 + 1] * 0.7152 + img.data[(by * img.w + Math.min(img.w - 1, bx + d)) * 3 + 2] * 0.0722;
    const lc = img.data[(Math.min(img.h - 1, by + d) * img.w + bx) * 3] * 0.2126 + img.data[(Math.min(img.h - 1, by + d) * img.w + bx) * 3 + 1] * 0.7152 + img.data[(Math.min(img.h - 1, by + d) * img.w + bx) * 3 + 2] * 0.0722;
    row.push(+lr.toFixed(3)); col.push(+lc.toFixed(3));
  }
  console.log(`   brightest (${bx},${by}) luma ${best.toFixed(3)}  +X: ${row.join(",")}  +Y: ${col.join(",")}`);
}
summarize("HDR scene ", H_);
summarize("bloom A   ", A);
summarize("bloom B   ", B);
