// Temporary diagnostic: run ONLY the blur pass on a synthetic impulse and
// print the resulting row, to see the real kernel shape.
import { create, globals } from "webgpu";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
Object.assign(globalThis, globals);
const root = process.cwd();
const { loadShaderParts, composePost } = await import(join(root, "src/sim/compose.js"));
const gpu = create([]);
const device = await (await gpu.requestAdapter()).requestDevice();
const parts = await loadShaderParts((p) => readFile(join(root, p), "utf8"));
const code = composePost(parts, "bloom");
const mod = device.createShaderModule({ code });

const W = 160, H = 120;
const fmt = "rgba16float";
const tex = (label) => device.createTexture({ label, size: [W, H], format: fmt, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
const src = tex("src"), dst = tex("dst"), other = tex("other");

const smp = device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" });
const layout = device.createBindGroupLayout({ entries: [
  { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
  { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
  { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
  { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
]});
const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
const blurPipe = device.createRenderPipeline({ layout: pl, vertex: { module: mod, entryPoint: "vsFullscreen" }, fragment: { module: mod, entryPoint: "fsBlur", targets: [{ format: fmt }] }, primitive: { topology: "triangle-list" } });

// Impulse generator: a fragment shader that lights a single texel.
const genMod = device.createShaderModule({ code: `
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var pos = array<vec2<f32>, 3>(vec2<f32>(-1.0,-1.0), vec2<f32>(3.0,-1.0), vec2<f32>(-1.0,3.0));
  let p = pos[vi];
  var o: VOut;
  o.clip = vec4<f32>(p, 0.0, 1.0);
  o.uv = vec2<f32>(p.x*0.5+0.5, 0.5-p.y*0.5);
  return o;
}
@fragment fn fs(i: VOut) -> @location(0) vec4<f32> {
  let px = vec2<i32>(i.uv * vec2<f32>(${W}.0, ${H}.0));
  if (px.x == 80 && px.y == 60) { return vec4<f32>(8.0, 8.0, 8.0, 1.0); }
  return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}` });
const genPipe = device.createRenderPipeline({ layout: pl, vertex: { module: genMod, entryPoint: "vs" }, fragment: { module: genMod, entryPoint: "fs", targets: [{ format: fmt }] }, primitive: { topology: "triangle-list" } });

const uniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
const bgFor = (buf, s, o) => device.createBindGroup({ layout, entries: [
  { binding: 0, resource: { buffer: buf } }, { binding: 1, resource: s.createView() },
  { binding: 2, resource: o.createView() }, { binding: 3, resource: smp }] });

async function run(radius, step) {
  device.queue.writeBuffer(uniform, 0, new Float32Array([0, 1, radius, step]));
  // 1. generate the impulse into src
  {
    const e = device.createCommandEncoder();
    const p = e.beginRenderPass({ colorAttachments: [{ view: other.createView(), loadOp: "clear", clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: "store" }] });
    p.setPipeline(genPipe); p.setBindGroup(0, bgFor(uniform, dst, src)); p.draw(3); p.end();
    device.queue.submit([e.finish()]);
  }
  // 2. blur src -> dst
  {
    const e = device.createCommandEncoder();
    const p = e.beginRenderPass({ colorAttachments: [{ view: dst.createView(), loadOp: "clear", clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: "store" }] });
    p.setPipeline(blurPipe); p.setBindGroup(0, bgFor(uniform, other, src)); p.draw(3); p.end();
    device.queue.submit([e.finish()]);
  }
  const bpr = Math.ceil(W * 8 / 256) * 256;
  const rb = device.createBuffer({ size: bpr * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const e = device.createCommandEncoder(); e.copyTextureToBuffer({ texture: dst }, { buffer: rb, bytesPerRow: bpr }, [W, H]); device.queue.submit([e.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const u16 = new Uint16Array(rb.getMappedRange().slice(0)); rb.unmap(); rb.destroy();
  const half = (h) => { const s = (h >> 15) & 1, ex = (h >> 10) & 0x1f, m = h & 0x3ff;
    let v = ex === 0 ? (m / 1024) * Math.pow(2, -14) : (ex === 31 ? NaN : (1 + m / 1024) * Math.pow(2, ex - 15));
    return s ? -v : v; };
  const row = [], col = [];
  for (let x = 66; x <= 94; x++) row.push(half(u16[(60 * W + x) * 4]).toFixed(2));
  for (let y = 46; y <= 74; y++) col.push(half(u16[(y * W + 80) * 4]).toFixed(2));
  console.log(`radius=${radius} step=${step.toFixed(5)}`);
  console.log(`  +X: ${row.join(" ")}`);
  console.log(`  +Y: ${col.join(" ")}`);
}
await run(1, 1 / W);
await run(4, 1 / W);
await run(1, 0);
