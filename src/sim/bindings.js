// bindings.js
// Buffer sizes/strides and the explicit bind group layouts. Must stay in sync
// with shaders/common.wgsl (structs + O_* offsets) and the *binding*.wgsl files.

export const AGENT_STRIDE = 48; // 3 x vec4
export const POINT_STRIDE = 32; // 2 x vec4

// Maximum counts; runtime counts are only the first N entries of each buffer,
// so tuning population never reallocates memory.
export const CAPACITY = {
  boids: 4096,
  predators: 128,
  food: 160,
  obstacles: 128,
};

export const MAX_CELLS = 65536;
export const SCAN_BLOCKS = 256; // MAX_CELLS / 256
// 3 * MAX_CELLS (counts, starts, cursor) + SCAN_BLOCKS (partial) + BOID_CAP + 8
export const GRID_WORDS = 3 * MAX_CELLS + SCAN_BLOCKS + CAPACITY.boids + 8;

// Model pool: 3 models (boid, predator, obstacle). Size is grown at load time
// to fit the baked geometry + joint frames; this is just a safe starting size.
export const MODEL_POOL_BYTES = 8 * 1024 * 1024;
// 4 models x (vec4<u32> + 2 x vec4<f32>) = 4 x 48 bytes.
export const MODEL_INFO_BYTES = 256;

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
const UNIFORM = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

export function computeLayout(device) {
  return device.createBindGroupLayout({
    label: "compute-layout",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });
}

export function renderLayout(device) {
  const VERTEX = GPUShaderStage.VERTEX;
  const FRAGMENT = GPUShaderStage.FRAGMENT;
  return device.createBindGroupLayout({
    label: "render-layout",
    entries: [
      { binding: 0, visibility: VERTEX | FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: VERTEX | FRAGMENT, buffer: { type: "uniform" } },
      { binding: 2, visibility: VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 5, visibility: VERTEX, buffer: { type: "read-only-storage" } },
      // 6: packed model geometry + baked joint frames.
      { binding: 6, visibility: VERTEX, buffer: { type: "read-only-storage" } },
      // 7: per-model metadata (uniform array of ModelInfo).
      { binding: 7, visibility: VERTEX, buffer: { type: "uniform" } },
      // 8: spatial-hash cells (read by the debug overlay only).
      { binding: 8, visibility: VERTEX, buffer: { type: "read-only-storage" } },
      // 9/10: the base-colour texture atlas shared by every model.
      { binding: 9, visibility: FRAGMENT, texture: { sampleType: "float" } },
      { binding: 10, visibility: FRAGMENT, sampler: { type: "filtering" } },
      // 11/12: the emissive ("glow") atlas used by the predator materials.
      { binding: 11, visibility: FRAGMENT, texture: { sampleType: "float" } },
      { binding: 12, visibility: FRAGMENT, sampler: { type: "filtering" } },
    ],
  });
}

export { STORAGE, UNIFORM };
