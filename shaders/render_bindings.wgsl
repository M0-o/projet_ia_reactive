// render_bindings.wgsl
// Bindings shared by every render pipeline.
// Binding indices must match src/sim/bindings.js (renderLayout).

struct Camera {
  viewProj   : mat4x4<f32>,
  view       : mat4x4<f32>,
  invViewProj: mat4x4<f32>,
  camPos     : vec4<f32>, // xyz = eye position
  viewport   : vec4<f32>, // width, height, near, far
};

// Per-model skinning metadata, uploaded once when a GLB is loaded.
// All `mesh` offsets are absolute word offsets into the shared pool.
//   mesh  = (vertexOffset, vertexCount, indexOffset, jointOffset)
//   info  = (sizeMultiplier, animLength, jointCount, frameCount)
//   extra = (yawOffset, glowStrength, 0, 0)
struct ModelInfo {
  mesh  : vec4<u32>,
  info  : vec4<f32>,
  extra : vec4<f32>,
};

struct ModelSet {
  models : array<ModelInfo, 4>,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> cam: Camera;
@group(0) @binding(2) var<storage, read> boids: array<Agent>;
@group(0) @binding(3) var<storage, read> predators: array<Agent>;
@group(0) @binding(4) var<storage, read> food: array<Point>;
@group(0) @binding(5) var<storage, read> obstacles: array<Point>;
// All model geometry packed back to back: per-vertex positions, normals, uvs
// and skin data (vec4 slots), then indices (u32) and joint matrices (f32).
@group(0) @binding(6) var<storage, read> modelPool: array<f32>;
@group(0) @binding(7) var<uniform> modelSet: ModelSet;
// Spatial-hash scratch (per-cell population counts live at its start). Only
// the debug overlay reads this; the simulation writes it in the compute pass.
@group(0) @binding(8) var<storage, read> grid: array<u32>;
// Base-colour texture atlas built from the GLB images (src/render/textures.js).
@group(0) @binding(9) var atlasTex: texture_2d<f32>;
@group(0) @binding(10) var atlasSmp: sampler;
// Emissive ("glow") atlas; black where a material does not glow.
@group(0) @binding(11) var emissiveTex: texture_2d<f32>;
@group(0) @binding(12) var emissiveSmp: sampler;

// Deterministic per-instance animation phase, so a flock does not flap in
// perfect unison. `i` is the instance (boid/predator) index.
fn animPhase(i: u32) -> f32 {
  return hash1(i * 2654435761u + 17u);
}

const BG_COLOR: vec3<f32> = vec3<f32>(0.010, 0.014, 0.032);

// Slice out a vec4 from the packed model pool.
fn poolVec4(wordOffset: u32) -> vec4<f32> {
  return vec4<f32>(modelPool[wordOffset], modelPool[wordOffset + 1u],
                   modelPool[wordOffset + 2u], modelPool[wordOffset + 3u]);
}

// Joint matrix `j` of model `m` (mat4x4 stored as 16 consecutive f32).
fn jointMatrix(m: ModelInfo, j: u32) -> mat4x4<f32> {
  let base = m.mesh.z + j * 16u;
  var mat: mat4x4<f32>;
  for (var c = 0u; c < 4u; c = c + 1u) {
    mat[c] = poolVec4(base + c * 4u);
  }
  return mat;
}
