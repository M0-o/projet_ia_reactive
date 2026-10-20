// hashcells.wgsl
// Debug overlay: outlines the spatial-hash cells that contain at least one
// boid, so you can watch the uniform grid adapt to the flock. Reads the
// per-cell population counts written by the counting sort (the first
// `dims.cells` words of the grid buffer).
//
// One instance per cell, 24 vertices each (a box wireframe). Empty cells are
// culled to a degenerate clip position.

struct HOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) color : vec3<f32>,
};

const EDGE_IDX: array<u32, 24> = array<u32, 24>(
  0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u,
  0u, 2u, 1u, 3u, 4u, 6u, 5u, 7u,
  0u, 4u, 1u, 5u, 2u, 6u, 3u, 7u
);

@vertex
fn vsHashCells(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> HOut {
  var o: HOut;
  let dim = vec3<u32>(u32(P.grid.y), u32(P.grid.z), u32(P.grid.w));
  let cells = dim.x * dim.y * dim.z;
  if (ii >= cells) {
    o.clip = cullClip();
    o.color = vec3<f32>(0.0);
    return o;
  }
  let count = grid[ii];
  if (count == 0u) {
    o.clip = cullClip();
    o.color = vec3<f32>(0.0);
    return o;
  }
  // Decode cell coordinates and its world-space box.
  let cx = ii % dim.x;
  let cy = (ii / dim.x) % dim.y;
  let cz = ii / (dim.x * dim.y);
  let cellSize = P.grid.x;
  let minB = P.boundsMin.xyz + vec3<f32>(f32(cx), f32(cy), f32(cz)) * cellSize;
  let edge = EDGE_IDX[vi];
  let local = vec3<f32>(f32(edge & 1u), f32((edge >> 1u) & 1u), f32((edge >> 2u) & 1u));
  let world = minB + local * cellSize;
  o.clip = cam.viewProj * vec4<f32>(world, 1.0);
  // Brighter where cells are more crowded (green -> blue).
  let t = clamp(f32(count) / 6.0, 0.0, 1.0);
  o.color = mix(vec3<f32>(0.20, 0.95, 0.45), vec3<f32>(0.30, 0.65, 1.0), t);
  return o;
}

@fragment
fn fsHashCells(i: HOut) -> @location(0) vec4<f32> {
  return vec4<f32>(i.color, 0.28);
}
