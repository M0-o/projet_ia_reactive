// box.wgsl
// The transparent bounding box of the world. A single line-list pipeline:
//   vsBoxEdges / fsBoxEdges -> 24 vertices
// Neutral light-grey edges so the box reads as a wireframe volume and does not
// tint the scene blue. Derived from P.boundsMin/P.boundsSize.

const EDGE_IDX: array<u32, 24> = array<u32, 24>(
  0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u,
  0u, 2u, 1u, 3u, 4u, 6u, 5u, 7u,
  0u, 4u, 1u, 5u, 2u, 6u, 3u, 7u
);

fn corner(i: u32) -> vec3<f32> {
  let minB = P.boundsMin.xyz;
  let size = P.boundsSize.xyz;
  return minB + vec3<f32>(f32(i & 1u), f32((i >> 1u) & 1u), f32((i >> 2u) & 1u)) * size;
}

struct BOut {
  @builtin(position) clip  : vec4<f32>,
  @location(0) color : vec3<f32>,
};

@vertex
fn vsBoxEdges(@builtin(vertex_index) vi: u32) -> BOut {
  var o: BOut;
  o.clip = cam.viewProj * vec4<f32>(corner(EDGE_IDX[vi]), 1.0);
  o.color = vec3<f32>(0.72, 0.75, 0.80);
  return o;
}

@fragment
fn fsBoxEdges(i: BOut) -> @location(0) vec4<f32> {
  return vec4<f32>(i.color, 0.35);
}

