// food.wgsl
// Camera-facing glowing sprites for food clusters. One instanced draw call.
// Blended additively on top of the scene.

struct FOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) uv    : vec2<f32>,
  @location(1) color : vec3<f32>,
  @location(2) world : vec3<f32>,
};

fn quadCorner(vi: u32) -> vec2<f32> {
  if (vi == 0u) { return vec2<f32>(-1.0, -1.0); }
  if (vi == 1u) { return vec2<f32>( 1.0, -1.0); }
  if (vi == 2u) { return vec2<f32>( 1.0,  1.0); }
  if (vi == 3u) { return vec2<f32>(-1.0, -1.0); }
  if (vi == 4u) { return vec2<f32>( 1.0,  1.0); }
  return vec2<f32>(-1.0, 1.0);
}

@vertex
fn vsFood(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> FOut {
  var o: FOut;
  let f = food[ii];
  if (f.extra.x < 0.5) {
    o.clip = cullClip();
    o.uv = vec2<f32>(0.0);
    o.color = vec3<f32>(0.0);
    o.world = vec3<f32>(0.0);
    return o;
  }
  let radius = max(f.posR.w, 0.5) * 2.6;
  let q = quadCorner(vi);
  let world = f.posR.xyz + (cameraRight() * q.x + cameraUp() * q.y) * radius;
  o.clip = cam.viewProj * vec4<f32>(world, 1.0);
  o.uv = q;
  o.color = mix(vec3<f32>(0.30, 0.95, 0.40), vec3<f32>(0.95, 1.0, 0.45), clamp(f.extra.w / 100.0, 0.0, 1.0));
  o.world = world;
  return o;
}

@fragment
fn fsFood(i: FOut) -> @location(0) vec4<f32> {
  let d = length(i.uv);
  if (d > 1.0) { discard; }
  let core = 1.0 - smoothstep(0.0, 1.0, d);
  let glow = core * core;
  let c = i.color * (0.35 + 1.65 * glow);
  return vec4<f32>(c, glow * 0.9);
}
