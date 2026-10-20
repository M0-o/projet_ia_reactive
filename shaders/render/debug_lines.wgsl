// debug_lines.wgsl
// Independently-toggleable debug overlays. Every entry point produces a
// line-list; the pipeline chooses which one to draw. Nothing here is drawn
// unless its toggle is enabled (see src/ui/panel.js + src/render/renderer.js).

struct DOut {
  @builtin(position) clip  : vec4<f32>,
  @location(0) color : vec3<f32>,
};

const EDGE_IDX: array<u32, 24> = array<u32, 24>(
  0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u,
  0u, 2u, 1u, 3u, 4u, 6u, 5u, 7u,
  0u, 4u, 1u, 5u, 2u, 6u, 3u, 7u
);

const HEADING_COLOR_BOID: vec3<f32> = vec3<f32>(0.20, 0.90, 1.00);
const HEADING_COLOR_PRED: vec3<f32> = vec3<f32>(1.00, 0.40, 0.90);
const VISION_COLOR_BOID: vec3<f32> = vec3<f32>(0.25, 0.55, 1.00);
const VISION_COLOR_PRED: vec3<f32> = vec3<f32>(1.00, 0.30, 0.50);
const HITBOX_COLOR: vec3<f32> = vec3<f32>(1.00, 0.70, 0.10);
const BBOX_COLOR: vec3<f32> = vec3<f32>(0.55, 1.00, 0.35);

fn deg(clip: vec4<f32>, color: vec3<f32>) -> DOut {
  var o: DOut;
  o.clip = clip;
  o.color = color;
  return o;
}

fn headingVS(vi: u32, a: Agent, color: vec3<f32>) -> DOut {
  if (a.info.x < 0.5) { return deg(cullClip(), color); }
  var p = a.posH.xyz;
  if (vi == 1u) { p = p + a.velD.xyz * max(P.debug.x, 0.0); }
  return deg(cam.viewProj * vec4<f32>(p, 1.0), color);
}

fn ringVS(vi: u32, center: vec3<f32>, radius: f32, segs: u32, color: vec3<f32>) -> DOut {
  let s = vi / 2u;
  let e = vi % 2u;
  let ang = (f32(s) + f32(e)) * TAU / f32(segs);
  let p = center + (cameraRight() * cos(ang) + cameraUp() * sin(ang)) * radius;
  return deg(cam.viewProj * vec4<f32>(p, 1.0), color);
}

fn bboxVS(vi: u32, center: vec3<f32>, half: vec3<f32>, color: vec3<f32>) -> DOut {
  let i = EDGE_IDX[vi];
  let local = vec3<f32>(f32(i & 1u) * 2.0 - 1.0, f32((i >> 1u) & 1u) * 2.0 - 1.0, f32((i >> 2u) & 1u) * 2.0 - 1.0);
  return deg(cam.viewProj * vec4<f32>(center + local * half, 1.0), color);
}

const RING_SEGS: u32 = 48u;
const HIT_SEGS: u32 = 16u;

@vertex
fn vsHeadingBoids(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> DOut {
  return headingVS(vi, boids[ii], HEADING_COLOR_BOID);
}
@vertex
fn vsHeadingPredators(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> DOut {
  return headingVS(vi, predators[ii], HEADING_COLOR_PRED);
}
@vertex
fn vsVisionBoids(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> DOut {
  let a = boids[ii];
  if (a.info.x < 0.5) { return deg(cullClip(), VISION_COLOR_BOID); }
  return ringVS(vi, a.posH.xyz, P.motion.z * max(P.debug.z, 0.0), RING_SEGS, VISION_COLOR_BOID);
}
@vertex
fn vsVisionPredators(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> DOut {
  let a = predators[ii];
  if (a.info.x < 0.5) { return deg(cullClip(), VISION_COLOR_PRED); }
  return ringVS(vi, a.posH.xyz, P.predator.z * max(P.debug.z, 0.0), RING_SEGS, VISION_COLOR_PRED);
}
@vertex
fn vsHitboxBoids(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> DOut {
  let a = boids[ii];
  if (a.info.x < 0.5) { return deg(cullClip(), HITBOX_COLOR); }
  return ringVS(vi, a.posH.xyz, a.info.w * max(P.debug.w, 0.0), HIT_SEGS, HITBOX_COLOR);
}
@vertex
fn vsHitboxPredators(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> DOut {
  let a = predators[ii];
  if (a.info.x < 0.5) { return deg(cullClip(), HITBOX_COLOR); }
  return ringVS(vi, a.posH.xyz, a.info.w * max(P.debug.w, 0.0), HIT_SEGS, HITBOX_COLOR);
}
@vertex
fn vsHitboxObstacles(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> DOut {
  let p = obstacles[ii];
  if (p.extra.x < 0.5) { return deg(cullClip(), HITBOX_COLOR); }
  return ringVS(vi, p.posR.xyz, p.posR.w * max(P.debug.w, 0.0), HIT_SEGS, HITBOX_COLOR);
}
@vertex
fn vsBBoxBoids(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> DOut {
  let a = boids[ii];
  if (a.info.x < 0.5) { return deg(cullClip(), BBOX_COLOR); }
  let half = vec3<f32>(a.info.w * max(P.debug.y, 0.0));
  return bboxVS(vi, a.posH.xyz, half, BBOX_COLOR);
}
@vertex
fn vsBBoxPredators(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> DOut {
  let a = predators[ii];
  if (a.info.x < 0.5) { return deg(cullClip(), BBOX_COLOR); }
  let half = vec3<f32>(a.info.w * max(P.debug.y, 0.0));
  return bboxVS(vi, a.posH.xyz, half, BBOX_COLOR);
}

@fragment
fn fsLine(i: DOut) -> @location(0) vec4<f32> {
  return vec4<f32>(i.color, 0.85);
}
