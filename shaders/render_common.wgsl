// render_common.wgsl
// Shared render helpers. Composition order:
//   common.wgsl + render_bindings.wgsl + render_common.wgsl + <module>.wgsl

// Night-sky background: a vertical gradient from deep navy at the horizon to
// near-black at the zenith, plus a deterministic field of stars. Used as the
// scene clear colour by the full-screen sky pass. `dir` is a normalized world
// direction from the camera.
fn skyColor(dir: vec3<f32>) -> vec3<f32> {
  let up = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
  let horizon = vec3<f32>(0.055, 0.075, 0.135);
  let zenith = vec3<f32>(0.008, 0.012, 0.030);
  var c = mix(horizon, zenith, smoothstep(0.25, 1.0, up));
  // Faint warm glow near the horizon so the void reads as sky, not a flat fill.
  c = c + vec3<f32>(0.020, 0.012, 0.006) * pow(1.0 - up, 6.0);
  return c;
}

// Star field. Projects the direction onto a coarse lattice and lights one
// pseudo-random cell per cell. Brightness varies; a few stars twinkle with a
// slow sine so the sky is not static.
fn starField(dir: vec3<f32>, time: f32) -> vec3<f32> {
  // Two layers at different densities.
  var total = vec3<f32>(0.0);
  for (var layer = 0; layer < 2; layer = layer + 1) {
    let scale = select(90.0, 170.0, layer == 1);
    let p = dir * scale;
    let cell = floor(p);
    let f = fract(p) - 0.5;
    let h = hash3(cell);
    // Only ~12% of cells hold a star.
    if (h.x < 0.88) { continue; }
    // Jitter the star's position inside its cell.
    let jitter = (vec3<f32>(hash2(u32(cell.x) + 1u, u32(cell.y) + 2u),
                            hash2(u32(cell.z) + 3u, u32(cell.x) + 4u),
                            hash2(u32(cell.y) + 5u, u32(cell.z) + 6u)) - 0.5) * 0.7;
    let d = length(f - jitter);
    let core = smoothstep(0.16, 0.0, d);
    let twinkle = 0.65 + 0.35 * sin(time * 2.1 + h.y * 41.0 + h.z * 17.0);
    let bright = (0.35 + 0.65 * h.z) * twinkle * select(1.0, 0.6, layer == 1);
    // Slight colour temperature variation.
    let tint = mix(vec3<f32>(0.75, 0.85, 1.0), vec3<f32>(1.0, 0.9, 0.75), h.y);
    total = total + tint * core * bright;
  }
  return total;
}

// Ambient light contributed by the star field (a cool, very dim fill) so that
// geometry is never pure black in shadow.
fn starAmbient(n: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
  let skyFill = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
  let cool = vec3<f32>(0.10, 0.13, 0.20);
  return cool * (0.25 + 0.75 * skyFill) + starField(dir, 0.0) * 0.35;
}

// Orthonormal basis that maps the model's local +Z onto `fwdIn` (the heading).
// Built as the *minimum rotation* taking world +Z onto the heading and applied
// to +X/+Y, so the roll reference varies continuously as the heading sweeps in
// any direction. The previous implementation hard-switched its reference up when
// |fwd.y| crossed 0.9, snapping the model through a visible roll flip; this has
// no thresholds at all (the only singularity is an exact 180 degree reversal,
// handled explicitly).
fn orthoBasis(fwdIn: vec3<f32>) -> mat3x3<f32> {
  let f = safeNormalize(fwdIn, vec3<f32>(0.0, 0.0, 1.0));
  let z = vec3<f32>(0.0, 0.0, 1.0);
  let c = cross(z, f);
  let d = dot(z, f);
  if (d < -0.9999) {
    // Heading is exactly -Z: rotate 180 degrees about X (any axis works; pick a
    // stable one). This single point cannot be continuous with its neighbours.
    return mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, -1.0, 0.0), vec3<f32>(0.0, 0.0, -1.0));
  }
  // Rodrigues with the shortest arc: v' = v + cross(c, v) + cross(c, cross(c, v)) / (1 + d).
  let k = 1.0 / (1.0 + d);
  let ex = vec3<f32>(1.0, 0.0, 0.0);
  let ey = vec3<f32>(0.0, 1.0, 0.0);
  let right = ex + cross(c, ex) + cross(c, cross(c, ex)) * k;
  let up = ey + cross(c, ey) + cross(c, cross(c, ey)) * k;
  return mat3x3<f32>(right, up, f);
}

fn cameraRight() -> vec3<f32> {
  return normalize(vec3<f32>(cam.view[0][0], cam.view[1][0], cam.view[2][0]));
}

fn cameraUp() -> vec3<f32> {
  return normalize(vec3<f32>(cam.view[0][1], cam.view[1][1], cam.view[2][1]));
}

// World-space direction the camera is looking (reconstructed from the inverse
// view-projection, used by the sky pass to pick a ray per pixel).
fn rayDir(ndc: vec2<f32>) -> vec3<f32> {
  let near = cam.invViewProj * vec4<f32>(ndc, 0.0, 1.0);
  let far  = cam.invViewProj * vec4<f32>(ndc, 1.0, 1.0);
  let n = near.xyz / near.w;
  let f = far.xyz / far.w;
  return normalize(f - n);
}

fn healthColor(h: f32) -> vec3<f32> {
  return mix(vec3<f32>(0.95, 0.12, 0.10), vec3<f32>(0.35, 1.0, 0.45), clamp(h / 100.0, 0.0, 1.0));
}

fn fogAmount(world: vec3<f32>) -> f32 {
  // Gentle distance haze only; the scene is viewed from a distance similar to
  // the world size, so a strong fog would wash every model into the night sky.
  return clamp((distance(cam.camPos.xyz, world) - 220.0) / 900.0, 0.0, 0.35);
}

fn shade(n: vec3<f32>, color: vec3<f32>, world: vec3<f32>) -> vec4<f32> {
  // Key light (cool, like moonlight) + star-driven ambient fill. A small
  // emissive term keeps models readable against the very dark night sky.
  let L = normalize(vec3<f32>(0.42, 0.80, 0.30));
  let diff = 0.45 + 0.75 * max(dot(n, L), 0.0);
  let viewDir = normalize(cam.camPos.xyz - world);
  let ambient = starAmbient(n, viewDir);
  let rim = pow(1.0 - max(dot(n, viewDir), 0.0), 2.0) * 0.55;
  var c = color * diff + ambient * color * 1.5 + vec3<f32>(rim) * color;
  c = c + color * 0.18; // gentle self-lit floor so nothing goes fully black
  c = mix(c, skyColor(viewDir), fogAmount(world));
  return vec4<f32>(c, 1.0);
}

// A clip-space position outside the view volume; used to discard whole
// instances (inactive food/particles) without branching per vertex.
fn cullClip() -> vec4<f32> {
  return vec4<f32>(0.0, 0.0, -2.0, 1.0);
}
