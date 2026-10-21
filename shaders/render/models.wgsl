// models.wgsl
// Instanced render of the loaded GLB models: boid, predator and obstacle.
//
// Skinning is "baked": every animation frame's joint matrices are precomputed
// at load time and stored in the model pool. The vertex shader picks the two
// frames bracketing each instance's animation time and blends them, so a whole
// flock animates on the GPU with a single instanced draw call per model.
//
// Layout inside the model pool (offsets in f32 words), see src/render/models.js:
//   [verts]  per vertex: pos(4), normal(4), uv(4)
//   [skin]   per vertex: joints(4), weights(4)
//   [index]  u32 indices, bitcast into f32
//   [frames] per frame: jointCount mat4x4 (jointCount*16 f32)

const BOID_TINT: vec3<f32> = vec3<f32>(0.62, 0.72, 0.90);
const PREDATOR_TINT: vec3<f32> = vec3<f32>(0.95, 0.42, 0.38);
const OBSTACLE_TINT: vec3<f32> = vec3<f32>(0.42, 0.40, 0.38);

// How far the emissive term is pushed above 1.0 (into HDR) at glow strength 1.
// Lit surfaces never exceed ~1.0, so any pixel brighter than that must come from
// a glowing material — which is exactly what the bloom bright pass wants to
// isolate. Without this the scene peaks below the bloom threshold and glow
// cannot bloom at all.
const EMISSIVE_HDR: f32 = 2.5;

struct MOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) normal : vec3<f32>,
  @location(1) world  : vec3<f32>,
  @location(2) uv     : vec2<f32>,
  @location(3) tint   : vec3<f32>,
  @location(4) emissiveUV : vec2<f32>,
  @location(5) glow   : f32,
  @location(6) alphaMode : f32, // 0 = OPAQUE, 1 = MASK, 2 = BLEND
  @location(7) alphaCutoff : f32,
  // The GLB's own material factors, so the albedo and glow match the asset.
  @location(8) emissiveFactor : vec3<f32>,
  @location(9) baseFactor : vec3<f32>,
};

fn culledOut(tint: vec3<f32>) -> MOut {
  var o: MOut;
  o.clip = cullClip();
  o.normal = vec3<f32>(0.0, 1.0, 0.0);
  o.world = vec3<f32>(0.0);
  o.uv = vec2<f32>(0.0);
  o.tint = tint;
  o.emissiveUV = vec2<f32>(0.0);
  o.glow = 0.0;
  o.alphaMode = 0.0;
  o.alphaCutoff = 0.5;
  o.emissiveFactor = vec3<f32>(0.0);
  o.baseFactor = vec3<f32>(1.0);
  return o;
}

fn vertPos(m: ModelInfo, v: u32) -> vec3<f32> {
  return poolVec4(m.mesh.x + v * 12u).xyz;
}
fn vertNormal(m: ModelInfo, v: u32) -> vec3<f32> {
  return poolVec4(m.mesh.x + v * 12u + 4u).xyz;
}
fn vertUV(m: ModelInfo, v: u32) -> vec2<f32> {
  return poolVec4(m.mesh.x + v * 12u + 8u).xy;
}
// Emissive UV lives in the same vertex record (words +10, +11).
fn vertEmissiveUV(m: ModelInfo, v: u32) -> vec2<f32> {
  return poolVec4(m.mesh.x + v * 12u + 8u).zw;
}
// Alpha mode ordinal (word +3 w) and cutoff (word +7 w), baked per vertex from
// the primitive's glTF material so a BLEND mesh (the phoenix) keeps its gaps.
fn vertAlphaMode(m: ModelInfo, v: u32) -> f32 {
  return poolVec4(m.mesh.x + v * 12u).w;
}
fn vertAlphaCutoff(m: ModelInfo, v: u32) -> f32 {
  return poolVec4(m.mesh.x + v * 12u + 4u).w;
}
// Skin block starts right after the vertex block: vertexCount * 12 words.
fn vertJoints(m: ModelInfo, v: u32) -> vec4<f32> {
  return poolVec4(m.mesh.x + m.mesh.y * 12u + v * 8u);
}
fn vertWeights(m: ModelInfo, v: u32) -> vec4<f32> {
  return poolVec4(m.mesh.x + m.mesh.y * 12u + v * 8u + 4u);
}

// Column `c` of joint `j`'s matrix at animation frame `f`. The joint block
// offset is precomputed on the CPU (models.js), so static meshes work too.
fn jointCol(m: ModelInfo, f: u32, j: u32, c: u32) -> vec4<f32> {
  let stride = u32(max(m.info.z, 0.0)) * 16u;
  let base = m.mesh.w + f * stride + j * 16u + c * 4u;
  return poolVec4(base);
}

fn indexAt(m: ModelInfo, k: u32) -> u32 {
  return bitcast<u32>(modelPool[m.mesh.z + k]);
}

// Skinned vertex position + normal for one vertex, blending two baked frames.
// Returns (position, normal) in model space.
fn skinVertex(m: ModelInfo, v: u32, inst: u32) -> array<vec3<f32>, 2> {
  let p = vertPos(m, v);
  let n = vertNormal(m, v);
  if (m.info.z < 0.5) {
    return array<vec3<f32>, 2>(p, n);
  }
  let frameCount = max(m.info.w, 1.0);
  let seconds = P.timing.y * 0.35 / max(m.info.y, 0.2);
  let t = fract(seconds + animPhase(inst)) * frameCount;
  let f0 = u32(t) % u32(frameCount);
  let f1 = (f0 + 1u) % u32(frameCount);
  let frac = t - floor(t);

  let joints = vertJoints(m, v);
  let weights = vertWeights(m, v);
  let ji = array<u32, 4>(u32(joints.x), u32(joints.y), u32(joints.z), u32(joints.w));
  let jw = array<f32, 4>(weights.x, weights.y, weights.z, weights.w);

  var sp = vec4<f32>(0.0);
  var sn = vec4<f32>(0.0);
  for (var k = 0u; k < 4u; k = k + 1u) {
    var m0 = mat4x4<f32>();
    var m1 = mat4x4<f32>();
    for (var c = 0u; c < 4u; c = c + 1u) {
      m0[c] = jointCol(m, f0, ji[k], c);
      m1[c] = jointCol(m, f1, ji[k], c);
    }
    let blended = m0 * (1.0 - frac) + m1 * frac;
    sp = sp + (blended * vec4<f32>(p, 1.0)) * jw[k];
    sn = sn + (blended * vec4<f32>(n, 0.0)) * jw[k];
  }
  return array<vec3<f32>, 2>(sp.xyz, sn.xyz);
}

// Rotate model-space coordinates to correct a model whose authored forward is
// not +Z. Yaw spins about the model's own up axis; pitch then tilts it about the
// (rotated) X axis, so a mesh authored "nose-up" along +Y can be laid forward
// without editing its data.
fn applyModelYawPitch(p: vec3<f32>, yaw: f32, pitch: f32) -> vec3<f32> {
  let cy = cos(yaw);
  let sy = sin(yaw);
  let y1 = p.y;
  var v = vec3<f32>(cy * p.x + sy * p.z, y1, -sy * p.x + cy * p.z);
  if (abs(pitch) > 1e-6) {
    let cp = cos(pitch);
    let sp = sin(pitch);
    v = vec3<f32>(v.x, cp * v.y - sp * v.z, sp * v.y + cp * v.z);
  }
  return v;
}

fn modelVS(m: ModelInfo, vi: u32, inst: u32, center: vec3<f32>, vel: vec3<f32>,
           radius: f32, tint: vec3<f32>, glow: f32) -> MOut {
  var o: MOut;
  let idx = indexAt(m, vi);
  let sk = skinVertex(m, idx, inst);
  let basis = orthoBasis(vel);
  // Baked model-space vertices are normalised to radius 1 at load time, so a
  // straight multiply places the mesh at the agent's physical radius.
  let yaw = m.extra.x;
  let pitch = m.extra.z;
  let local = applyModelYawPitch(sk[0], yaw, pitch) * radius;
  let normal = applyModelYawPitch(sk[1], yaw, pitch);
  let world = center + basis * local;
  o.clip = cam.viewProj * vec4<f32>(world, 1.0);
  o.normal = normalize(basis * safeNormalize(normal, vec3<f32>(0.0, 1.0, 0.0)));
  o.world = world;
  o.uv = vertUV(m, idx);
  o.tint = tint;
  o.emissiveUV = vertEmissiveUV(m, idx);
  o.glow = glow;
  o.alphaMode = vertAlphaMode(m, idx);
  o.alphaCutoff = vertAlphaCutoff(m, idx);
  o.emissiveFactor = m.mat.xyz;
  o.baseFactor = vec3<f32>(m.mat.w, m.mat.w, m.mat.w);
  return o;
}

@vertex
fn vsBoid(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> MOut {
  let m = modelSet.models[0];
  let a = boids[ii];
  if (a.info.x < 0.5) { return culledOut(BOID_TINT); }
  // Baked vertices are normalised to unit radius, so a plain multiply by the
  // agent's physical radius makes the drawn mesh exactly fill its hitbox.
  // "Boid radius" and "Boid model scale" scale both together.
  let r = max(a.info.w, 0.5) * m.info.x;
  return modelVS(m, vi, ii, a.posH.xyz, a.velD.xyz, r, BOID_TINT, m.extra.y);
}

@vertex
fn vsPredator(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> MOut {
  let m = modelSet.models[1];
  let a = predators[ii];
  if (a.info.x < 0.5) { return culledOut(PREDATOR_TINT); }
  // Predator size is driven entirely by the "Predator model scale" slider (the
  // phoenix mesh has a much larger wingspan than a boid, so its default is low).
  let r = max(a.info.w, 0.5) * m.info.x;
  return modelVS(m, vi, ii, a.posH.xyz, a.velD.xyz, r, PREDATOR_TINT, m.extra.y);
}

@vertex
fn vsObstacle(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> MOut {
  let p = obstacles[ii];
  if (p.extra.x < 0.5) { return culledOut(OBSTACLE_TINT); }
  let m = modelSet.models[2];
  let idx = indexAt(m, vi);
  // A little larger than the physics radius so obstacles read as solid bodies.
  let r = max(p.posR.w, 0.5) * m.info.x * 0.9;
  let world = p.posR.xyz + applyModelYawPitch(vertPos(m, idx), m.extra.x, m.extra.z) * r;
  var o: MOut;
  o.clip = cam.viewProj * vec4<f32>(world, 1.0);
  o.normal = safeNormalize(applyModelYawPitch(vertNormal(m, idx), m.extra.x, m.extra.z), vec3<f32>(0.0, 1.0, 0.0));
  o.world = world;
  o.uv = vertUV(m, idx);
  o.tint = OBSTACLE_TINT;
  o.emissiveUV = vertEmissiveUV(m, idx);
  o.glow = m.extra.y;
  o.alphaMode = vertAlphaMode(m, idx);
  o.alphaCutoff = vertAlphaCutoff(m, idx);
  o.emissiveFactor = m.mat.xyz;
  o.baseFactor = vec3<f32>(m.mat.w, m.mat.w, m.mat.w);
  return o;
}

@fragment
fn fsModel(i: MOut) -> @location(0) vec4<f32> {
  let tex = textureSample(atlasTex, atlasSmp, i.uv);

  // Alpha cut-out. The phoenix declares alphaMode BLEND and stores the gaps
  // between its feathers in the base-colour alpha, so a fragment whose alpha
  // falls below the cutoff must be discarded — otherwise those transparent
  // pixels get filled with lit colour and the feathers read as webbed.
  // MASK uses the material's own alphaCutoff; BLEND uses the same threshold
  // (a real blend would need sorted overdraw and a pipeline blend state).
  if (i.alphaMode > 0.5) {
    if (tex.a < i.alphaCutoff) { discard; }
  }

  // The GLB's base-colour factor tints the sampled texture; the small type tint
  // only nudges the three model kinds apart for readability.
  let albedo = tex.rgb * i.baseFactor;
  let base = mix(i.tint, albedo, 0.92);
  var c = shade(normalize(i.normal), base, i.world);
  // Emissive ("glow") is lighting-independent per glTF, multiplied by the
  // material's emissiveFactor. The "Model glow strength" slider scales it so the
  // glow can still be dialled down at night.
  //
  // The scene renders into an HDR (rgba16float) target, but `shade()` only ever
  // produces values up to ~1.0 — so without a push above 1.0 an emissive part is
  // indistinguishable from a lit white surface and the bloom bright pass (which
  // keys off a threshold) selects nothing. EMISSIVE_HDR scales the glow into
  // genuine HDR so glowing materials bloom, while lit surfaces stay <= 1.
  let glow = textureSample(emissiveTex, emissiveSmp, i.emissiveUV).rgb;
  c = vec4<f32>(c.rgb + glow * i.emissiveFactor * (i.glow * EMISSIVE_HDR), 1.0);
  return c;
}
