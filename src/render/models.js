// models.js (loader/baker)
// Turns a GLB file into the flat "model pool" layout the WGSL models.wgsl
// expects. This is the CPU side of the baked-skinning scheme:
//
//   1. Flatten the mesh: gather POSITION / NORMAL / TEXCOORD_0, the skin's
//      JOINTS_0 / WEIGHTS_0, and the index list into single arrays.
//   2. Bake animation: for each of `frameCount` evenly-spaced times, evaluate
//      the animation, build each joint's world matrix and multiply by its
//      inverse bind matrix. Store jointCount mat4x4 per frame.
//   3. Pack everything into one Float32Array (indices bitcast into f32 slots).

import { parseGLB, readAccessor, nodeLocalMatrix, mat4Mul, mat4Identity,
         sampleAnimation, readImageBytes, materialBaseColorTexture,
         materialEmissive } from "../io/gltf.js";

const DEFAULT_FRAMES = 48; // baked samples per animation clip

const MESH_WORDS_PER_VERT = 12; // pos(4) + normal(4) + uv(4)
const SKIN_WORDS_PER_VERT = 8;  // joints(4) + weights(4)

// Load and bake a GLB at `url`. Returns a descriptor including the packed
// pool, its word offset within a shared pool, and the ModelInfo values.
export async function loadModel(url, opts = {}) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load model ${url}: ${res.status}`);
  const buffer = await res.arrayBuffer();
  const keyPrefix = url.replace(/^.*\/|\.glb$/g, "");
  return bakeModel(buffer, { keyPrefix, ...opts });
}

// Concatenate several baked models into one pool buffer. Each model's
// `mesh.vertexOffset` is rewritten to its absolute word offset in the pool.
export function buildModelPool(baked) {
  let words = 0;
  for (const m of baked) words += m.words;
  // Pad so the total is a multiple of 4 f32 (16 bytes) for uniform safety.
  const pool = new Float32Array(words);
  let offset = 0;
  for (const m of baked) {
    pool.set(m.pool, offset);
    // Re-base the three section offsets into absolute pool words.
    m.mesh.vertexOffset = offset;
    m.mesh.indexOffset += offset;
    m.mesh.jointOffset += offset;
    offset += m.words;
  }
  return pool;
}

export function bakeModel(buffer, { frameCount = DEFAULT_FRAMES, scale = 1, keyPrefix = "m" } = {}) {
  const { json, bin } = parseGLB(buffer);
  const nodeWorld = computeNodeWorldMatrices(json);

  // ---- 1. Flatten geometry from every mesh primitive that has POSITION ----
  const positions = [];
  const normals = [];
  const uvs = [];
  const jointArr = [];
  const weightArr = [];
  const indices = [];
  let vertexBase = 0;
  let hasSkin = false;

  const meshNodes = collectMeshNodes(json, nodeWorld);
  // One entry per primitive: which texture it uses and the vertex range it
  // occupies. The atlas pass (applyAtlas) rewrites UVs to point into the atlas.
  const primitives = [];
  for (const mn of meshNodes) {
    const mesh = json.meshes[mn.meshIndex];
    for (const prim of mesh.primitives) {
      if (prim.attributes.POSITION == null) continue;
      const pos = readAccessor(json, bin, prim.attributes.POSITION);
      const nrm = prim.attributes.NORMAL != null ? readAccessor(json, bin, prim.attributes.NORMAL) : null;
      const uv = prim.attributes.TEXCOORD_0 != null ? readAccessor(json, bin, prim.attributes.TEXCOORD_0) : null;
      const jnt = prim.attributes.JOINTS_0 != null ? readAccessor(json, bin, prim.attributes.JOINTS_0) : null;
      const wgt = prim.attributes.WEIGHTS_0 != null ? readAccessor(json, bin, prim.attributes.WEIGHTS_0) : null;
      const count = pos.length / 3;
      for (let i = 0; i < count; i++) {
        positions.push(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
        normals.push(nrm ? nrm[i * 3] : 0, nrm ? nrm[i * 3 + 1] : 1, nrm ? nrm[i * 3 + 2] : 0);
        uvs.push(uv ? uv[i * 2] : 0, uv ? uv[i * 2 + 1] : 0);
        jointArr.push(jnt ? jnt[i * 4] : 0, jnt ? jnt[i * 4 + 1] : 0,
                      jnt ? jnt[i * 4 + 2] : 0, jnt ? jnt[i * 4 + 3] : 0);
        weightArr.push(wgt ? wgt[i * 4] : 0, wgt ? wgt[i * 4 + 1] : 0,
                       wgt ? wgt[i * 4 + 2] : 0, wgt ? wgt[i * 4 + 3] : 0);
      }
      if (jnt) hasSkin = true;
      const srcIdx = materialBaseColorTexture(json, prim.material);
      const emissive = materialEmissive(json, prim.material);
      const emIdx = emissive && emissive.textureSource != null ? emissive.textureSource : null;
      primitives.push({
        vertexStart: vertexBase,
        vertexCount: count,
        // Globally unique per (bake, mesh, image): two different GLBs must not
        // collide onto the same atlas cell just because both are "mesh 0".
        textureKey: srcIdx == null ? null : `${keyPrefix}:${mn.meshIndex}:${srcIdx}`,
        sourceIndex: srcIdx,
        // Emissive ("glowing") channel, sampled from a second atlas.
        emissiveKey: emIdx == null ? null : `${keyPrefix}:em:${mn.meshIndex}:${emIdx}`,
        emissiveSource: emIdx,
        emissiveFactor: emissive ? emissive.factor : null,
      });
      if (prim.indices != null) {
        const idx = readAccessor(json, bin, prim.indices);
        for (let i = 0; i < idx.length; i++) indices.push(idx[i] + vertexBase);
      } else {
        for (let i = 0; i < count; i++) indices.push(i + vertexBase);
      }
      vertexBase += count;
    }
  }

  // ---- 2. Bake joint frames ----
  const skin = (json.skins || [])[0];
  const jointCount = skin ? skin.joints.length : 0;
  const ibm = skin && skin.inverseBindMatrices != null
    ? readAccessor(json, bin, skin.inverseBindMatrices)
    : null;

  const anim = (json.animations || [])[0];
  const animLength = anim ? animationLength(json, bin, anim) : 0;
  const frames = skin && anim ? frameCount : 1;
  const jointFrames = new Float32Array(frames * jointCount * 16);

  if (skin && anim) {
    const targets = buildAnimationTargets(json, bin, anim);
    for (let f = 0; f < frames; f++) {
      const t = (f / (frames - 1 || 1)) * animLength;
      bakeFrame(json, bin, skin, ibm, jointCount, targets, t,
                jointFrames, f * jointCount * 16);
    }
  } else if (skin) {
    // Rest pose.
    for (let j = 0; j < jointCount; j++) {
      const world = nodeWorld.get(skin.joints[j]) || mat4Identity();
      const bind = ibm ? mat4FromFlat(ibm, j) : mat4Identity();
      mat4Mul(world, bind).forEach((v, k) => { jointFrames[j * 16 + k] = v; });
    }
  }

  // ---- 3. Normalise scale: fit the (skinned, rest-pose) mesh to unit radius.
  // For skinned meshes the raw POSITION values are in bind space and get
  // transformed by the joint matrices, so measure the rest pose instead.
  const baseRadius = restPoseRadius(positions, jointArr, weightArr, hasSkin, jointFrames, jointCount);
  const modelScale = baseRadius > 0 ? scale / baseRadius : 1;

  // Apply the normalisation to skinned meshes by scaling the joint matrices on
  // the left (S * joint): that scales the whole skinned output — both the
  // linear part and the joint translation — so the mesh stays intact. Scaling
  // the input positions instead would only scale the linear part and leave the
  // translation at raw scale, which tears the mesh apart.
  if (hasSkin && modelScale !== 1) {
    for (let j = 0; j < jointFrames.length; j += 16) {
      for (let c = 0; c < 4; c++) {
        jointFrames[j + c * 4 + 0] *= modelScale;
        jointFrames[j + c * 4 + 1] *= modelScale;
        jointFrames[j + c * 4 + 2] *= modelScale;
        // row 3 (the homogeneous row) is left untouched
      }
    }
  }

  // ---- 4. Pack into the pool layout ----
  const vertexCount = positions.length / 3;
  const indexWords = indices.length;
  const vertWords = vertexCount * MESH_WORDS_PER_VERT;
  const skinWords = hasSkin ? vertexCount * SKIN_WORDS_PER_VERT : 0;
  const totalWords = vertWords + skinWords + indexWords + jointFrames.length;

  // Explicit section offsets (relative to the vertex block). The shader uses
  // these instead of recomputing strides, so skinned and static meshes share
  // one code path. See ModelInfo.mesh in shaders/render_bindings.wgsl.
  const indexOffset = vertWords + skinWords;
  const jointOffset = indexOffset + indexWords;

  const pool = new Float32Array(totalWords);
  const u32 = new Uint32Array(pool.buffer);
  let w = 0;
  // Normalise to unit radius so a drawn mesh matches its hitbox.
  // - Static meshes have no skin transform, so their positions are scaled here.
  // - Skinned meshes must NOT have their positions pre-scaled (that would scale
  //   only the linear part of `joint * p` and leave the joint translation at raw
  //   scale, shredding the mesh). Their joint matrices are scaled instead.
  const vertScale = hasSkin ? 1 : modelScale;
  for (let i = 0; i < vertexCount; i++) {
    pool[w++] = positions[i * 3] * vertScale;
    pool[w++] = positions[i * 3 + 1] * vertScale;
    pool[w++] = positions[i * 3 + 2] * vertScale;
    pool[w++] = 0;
    pool[w++] = normals[i * 3]; pool[w++] = normals[i * 3 + 1];
    pool[w++] = normals[i * 3 + 2]; pool[w++] = 0;
    pool[w++] = uvs[i * 2]; pool[w++] = uvs[i * 2 + 1]; pool[w++] = 0; pool[w++] = 0;
  }
  for (let i = 0; hasSkin && i < vertexCount; i++) {
    pool[w++] = jointArr[i * 4]; pool[w++] = jointArr[i * 4 + 1];
    pool[w++] = jointArr[i * 4 + 2]; pool[w++] = jointArr[i * 4 + 3];
    pool[w++] = weightArr[i * 4]; pool[w++] = weightArr[i * 4 + 1];
    pool[w++] = weightArr[i * 4 + 2]; pool[w++] = weightArr[i * 4 + 3];
  }
  for (let i = 0; i < indexWords; i++) u32[w++] = indices[i];
  for (let i = 0; i < jointFrames.length; i++) pool[w++] = jointFrames[i];

  return {
    pool,
    words: totalWords,
    vertexCount,
    indexCount: indexWords,
    // Word offsets into the shared pool. `buildModelPool` re-bases all three.
    mesh: { vertexOffset: 0, vertexCount, indexOffset, jointOffset },
    info: { scale: modelScale, animLength, jointCount, frameCount: frames },
    hasSkin,
    // Per-primitive material info, plus the raw GLB bytes (needed to pull the
    // embedded PNGs out for the texture atlas — see collectModelImages).
    primitives,
    glb: buffer,
    uvs,
  };
}

// The distinct images a baked model needs, as atlas inputs. `channel` selects
// the base-colour or the emissive maps. `bakeModel` keeps the GLB bytes, so no
// re-read of the file is required.
export function collectModelImages(model, channel = "base") {
  const { json, bin } = parseGLB(model.glb);
  const isBase = channel === "base";
  const out = [];
  const seen = new Set();
  for (const prim of model.primitives) {
    const key = isBase ? prim.textureKey : prim.emissiveKey;
    const src = isBase ? prim.sourceIndex : prim.emissiveSource;
    if (key == null || src == null || seen.has(key)) continue;
    seen.add(key);
    const img = readImageBytes(json, bin, src);
    if (img) out.push({ key, bytes: img.bytes, mimeType: img.mimeType });
  }
  return out;
}

// Rewrite each vertex's UVs so the primitive samples its cells in the base and
// emissive atlases. `basePlacements` / `emissivePlacements` map texture keys to
// {u0,v0,u1,v1}; `cell` is the atlas cell size used for the half-texel inset.
// Word layout per vertex: uv(2) at +8, emissive uv(2) at +10.
export function applyAtlas(model, basePlacements, emissivePlacements = new Map(), cell = 512) {
  const uvs = model.uvs;
  const pool = model.pool;
  const base = model.mesh.vertexOffset;
  if (!uvs) return;
  const pad = 0.5 / cell;
  const inset = (t) => {
    const f = t - Math.floor(t);            // fold repeat-wrapped UVs back in
    return pad + f * (1 - 2 * pad);
  };
  const mapTo = (place, u, v) => {
    const iu = inset(u), iv = inset(v);
    return [place.u0 + iu * (place.u1 - place.u0), place.v0 + iv * (place.v1 - place.v0)];
  };
  for (const prim of model.primitives) {
    const brim = prim.textureKey != null ? basePlacements.get(prim.textureKey) : null;
    const emir = prim.emissiveKey != null ? emissivePlacements.get(prim.emissiveKey) : null;
    if (!brim && !emir) continue;
    for (let v = prim.vertexStart; v < prim.vertexStart + prim.vertexCount; v++) {
      const u = uvs[v * 2], w = uvs[v * 2 + 1];
      const word = base + v * 12;
      if (brim) {
        const [au, av] = mapTo(brim, u, w);
        pool[word + 8] = au; pool[word + 9] = av;
      }
      // The emissive map uses the same TEXCOORD_0; only its atlas cell differs.
      if (emir) {
        const [eu, ev] = mapTo(emir, u, w);
        pool[word + 10] = eu; pool[word + 11] = ev;
      }
    }
  }
}

// ---- helpers ---------------------------------------------------------------

function mat4FromFlat(flat, i) {
  const m = new Float32Array(16);
  for (let k = 0; k < 16; k++) m[k] = flat[i * 16 + k];
  return m;
}

function meshRadius(positions) {
  let max = 0;
  for (let i = 0; i < positions.length; i += 3) {
    max = Math.max(max, Math.hypot(positions[i], positions[i + 1], positions[i + 2]));
  }
  return max;
}

// Radius of the mesh after skinning with the first baked frame (rest pose).
function restPoseRadius(positions, jointArr, weightArr, hasSkin, jointFrames, jointCount) {
  if (!hasSkin || jointCount === 0) return meshRadius(positions);
  const vertCount = positions.length / 3;
  let max = 0;
  for (let i = 0; i < vertCount; i++) {
    let x = 0, y = 0, z = 0;
    for (let k = 0; k < 4; k++) {
      const j = jointArr[i * 4 + k];
      const w = weightArr[i * 4 + k];
      if (w === 0) continue;
      const o = j * 16;
      x += (jointFrames[o] * positions[i * 3] + jointFrames[o + 4] * positions[i * 3 + 1] +
            jointFrames[o + 8] * positions[i * 3 + 2] + jointFrames[o + 12]) * w;
      y += (jointFrames[o + 1] * positions[i * 3] + jointFrames[o + 5] * positions[i * 3 + 1] +
            jointFrames[o + 9] * positions[i * 3 + 2] + jointFrames[o + 13]) * w;
      z += (jointFrames[o + 2] * positions[i * 3] + jointFrames[o + 6] * positions[i * 3 + 1] +
            jointFrames[o + 10] * positions[i * 3 + 2] + jointFrames[o + 14]) * w;
    }
    max = Math.max(max, Math.hypot(x, y, z));
  }
  return max;
}

// World matrix per node (walking parents), plus a parent map.
function computeNodeWorldMatrices(json) {
  const nodes = json.nodes || [];
  const parents = new Int32Array(nodes.length).fill(-1);
  nodes.forEach((n, i) => (n.children || []).forEach((c) => { parents[c] = i; }));
  const world = new Map();
  const visiting = new Set();
  function get(i) {
    if (world.has(i)) return world.get(i);
    if (visiting.has(i)) return mat4Identity();
    visiting.add(i);
    const local = nodeLocalMatrix(nodes[i]);
    const p = parents[i];
    const m = p >= 0 ? mat4Mul(get(p), local) : local;
    world.set(i, m);
    visiting.delete(i);
    return m;
  }
  for (let i = 0; i < nodes.length; i++) get(i);
  return world;
}

function collectMeshNodes(json, nodeWorld) {
  const out = [];
  (json.nodes || []).forEach((n, i) => {
    if (n.mesh != null) out.push({ meshIndex: n.mesh, node: i });
  });
  return out;
}

function animationLength(json, bin, anim) {
  let max = 0;
  for (const s of anim.samplers) {
    const acc = json.accessors[s.input];
    if (acc.max) max = Math.max(max, acc.max[0]);
  }
  return max;
}

// Pre-index animation channels by node for fast per-frame evaluation.
function buildAnimationTargets(json, bin, anim) {
  const map = new Map(); // nodeIndex -> { translation, rotation, scale }
  for (const ch of anim.channels) {
    const node = ch.target.node;
    if (node == null) continue;
    if (!map.has(node)) map.set(node, {});
    map.get(node)[ch.target.path] = anim.samplers[ch.sampler];
  }
  return map;
}

// Build each joint's world matrix at time t, skinned by its bind matrix.
function bakeFrame(json, bin, skin, ibm, jointCount, targets, t, out, outOff) {
  const nodes = json.nodes;
  // Evaluate animated TRS for every node at time t.
  const trs = new Map();
  for (const [node, chans] of targets) {
    const nodeRef = nodes[node] || {};
    const entry = {
      translation: nodeRef.translation || [0, 0, 0],
      rotation: nodeRef.rotation || [0, 0, 0, 1],
      scale: nodeRef.scale || [1, 1, 1],
    };
    if (chans.translation) entry.translation = Array.from(sampleAnimation(json, bin, chans.translation, t));
    if (chans.rotation) entry.rotation = Array.from(sampleAnimation(json, bin, chans.rotation, t));
    if (chans.scale) entry.scale = Array.from(sampleAnimation(json, bin, chans.scale, t));
    trs.set(node, entry);
  }

  // Recompute the world matrix for each joint, honouring the animated TRS of
  // any joint ancestors (a joint's parent chain is inside the skin).
  const localCache = new Map();
  const worldCache = new Map();
  function localOf(i) {
    if (localCache.has(i)) return localCache.get(i);
    const node = nodes[i];
    const over = trs.get(i);
    let m;
    if (over) m = nodeLocalMatrix(over);
    else m = nodeLocalMatrix(node);
    localCache.set(i, m);
    return m;
  }
  const parentOf = new Int32Array(nodes.length).fill(-1);
  nodes.forEach((n, i) => (n.children || []).forEach((c) => { parentOf[c] = i; }));
  function worldOf(i) {
    if (worldCache.has(i)) return worldCache.get(i);
    const p = parentOf[i];
    const m = p >= 0 ? mat4Mul(worldOf(p), localOf(i)) : localOf(i);
    worldCache.set(i, m);
    return m;
  }

  for (let j = 0; j < jointCount; j++) {
    const world = worldOf(skin.joints[j]);
    const bind = ibm ? mat4FromFlat(ibm, j) : mat4Identity();
    const skinMat = mat4Mul(world, bind);
    for (let k = 0; k < 16; k++) out[outOff + j * 16 + k] = skinMat[k];
  }
}
