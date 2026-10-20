// gltf.js
// Tiny, dependency-free GLB/glTF reader. It extracts exactly what the renderer
// needs: mesh positions, normals, UVs, skin joints/weights, indices, the
// skeleton hierarchy, inverse bind matrices, and animation samplers. It makes
// no attempt to be a complete glTF implementation.

// Data accessor component types.
const COMP = {
  5120: { array: Int8Array, size: 1, norm: (v) => v / 127 },
  5121: { array: Uint8Array, size: 1, norm: (v) => v / 255 },
  5122: { array: Int16Array, size: 2, norm: (v) => v / 32767 },
  5123: { array: Uint16Array, size: 2, norm: (v) => v / 65535 },
  5125: { array: Uint32Array, size: 4, norm: (v) => v },
  5126: { array: Float32Array, size: 4, norm: (v) => v },
};
const NUM_COMPONENTS = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };

// Parse a .glb ArrayBuffer into { json, bin }.
export function parseGLB(buffer) {
  const dv = new DataView(buffer);
  if (dv.getUint32(0, true) !== 0x46546c67) throw new Error("not a GLB file");
  const version = dv.getUint32(4, true);
  if (version !== 2) throw new Error(`unsupported glTF version ${version}`);
  const total = dv.getUint32(8, true);

  let offset = 12;
  let json = null;
  let bin = null;
  while (offset < total) {
    const len = dv.getUint32(offset, true);
    const type = dv.getUint32(offset + 4, true);
    const start = offset + 8;
    if (type === 0x4e4f534a) {
      json = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, start, len)));
    } else if (type === 0x004e4942) {
      bin = new Uint8Array(buffer, start, len);
    }
    offset = start + len;
  }
  if (!json) throw new Error("GLB has no JSON chunk");
  return { json, bin };
}

// Read an accessor into a flat typed array, applying bufferView byte offsets.
export function readAccessor(json, bin, index) {
  const acc = json.accessors[index];
  const view = json.bufferViews[acc.bufferView];
  const comp = COMP[acc.componentType];
  const n = NUM_COMPONENTS[acc.type];
  const base = (view.byteOffset || 0) + (acc.byteOffset || 0);
  const stride = view.byteStride || 0;

  // Fast path: tightly packed floats/ints.
  if (!stride || stride === comp.size * n) {
    const out = new comp.array(bin.buffer, bin.byteOffset + base, acc.count * n);
    return out;
  }
  // Interleaved: gather element by element.
  const out = new comp.array(acc.count * n);
  for (let i = 0; i < acc.count; i++) {
    const off = base + i * stride;
    for (let c = 0; c < n; c++) {
      out[i * n + c] = comp.norm(readScalar(bin, off + c * comp.size, acc.componentType));
    }
  }
  return out;
}

function readScalar(bin, off, componentType) {
  const dv = new DataView(bin.buffer, bin.byteOffset);
  switch (componentType) {
    case 5120: return dv.getInt8(off);
    case 5121: return dv.getUint8(off);
    case 5122: return dv.getInt16(off, true);
    case 5123: return dv.getUint16(off, true);
    case 5125: return dv.getUint32(off, true);
    case 5126: return dv.getFloat32(off, true);
    default: throw new Error(`unknown component type ${componentType}`);
  }
}

// Small column-major mat4 helpers (kept local so gltf.js is standalone).
export function mat4Identity() {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

export function mat4Mul(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      o[c * 4 + r] =
        a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] +
        a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
  }
  return o;
}

// Raw bytes of an image (its PNG/JPEG payload as embedded in the GLB).
export function readImageBytes(json, bin, imageIndex) {
  const img = (json.images || [])[imageIndex];
  if (!img || img.bufferView == null) return null;
  const view = json.bufferViews[img.bufferView];
  const start = (view.byteOffset || 0);
  return { bytes: bin.subarray(start, start + view.byteLength), mimeType: img.mimeType || "image/png" };
}

// Base-colour texture for a material. Handles the modern
// pbrMetallicRoughness.baseColorTexture and the legacy
// KHR_materials_pbrSpecularGlossiness.diffuseTexture extension.
export function materialBaseColorTexture(json, materialIndex) {
  const mat = (json.materials || [])[materialIndex];
  if (!mat) return null;
  if (mat.pbrMetallicRoughness && mat.pbrMetallicRoughness.baseColorTexture) {
    return json.textures[mat.pbrMetallicRoughness.baseColorTexture.index].source;
  }
  const sg = mat.extensions && mat.extensions.KHR_materials_pbrSpecularGlossiness;
  if (sg && sg.diffuseTexture) {
    return json.textures[sg.diffuseTexture.index].source;
  }
  return null;
}

// Emissive (self-lit) texture + factor for a material, or null when the
// material does not glow. The factor is a vec3 multiplier over the texture.
export function materialEmissive(json, materialIndex) {
  const mat = (json.materials || [])[materialIndex];
  if (!mat) return null;
  const tex = mat.emissiveTexture
    ? json.textures[mat.emissiveTexture.index].source
    : null;
  const factor = mat.emissiveFactor || [0, 0, 0];
  const glows = tex != null || factor[0] + factor[1] + factor[2] > 0;
  if (!glows) return null;
  return { textureSource: tex, factor };
}

// Compose a node's local matrix from TRS or a raw matrix.
export function nodeLocalMatrix(node) {
  if (node.matrix) return Float32Array.from(node.matrix);
  const t = node.translation || [0, 0, 0];
  const r = node.rotation || [0, 0, 0, 1];
  const s = node.scale || [1, 1, 1];
  const [x, y, z, w] = r;
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  const m = new Float32Array(16);
  m[0] = (1 - (yy + zz)) * s[0];
  m[1] = (xy + wz) * s[0];
  m[2] = (xz - wy) * s[0];
  m[3] = 0;
  m[4] = (xy - wz) * s[1];
  m[5] = (1 - (xx + zz)) * s[1];
  m[6] = (yz + wx) * s[1];
  m[7] = 0;
  m[8] = (xz + wy) * s[2];
  m[9] = (yz - wx) * s[2];
  m[10] = (1 - (xx + yy)) * s[2];
  m[11] = 0;
  m[12] = t[0]; m[13] = t[1]; m[14] = t[2]; m[15] = 1;
  return m;
}

// Spherical / linear / cubic sampling of an animation sampler at time t.
// Returns a Float32Array of `n` components.
export function sampleAnimation(json, bin, sampler, t) {
  const input = readAccessor(json, bin, sampler.input);
  const output = readAccessor(json, bin, sampler.output);
  const interp = sampler.interpolation || "LINEAR";
  const frames = input.length;
  // Values per keyframe.
  const per = output.length / frames;

  if (t <= input[0]) return output.slice(0, per);
  if (t >= input[frames - 1]) return output.slice((frames - 1) * per, frames * per);

  let i = 0;
  while (i < frames - 1 && input[i + 1] < t) i++;
  const t0 = input[i], t1 = input[i + 1];
  const u = t1 > t0 ? (t - t0) / (t1 - t0) : 0;

  if (interp === "STEP") return output.slice(i * per, (i + 1) * per);

  const out = new Float32Array(per);
  if (interp === "CUBICSPLINE") {
    // 3 values per keyframe: in-tangent, value, out-tangent.
    const stride = per * 3;
    const a = i * stride, b = (i + 1) * stride;
    for (let k = 0; k < per; k++) {
      const v0 = output[a + per + k], v1 = output[b + per + k];
      const m0 = output[a + 2 * per + k], m1 = output[b + k];
      const u2 = u * u, u3 = u2 * u;
      out[k] = (2 * u3 - 3 * u2 + 1) * v0 + (u3 - 2 * u2 + u) * m0 +
               (-2 * u3 + 3 * u2) * v1 + (u3 - u2) * m1;
    }
    return out;
  }

  // LINEAR (slerp for rotations where the caller knows it is a quaternion).
  const a = i * per, b = (i + 1) * per;
  for (let k = 0; k < per; k++) out[k] = output[a + k] * (1 - u) + output[b + k] * u;
  return out;
}

export { COMP, NUM_COMPONENTS };
