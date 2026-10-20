// textures.js
// Builds a single GPU texture atlas from the base-colour images embedded in the
// GLB models. Everything the renderer needs is one texture + one sampler, so a
// model that uses several materials still draws in a single call: the CPU side
// rewrites each vertex's UV to point into the atlas cell (see models.js).
//
// Images are decoded with `createImageBitmap` in the browser. In Node there is
// no decoder, so `buildAtlas` accepts a decoder injection — the test harness
// passes a stub and the atlas still packs (with placeholder pixels).

// Lay out `images` in a grid of `cols` x `rows` cells, each `cell` pixels
// square. Returns the atlas pixels plus per-image cell placement.
function planGrid(count, cell) {
  const cols = Math.max(1, Math.ceil(Math.sqrt(count)));
  const rows = Math.max(1, Math.ceil(count / cols));
  return { cols, rows, cell, width: cols * cell, height: rows * cell };
}

// `images`: [{ key, bytes, mimeType }]. `decode` maps (bytes, mime) to a
// { width, height, data: Uint8ClampedArray } RGBA bitmap.
// Returns { rgba, width, height, placements: Map<key, {u0,v0,u1,v1}> }.
//
// 512px cells keep the GLB art crisp; the browser decoder already returns the
// native resolution, so blitting to 512 is usually a mild downscale.
export async function buildAtlas(images, decode, cell = 512) {
  const unique = [];
  const seen = new Set();
  for (const im of images) {
    if (!im || !im.bytes || seen.has(im.key)) continue;
    seen.add(im.key);
    unique.push(im);
  }

  const plan = planGrid(Math.max(unique.length, 1), cell);
  const rgba = new Uint8Array(plan.width * plan.height * 4);
  // Neutral mid-grey fill so an undecoded cell never renders as black.
  for (let i = 0; i < rgba.length; i += 4) {
    rgba[i] = 150; rgba[i + 1] = 150; rgba[i + 2] = 150; rgba[i + 3] = 255;
  }

  const placements = new Map();
  for (let i = 0; i < unique.length; i++) {
    const im = unique[i];
    const cx = (i % plan.cols) * cell;
    const cy = Math.floor(i / plan.cols) * cell;
    let bmp = null;
    try {
      bmp = decode ? await decode(im.bytes, im.mimeType) : null;
    } catch {
      bmp = null;
    }
    if (bmp && bmp.width && bmp.height) {
      blitBilinear(rgba, plan.width, plan.height, bmp, cx, cy, cell);
    }
    // Atlases are sampled top-left origin; WGSL UV v grows downward, matching
    // the bitmap's row order, so no flip is needed here.
    placements.set(im.key, {
      u0: cx / plan.width,
      v0: cy / plan.height,
      u1: (cx + cell) / plan.width,
      v1: (cy + cell) / plan.height,
    });
  }
  return { rgba, width: plan.width, height: plan.height, cell, placements };
}

// Bilinear scale of a source bitmap into a `cell` x `cell` atlas cell. Keeps
// this dependency-free while avoiding the aliasing a nearest-neighbour blit
// would introduce when downscaling (the boid texture is 512x1024).
function blitBilinear(dst, dstW, dstH, src, cx, cy, cell) {
  const s = src.data;
  if (!s) return;
  const sw = src.width, sh = src.height;
  for (let y = 0; y < cell; y++) {
    // Sample at pixel centres, then clamp to the valid source range.
    const fy = Math.min(sh - 1, Math.max(0, ((y + 0.5) / cell) * sh - 0.5));
    const y0 = Math.floor(fy), y1 = Math.min(sh - 1, y0 + 1), ty = fy - y0;
    for (let x = 0; x < cell; x++) {
      const fx = Math.min(sw - 1, Math.max(0, ((x + 0.5) / cell) * sw - 0.5));
      const x0 = Math.floor(fx), x1 = Math.min(sw - 1, x0 + 1), tx = fx - x0;
      const di = ((cy + y) * dstW + (cx + x)) * 4;
      for (let k = 0; k < 4; k++) {
        const a = s[(y0 * sw + x0) * 4 + k];
        const b = s[(y0 * sw + x1) * 4 + k];
        const c = s[(y1 * sw + x0) * 4 + k];
        const d = s[(y1 * sw + x1) * 4 + k];
        const top = a + (b - a) * tx;
        const bot = c + (d - c) * tx;
        dst[di + k] = top + (bot - top) * ty;
      }
    }
  }
}

// Browser decoder: PNG/JPEG bytes -> ImageBitmap -> RGBA pixels.
export async function browserDecode(bytes, mimeType) {
  const blob = new Blob([bytes], { type: mimeType || "image/png" });
  const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none" });
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, 0, 0);
  const img = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  bitmap.close?.();
  return { width: img.width, height: img.height, data: img.data };
}
