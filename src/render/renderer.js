// renderer.js
// Owns every render pipeline, the depth buffer, and the frame's render pass.
// One draw call per agent type / overlay. Debug overlays are gated individually
// so nothing is drawn unless requested.
//
// Draw order:
//   sky (full-screen background) -> opaque (models, food-ish) -> additive food
//   -> box edges -> debug lines. Transparent passes write no depth.

import { composeRender } from "../sim/compose.js";
import { renderLayout, MODEL_INFO_BYTES } from "../sim/bindings.js";
import { CAMERA_BYTES } from "./camera.js";

// Per-module vertex/index counts (procedural meshes + loaded models).
const QUAD_VERTS = 6;
const BOX_EDGE_VERTS = 24;
const RING_VERTS = 48 * 2;  // line-list, RING_SEGS * 2
const HIT_VERTS = 16 * 2;   // line-list, HIT_SEGS * 2
const CELL_EDGE_VERTS = 24; // hash cell box wireframe

const OPAQUE = { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" };
const TRANS_DEPTH = { format: "depth24plus", depthWriteEnabled: false, depthCompare: "less" };
const SKY_DEPTH = { format: "depth24plus", depthWriteEnabled: false, depthCompare: "always" };

const BLEND_ALPHA = {
  color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
  alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
};
const BLEND_ADD = {
  color: { srcFactor: "src-alpha", dstFactor: "one", operation: "add" },
  alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
};

// 1x1 fallback texture, used when no atlas was supplied (headless tests) or an
// atlas has no images. A grey texel keeps untextured models visible; a black
// texel means "nothing glows".
function solidTexture(device, rgba) {
  const tex = device.createTexture({
    label: "atlas-fallback",
    size: [1, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture({ texture: tex }, new Uint8Array(rgba),
    { bytesPerRow: 4 }, [1, 1]);
  return tex.createView();
}

// Upload an atlas ({ rgba, width, height }) as a sampled GPU texture and return
// its view. `generateMipmaps` is skipped: the models are small on screen and one
// level keeps load time and memory low.
export function createAtlasTexture(device, atlas) {
  const tex = device.createTexture({
    label: "model-atlas",
    size: [atlas.width, atlas.height],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture(
    { texture: tex },
    atlas.rgba,
    { bytesPerRow: atlas.width * 4, rowsPerImage: atlas.height },
    [atlas.width, atlas.height],
  );
  return tex.createView();
}

// `models` describes each loaded GLB: { poolOffset (words), info } and drives
// how many indices to draw per instance.
export function createRenderer({ device, context, format, shaderParts, world, models }) {  const camBuffer = device.createBuffer({
    label: "camera",
    size: CAMERA_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const modelSetBuffer = device.createBuffer({
    label: "model-set",
    size: MODEL_INFO_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Base-colour atlas. If the caller did not supply one (headless tests, or a
  // GLB with no images) fall back to a 1x1 neutral texel so the shader's
  // textureSample is always valid. The emissive atlas falls back to black,
  // which means "no glow".
  const atlasView = models.atlasView || solidTexture(device, [180, 180, 180, 255]);
  const emissiveView = models.emissiveView || solidTexture(device, [0, 0, 0, 255]);
  const atlasSampler = device.createSampler({
    label: "atlas-sampler",
    magFilter: "linear", minFilter: "linear",
    mipmapFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge",
  });

  const layout = renderLayout(device);
  const bindGroup = device.createBindGroup({
    label: "render-bind-group",
    layout,
    entries: [
      { binding: 0, resource: { buffer: world.buffers.params } },
      { binding: 1, resource: { buffer: camBuffer } },
      { binding: 2, resource: { buffer: world.buffers.boids } },
      { binding: 3, resource: { buffer: world.buffers.predators } },
      { binding: 4, resource: { buffer: world.buffers.food } },
      { binding: 5, resource: { buffer: world.buffers.obstacles } },
      { binding: 6, resource: { buffer: models.poolBuffer } },
      { binding: 7, resource: { buffer: modelSetBuffer } },
      { binding: 8, resource: { buffer: world.buffers.grid } },
      { binding: 9, resource: atlasView },
      { binding: 10, resource: atlasSampler },
      { binding: 11, resource: emissiveView },
      { binding: 12, resource: atlasSampler },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({
    label: "render-pipeline-layout",
    bindGroupLayouts: [layout],
  });

  const modules = {};
  for (const name of ["sky", "models", "food", "box", "debug", "hashcells"]) {
    modules[name] = device.createShaderModule({
      label: `render-${name}`,
      code: composeRender(shaderParts, name),
    });
  }

  const mk = (label, module, vs, fs, topology, depth, blend) =>
    device.createRenderPipeline({
      label,
      layout: pipelineLayout,
      vertex: { module, entryPoint: vs },
      fragment: { module, entryPoint: fs, targets: [{ format, ...(blend ? { blend } : {}) }] },
      primitive: { topology, cullMode: "none" },
      depthStencil: depth,
    });

  const pipelines = {
    sky: mk("sky", modules.sky, "vsSky", "fsSky", "triangle-list", SKY_DEPTH, null),
    boidModel: mk("model-boid", modules.models, "vsBoid", "fsModel", "triangle-list", OPAQUE, null),
    predatorModel: mk("model-predator", modules.models, "vsPredator", "fsModel", "triangle-list", OPAQUE, null),
    obstacleModel: mk("model-obstacle", modules.models, "vsObstacle", "fsModel", "triangle-list", OPAQUE, null),
    boxEdges: mk("box-edges", modules.box, "vsBoxEdges", "fsBoxEdges", "line-list", TRANS_DEPTH, BLEND_ALPHA),
    food: mk("food", modules.food, "vsFood", "fsFood", "triangle-list", TRANS_DEPTH, BLEND_ADD),
    hashCells: mk("hash-cells", modules.hashcells, "vsHashCells", "fsHashCells", "line-list", TRANS_DEPTH, BLEND_ALPHA),
    headingBoids: mk("dbg-heading-boids", modules.debug, "vsHeadingBoids", "fsLine", "line-list", TRANS_DEPTH, BLEND_ALPHA),
    headingPredators: mk("dbg-heading-pred", modules.debug, "vsHeadingPredators", "fsLine", "line-list", TRANS_DEPTH, BLEND_ALPHA),
    visionBoids: mk("dbg-vision-boids", modules.debug, "vsVisionBoids", "fsLine", "line-list", TRANS_DEPTH, BLEND_ALPHA),
    visionPredators: mk("dbg-vision-pred", modules.debug, "vsVisionPredators", "fsLine", "line-list", TRANS_DEPTH, BLEND_ALPHA),
    hitboxBoids: mk("dbg-hitbox-boids", modules.debug, "vsHitboxBoids", "fsLine", "line-list", TRANS_DEPTH, BLEND_ALPHA),
    hitboxPredators: mk("dbg-hitbox-pred", modules.debug, "vsHitboxPredators", "fsLine", "line-list", TRANS_DEPTH, BLEND_ALPHA),
    hitboxObstacles: mk("dbg-hitbox-obst", modules.debug, "vsHitboxObstacles", "fsLine", "line-list", TRANS_DEPTH, BLEND_ALPHA),
    bboxBoids: mk("dbg-bbox-boids", modules.debug, "vsBBoxBoids", "fsLine", "line-list", TRANS_DEPTH, BLEND_ALPHA),
    bboxPredators: mk("dbg-bbox-pred", modules.debug, "vsBBoxPredators", "fsLine", "line-list", TRANS_DEPTH, BLEND_ALPHA),
  };

  // Depth texture, recreated on resize.
  let depthTex = null, depthView = null, curW = 0, curH = 0;
  function ensureDepth(w, h) {
    if (w === curW && h === curH && depthView) return depthView;
    depthTex?.destroy();
    depthTex = device.createTexture({
      label: "depth", size: [w, h], format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    depthView = depthTex.createView();
    curW = w; curH = h;
    return depthView;
  }

  function draw(pass, name, verts, instances = 1) {
    pass.setPipeline(pipelines[name]);
    pass.setBindGroup(0, bindGroup);
    pass.draw(verts, instances, 0, 0);
  }
  function drawIndexed(pass, name, indexCount, instances) {
    // The models shader fetches its indices from the storage pool and uses
    // `vertex_index` to walk them, so no GPU index buffer is bound.
    pass.setPipeline(pipelines[name]);
    pass.setBindGroup(0, bindGroup);
    pass.draw(indexCount, instances, 0, 0);
  }

  // Update the per-model metadata uniform (scale + orientation from params).
  // `opts` = { boidScale, predatorScale, boidYaw, predatorYaw, glow }.
  function updateModels(opts = {}) {
    const data = new ArrayBuffer(MODEL_INFO_BYTES);
    const u32 = new Uint32Array(data);
    const f32 = new Float32Array(data);
    const DEG = Math.PI / 180;
    const glow = opts.glow ?? 0;
    const write = (i, m, scaleMul, yawDeg) => {
      const o = i * 12;
      // mesh = (vertexOffset, vertexCount, indexOffset, jointOffset)
      u32[o + 0] = m.mesh.vertexOffset;
      u32[o + 1] = m.mesh.vertexCount;
      u32[o + 2] = m.mesh.indexOffset;
      u32[o + 3] = m.mesh.jointOffset;
      // Baking normalises each mesh to radius 1, so info.x is a pure size
      // multiplier on the agent's physics radius.
      f32[o + 4] = scaleMul;
      f32[o + 5] = m.info.animLength;
      f32[o + 6] = m.info.jointCount;
      f32[o + 7] = m.info.frameCount;
      f32[o + 8] = (yawDeg || 0) * DEG; // model-forward correction
      f32[o + 9] = glow;                // emissive map strength
    };
    write(0, models.boid, opts.boidScale ?? 1, opts.boidYaw);
    write(1, models.predator, opts.predatorScale ?? 1, opts.predatorYaw);
    write(2, models.obstacle, 1, 0);
    write(3, models.obstacle, 1, 0);
    device.queue.writeBuffer(modelSetBuffer, 0, data);
  }

  // counts: { boids, predators, food, obstacles }
  // debug:  { heading, vision, hitbox, bbox, hashCells } individual toggles
  function render(counts, debug, widthPx, heightPx) {
    const dv = ensureDepth(widthPx, heightPx);
    const enc = device.createCommandEncoder({ label: "render" });
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.008, g: 0.012, b: 0.030, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
      depthStencilAttachment: {
        view: dv, depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "store",
      },
    });

    // Background night sky.
    draw(pass, "sky", 3, 1);

    // Opaque loaded models.
    drawIndexed(pass, "boidModel", models.boid.indexCount, counts.boids);
    drawIndexed(pass, "predatorModel", models.predator.indexCount, counts.predators);
    drawIndexed(pass, "obstacleModel", models.obstacle.indexCount, counts.obstacles);

    // Additive food glow.
    draw(pass, "food", QUAD_VERTS, counts.food);

    // Box edges.
    draw(pass, "boxEdges", BOX_EDGE_VERTS, 1);

    // Spatial-hash occupancy cells (optional).
    if (debug.hashCells) draw(pass, "hashCells", CELL_EDGE_VERTS, world.getDims().cells);

    // Debug overlays — each drawn only if its toggle is on.
    if (debug.heading) {
      draw(pass, "headingBoids", 2, counts.boids);
      draw(pass, "headingPredators", 2, counts.predators);
    }
    if (debug.vision) {
      draw(pass, "visionBoids", RING_VERTS, counts.boids);
      draw(pass, "visionPredators", RING_VERTS, counts.predators);
    }
    if (debug.hitbox) {
      draw(pass, "hitboxBoids", HIT_VERTS, counts.boids);
      draw(pass, "hitboxPredators", HIT_VERTS, counts.predators);
      draw(pass, "hitboxObstacles", HIT_VERTS, counts.obstacles);
    }
    if (debug.bbox) {
      draw(pass, "bboxBoids", BOX_EDGE_VERTS, counts.boids);
      draw(pass, "bboxPredators", BOX_EDGE_VERTS, counts.predators);
    }

    pass.end();
    device.queue.submit([enc.finish()]);
  }

  function updateCamera(cameraData) {
    device.queue.writeBuffer(camBuffer, 0, cameraData);
  }

  return { render, updateCamera, updateModels, pipelines, camBuffer, ensureDepth };
}
