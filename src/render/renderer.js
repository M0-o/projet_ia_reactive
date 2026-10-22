// renderer.js
// Owns every render pipeline, the depth buffer, and the frame's render pass.
// One draw call per agent type / overlay. Debug overlays are gated individually
// so nothing is drawn unless requested.
//
// Draw order:
//   sky (full-screen background) -> opaque (models, food-ish) -> additive food
//   -> box edges -> debug lines. Transparent passes write no depth.

import { composeRender, composePost } from "../sim/compose.js";
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

// The scene is rendered into an HDR target so the phoenix's emissive can carry
// values above 1.0 and the bloom bright-pass has real headroom to extract.
const HDR_FORMAT = "rgba16float";
// Bloom runs at half the viewport resolution: the blur is a wide, low-frequency
// effect, so the resolution drop is invisible and the pass is far cheaper.
const BLOOM_DIV = 2;
const POST_UNIFORM_BYTES = 16; // vec4: threshold, strength, radius, texel step

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
  const bloomModule = device.createShaderModule({
    label: "render-bloom",
    code: composePost(shaderParts, "bloom"),
  });

  // Scene pipelines all render into the HDR target (format `HDR_FORMAT`); the
  // post pipelines target the swapchain format so the composite is the only
  // thing the canvas ever sees.
  const mk = (label, module, vs, fs, topology, depth, blend, targetFormat = HDR_FORMAT) =>
    device.createRenderPipeline({
      label,
      layout: pipelineLayout,
      vertex: { module, entryPoint: vs },
      fragment: { module, entryPoint: fs, targets: [{ format: targetFormat, ...(blend ? { blend } : {}) }] },
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

  // ---- bloom / post-processing -------------------------------------------
  // Its own bind group: one uniform + the source texture, the bloom texture, and
  // a linear sampler. Bindings must match shaders/render/bloom.wgsl.
  //
  // Each pass gets its OWN uniform buffer. A single shared buffer would not work:
  // every `queue.writeBuffer` is queued before the one `submit`, so all passes
  // would read the last-written value — the blur would then sample with a zero
  // texel step (no spread at all) and the radius would do nothing.
  const postUniforms = {
    bright: device.createBuffer({ label: "post-bright", size: POST_UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    blurH: device.createBuffer({ label: "post-blurh", size: POST_UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    blurV: device.createBuffer({ label: "post-blurv", size: POST_UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    composite: device.createBuffer({ label: "post-composite", size: POST_UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
  };
  const postLayout = device.createBindGroupLayout({
    label: "post-layout",
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
    ],
  });
  const postPipelineLayout = device.createPipelineLayout({
    label: "post-pipeline-layout",
    bindGroupLayouts: [postLayout],
  });
  const postSmp = device.createSampler({
    label: "post-sampler",
    magFilter: "linear", minFilter: "linear",
    addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge",
  });
  const mkPost = (label, fs, targetFormat) => device.createRenderPipeline({
    label,
    layout: postPipelineLayout,
    vertex: { module: bloomModule, entryPoint: "vsFullscreen" },
    fragment: { module: bloomModule, entryPoint: fs, targets: [{ format: targetFormat }] },
    primitive: { topology: "triangle-list", cullMode: "none" },
  });
  // Bright pass + blur write into the HDR bloom targets; only the composite
  // writes to the swapchain, so its pipeline targets the canvas format.
  const postPipelines = {
    bright: mkPost("bloom-bright", "fsBright", HDR_FORMAT),
    blur: mkPost("bloom-blur", "fsBlur", HDR_FORMAT),
    composite: mkPost("bloom-composite", "fsComposite", format),
  };

  // Scene HDR target + bloom ping-pong targets, recreated on resize.
  let hdrTex = null, hdrView = null;
  let bloomA = null, bloomB = null, bloomAView = null, bloomBView = null;
  let postVW = 0, postVH = 0;
  function ensurePost(w, h) {
    if (w === postVW && h === postVH && hdrView) return;
    hdrTex?.destroy(); bloomA?.destroy(); bloomB?.destroy();
    hdrTex = device.createTexture({
      label: "scene-hdr", size: [w, h], format: HDR_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    hdrView = hdrTex.createView();
    const bw = Math.max(1, Math.ceil(w / BLOOM_DIV));
    const bh = Math.max(1, Math.ceil(h / BLOOM_DIV));
    const bloomUsage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC;
    bloomA = device.createTexture({ label: "bloom-a", size: [bw, bh], format: HDR_FORMAT, usage: bloomUsage });
    bloomB = device.createTexture({ label: "bloom-b", size: [bw, bh], format: HDR_FORMAT, usage: bloomUsage });
    bloomAView = bloomA.createView();
    bloomBView = bloomB.createView();
    postVW = w; postVH = h;
  }
  // A bind group over (source texture, second texture). All four bindings must
  // be present, but WebGPU forbids binding a texture as both a sampled source
  // and the colour target of the same pass — so callers pass a *distinct*
  // texture for the second slot (typically the scene, which is never a bloom
  // pass's target). `src` is binding 1; `bloom` is binding 2. `uniform` is the
  // pass's own parameter block (see postUniforms).
  function postBindGroup(uniform, src, bloom) {
    return device.createBindGroup({
      layout: postLayout,
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: src },
        { binding: 2, resource: bloom },
        { binding: 3, resource: postSmp },
      ],
    });
  }

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
    // 16 floats (4 vec4) per model: mesh(4u) + info(4) + extra(4) + mat(4).
    const STRIDE = 16;
    const write = (i, m, scaleMul, yawDeg, pitchDeg) => {
      const o = i * STRIDE;
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
      f32[o + 8] = (yawDeg || 0) * DEG;   // model-forward correction (spin)
      f32[o + 9] = glow;                  // emissive map strength
      f32[o + 10] = (pitchDeg || 0) * DEG; // model-forward correction (tilt)
      f32[o + 11] = 0;
      // mat = (emissive factor rgb, base-colour factor r). The shaders use
      // these so the glow and albedo match the GLB, not a fixed tint.
      const ef = m.emissiveFactor || [0, 0, 0];
      const bf = m.baseColorFactor || [1, 1, 1];
      f32[o + 12] = ef[0];
      f32[o + 13] = ef[1];
      f32[o + 14] = ef[2];
      f32[o + 15] = bf[0];
    };
    write(0, models.boid, opts.boidScale ?? 1, opts.boidYaw, opts.boidPitch);
    write(1, models.predator, opts.predatorScale ?? 1, opts.predatorYaw, opts.predatorPitch);
    write(2, models.obstacle, 1, 0, 0);
    write(3, models.obstacle, 1, 0, 0);
    device.queue.writeBuffer(modelSetBuffer, 0, data);
  }

  // counts: { boids, predators, food, obstacles }
  // debug:  { heading, vision, hitbox, bbox, hashCells } individual toggles
  // post:   { bloom, threshold, strength, radius } bloom settings
  function render(counts, debug, widthPx, heightPx, post = {}) {
    const dv = ensureDepth(widthPx, heightPx);
    ensurePost(widthPx, heightPx);
    const enc = device.createCommandEncoder({ label: "render" });

    // --- scene pass: into the HDR target -----------------------------------
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: hdrView,
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

    const swapView = context.getCurrentTexture().createView();

    if (!post.bloom) {
      // Bloom disabled: straight copy of the HDR scene to the canvas.
      writePostUniform("composite", post.threshold ?? 1, 0, 0, 0); // strength 0 => pure copy
      const cpass = enc.beginRenderPass({
        colorAttachments: [{ view: swapView, loadOp: "clear", clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: "store" }],
      });
      cpass.setPipeline(postPipelines.composite);
      cpass.setBindGroup(0, postBindGroup(postUniforms.composite, hdrView, bloomAView));
      cpass.draw(3, 1, 0, 0);
      cpass.end();
      device.queue.submit([enc.finish()]);
      return;
    }

    // --- bloom chain -------------------------------------------------------
    const bw = Math.max(1, Math.ceil(widthPx / BLOOM_DIV));
    const bh = Math.max(1, Math.ceil(heightPx / BLOOM_DIV));
    const threshold = post.threshold ?? 1.0;
    const strength = post.strength ?? 0.8;
    const radius = post.radius ?? 1.0;

    // 1. Bright pass: HDR scene -> bloomA (downsampled). Its bind group must not
    // reference bloomA (the target), so the unused second texture is bloomB.
    // The bright pass dilates the source by one texel in each direction, so it
    // needs a real texel step (it used to point-sample and passed 0).
    writePostUniform("bright", threshold, strength, radius, 1 / bw);
    const brightPass = enc.beginRenderPass({
      colorAttachments: [{ view: bloomAView, loadOp: "clear", clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: "store" }],
    });
    brightPass.setPipeline(postPipelines.bright);
    brightPass.setBindGroup(0, postBindGroup(postUniforms.bright, hdrView, bloomBView));
    brightPass.draw(3, 1, 0, 0);
    brightPass.end();

    // 2. Separable blur: horizontal then vertical, ping-ponging A->B->A.
    const stepX = 1 / bw, stepY = 1 / bh;
    writePostUniform("blurH", threshold, strength, radius, stepX);
    const blurH = enc.beginRenderPass({
      colorAttachments: [{ view: bloomBView, loadOp: "clear", clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: "store" }],
    });
    blurH.setPipeline(postPipelines.blur);
    blurH.setBindGroup(0, postBindGroup(postUniforms.blurH, bloomAView, hdrView));
    blurH.draw(3, 1, 0, 0);
    blurH.end();

    writePostUniform("blurV", threshold, strength, radius, stepY);
    const blurV = enc.beginRenderPass({
      colorAttachments: [{ view: bloomAView, loadOp: "clear", clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: "store" }],
    });
    blurV.setPipeline(postPipelines.blur);
    blurV.setBindGroup(0, postBindGroup(postUniforms.blurV, bloomBView, hdrView));
    blurV.draw(3, 1, 0, 0);
    blurV.end();

    // 3. Composite: HDR scene + bloom -> swapchain.
    writePostUniform("composite", threshold, strength, radius, 0);
    const comp = enc.beginRenderPass({
      colorAttachments: [{ view: swapView, loadOp: "clear", clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: "store" }],
    });
    comp.setPipeline(postPipelines.composite);
    comp.setBindGroup(0, postBindGroup(postUniforms.composite, hdrView, bloomAView));
    comp.draw(3, 1, 0, 0);
    comp.end();

    device.queue.submit([enc.finish()]);
  }

  // Write one post pass's uniform: (threshold, strength, radius, texelStep).
  function writePostUniform(which, threshold, strength, radius, texelStep) {
    device.queue.writeBuffer(postUniforms[which], 0,
      new Float32Array([threshold, strength, radius, texelStep]));
  }

  function updateCamera(cameraData) {
    device.queue.writeBuffer(camBuffer, 0, cameraData);
  }

  // `hdrTexture()` / `bloomTextures()` return the current targets (recreated on
  // resize); exposed for diagnostics/tests that inspect the post chain.
  const hdrTexture = () => hdrTex;
  const bloomTextures = () => ({ a: bloomA, b: bloomB });

  return { render, updateCamera, updateModels, pipelines, camBuffer, ensureDepth, hdrTexture, bloomTextures };
}
