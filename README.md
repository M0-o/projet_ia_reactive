# 3D Artificial Life — WebGPU Boids

A self-contained 3D artificial-life simulation: flocking **boids**, **food
clusters**, **predators**, and **obstacles**, set under a **night sky with
stars**. All simulation math runs on the GPU in **WGSL compute shaders**; the
visuals — sky, lit animated models, glowing food, debug overlays — are generated
in vertex/fragment shaders. No bundler, no build step — just static ES modules
served by a tiny local server.

## Run it

```bash
npm run serve     # http://localhost:8080
```

Open the URL in a WebGPU-capable browser (recent Chrome or Edge). WebGPU
requires a secure context; `localhost` qualifies. If WebGPU is missing or the
machine has no adapter, the app shows a clear on-screen message with a hint
instead of a blank canvas.

## What you get

- **Boids** steer with alignment / cohesion / separation, avoid obstacles and
  the predator, and seek food clusters.
- **Health** decays over time; eating a food cluster restores it to 100%.
- **Predators** chase boids; each contact deals damage and counts as a hit —
  after `hitsToKill` hits the boid dies. Dead boids respawn after a delay.
- **Food clusters** regrow after being eaten.
- A **transparent, neutral-grey bounding box** marks the world volume.
- The boids, predators and obstacles are **animated, textured GLB models**
  (skeletal animation baked to joint frames and skinned on the GPU, with a
  random phase offset per instance so the flock does not flap in unison).

## Controls

- **Drag** to orbit, **scroll** to zoom, **shift-drag / middle-drag** to pan.
- The **right-hand panel** exposes every tunable parameter (generated from the
  schema in `src/sim/params.js`) plus Pause / Reset / Randomize-seed. The
  **Scale** section adjusts each model's size, yaw and pitch if a mesh is not
  pointing the way you want, plus the **glow strength** of the emissive
  (self-lit) maps. The **Bloom** section controls the HDR glow pass: threshold
  (measured against the scene's real range — raise it to keep only the brightest
  emissive highlights, lower it to let more of the scene glow), strength and
  radius (the halo width).
- **Debug / effect overlays** are six independent toggles — heading vectors,
  vision radius, hitboxes, bounding boxes, **spatial-hash cells** (the occupied
  cells of the neighbour grid), and **bloom** (the glow post-process). Only the
  ones you enable are drawn.

## Architecture

| Piece | Where |
| --- | --- |
| Compute kernels (10 entry points) | `shaders/sim.wgsl` |
| Shared structs / bindings | `shaders/common.wgsl`, `shaders/*_bindings.wgsl` |
| Render modules | `shaders/render/*.wgsl` |
| Simulation buffers + `step()` | `src/sim/world.js` |
| Parameters + UI schema | `src/sim/params.js` |
| GLB reader + animation baker | `src/io/gltf.js`, `src/render/models.js` |
| WebGPU bootstrap (with friendly errors) | `src/gpu/device.js` |
| Renderer (one draw call per type) | `src/render/renderer.js` |
| Orbit camera | `src/render/camera.js` |
| Control panel / stats | `src/ui/panel.js`, `src/ui/stats.js` |
| App loop | `src/main.js` |
| 3D reference subclasses | `src/legacy/*.js` |

The spatial hash used for neighbour queries is a GPU counting sort with a
two-level prefix scan; all scratch lives in a single `grid` storage buffer so
the whole simulation stays within WebGPU's 8-storage-buffer limit.

### GLB models

`3D-models/` holds `boid.glb`, `predator.glb` and `obstacle.glb`. At startup
`src/render/models.js` parses each GLB, flattens its mesh and skin data, and
bakes the animation into per-frame joint matrices; everything is packed into a
single storage buffer. The `models` shader then renders each agent with one
instanced draw, interpolating between two baked frames. Each mesh is normalised
to radius 1, so it scales directly with the agent's physics radius.

The models are **textured**: every base-colour image referenced by the GLBs is
decoded and packed into one shared atlas (`src/render/textures.js`), and each
vertex's UV is rewritten to sample its material's cell. That keeps a single draw
call per model type even when a mesh uses several materials (the predator has
two).

### Legacy reference classes

The original 2D `boids.js` is left **untouched**. Its behaviours are generalised
to 3D in `src/legacy/boid3d.js` (`Boid3D extends Boid`), with `Vehicle3D`,
`Predator3D`, `Food3D`, and `Obstacle3D` on top. These classic scripts mirror the
GPU data layout and serve as a readable reference; the live simulation runs
entirely in WGSL.

## Tests

```bash
npm test
```

`test/harness.mjs` runs headlessly against the Dawn (`webgpu`) Node bindings and
checks shader compilation, the spatial-hash sort/scan, boid behaviour (bounds,
health, eating, avoidance, predator hits), GLB parsing/animation baking, the JS
3D reference classes, and an **offscreen render-to-texture readback** that
asserts visible pixels.

Note: a real GPU adapter is not available in every environment (e.g. a headless
CI container), so the night sky, model orientation and the spatial-hash overlay
are best confirmed in a WebGPU-capable browser via `npm run serve`.
