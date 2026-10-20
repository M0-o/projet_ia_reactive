# AGENTS.md

## What this is
A 3D artificial-life simulation (flocking boids, food clusters, predators,
obstacles) rendered with **WebGPU**. All simulation math runs in **WGSL compute
shaders**; the look — a night-sky background with stars, lit GLB models, and
individually toggleable debug overlays — is done in vertex/fragment shaders. The
app itself is **plain static ES modules**: no bundler, no build step, no
transpiler. `package.json` exists only for the dev server and the headless test
harness.

## Running
- `npm run serve` → serves the repo on <http://localhost:8080>. Open that URL
  in a WebGPU-capable browser (recent Chrome/Edge). WebGPU needs a secure
  context; `localhost` counts as secure.
- `npm test` → runs `test/harness.mjs` headlessly against Dawn (`webgpu` npm
  bindings). Currently **38 tests pass**.
- There is **no** build/lint step.

## Layout
Top-level entry:
- `index.html` — loads the legacy classic scripts (shim → `boids.js` →
  subclasses) **then** `src/main.js` as a module.
- `styles.css` — canvas + panel + stats/error overlay styling.
- `serve.mjs` — zero-dependency static server.
- `3D-models/` — the GLB assets (`boid.glb`, `predator.glb`, `obstacle.glb`).

Shader sources (`shaders/`), concatenated at load time by `src/sim/compose.js`
(there is no WGSL `#include`):
- `common.wgsl` — shared structs (`Agent`, `Point`, `Params`), the `O_*`
  spatial-hash offsets, RNG/vector helpers.
- `compute_bindings.wgsl` — compute bind group (6 buffers; `smoke` removed).
- `render_bindings.wgsl` — `Camera` + `ModelInfo` structs + render bind group.
- `render_common.wgsl` — night-sky gradient + star field, `shade()`, `rayDir()`,
  `orthoBasis()`, `cullClip()`.
- `sim.wgsl` — **11 compute entry points**: `gridClear`, `gridCount`,
  `gridScanLocal`, `gridScanPartial`, `gridScanAdd`, `gridCursorInit`,
  `gridScatter`, `boidsUpdate`, `predatorsUpdate`, `foodUpdate`,
  `flockHealUpdate`. (The smoke kernels were removed.)
- `render/*.wgsl` — `sky` (full-screen night sky), `models` (instanced skinned
  GLB rendering + texture atlas sampling), `food`, `box`, `debug_lines`,
  `hashcells`.
  Every pipeline is a **single instanced / procedural draw** — the models shader
  reads vertices, indices and joint frames from one storage buffer, so there is
  still no vertex/index buffer bound anywhere.

Simulation core (`src/sim/`):
- `params.js` — `DEFAULTS` + `SCHEMA` (the UI is generated from `SCHEMA`) +
  `ParamPacker` (mirrors the WGSL `Params` struct: 15×vec4 = 240 bytes).
- `bindings.js` — buffer strides/capacities, `GRID_WORDS`, bind group layouts,
  `MODEL_INFO_BYTES`.
- `compose.js` — shader file list + concatenation order.
- `world.js` — owns every buffer + compute pipeline and one `step()`. Shared
  verbatim by the browser app and the Node harness.

GLB pipeline:
- `src/io/gltf.js` — a small dependency-free GLB/glTF reader (accessors,
  node hierarchy, animation sampler evaluation, embedded image bytes and
  base-colour material lookup; casts are validated by tests).
- `src/render/models.js` — bakes a GLB into the flat **model pool** layout:
  flattens mesh + skin data, evaluates the animation at N evenly-spaced times
  and stores each joint's `worldJoint × inverseBind` matrix per frame, then
  packs verts / skin / indices / joint-frames into one `Float32Array`. Each mesh
  is normalised to **radius 1** so the shader can scale it by a physics radius.
  `collectModelImages` pulls the base-colour PNGs back out and `applyAtlas`
  rewrites each vertex's UV to sample its cell in the shared texture atlas.
- `src/render/textures.js` — packs every GLB base-colour image **and** its
  emissive ("glow") map into two bilinear texture atlases (512px cells), so a
  multi-material mesh still draws in a single call. Decoding uses
  `createImageBitmap` + `OffscreenCanvas` in the browser.
- `src/render/renderer.js` — uploads the pool + both atlases once, writes the
  per-model `ModelInfo` uniform each frame (mesh offsets, size multiplier, yaw,
  glow, and the GLB's emissive/base-colour factors), and
  issues one `draw()` per model type (no index buffer; the shader walks indices
  itself).

Browser plumbing:
- `src/gpu/device.js` — `initWebGPU()`; on failure throws an error with a
  human-readable `userMessage` (shown in the UI instead of a blank canvas).
- `src/math/mat4.js` — column-major matrices, **WebGPU clip space (z ∈ [0,1])**.
- `src/render/camera.js` — orbit camera → the `Camera` uniform (256 bytes).
- `src/ui/panel.js` — control panel generated from `SCHEMA`; **5** debug
  checkboxes (heading, vision, hitbox, bbox, spatial-hash cells).
- `src/ui/stats.js` — throttled (~250 ms) boid-buffer readback → stats overlay.
- `src/main.js` — loop, model load/bake, param writeback, debug toggles, resize,
  error display.

Legacy 3D reference (`src/legacy/`), loaded as **classic scripts** and exposed
on `globalThis.AL`:
- `p5shim.js` — just enough global-mode p5 (`Vec3`, `createVector`, `dist`,
  `random`, no-op draw calls) for the untouched `boids.js` to load.
- `boid3d.js` (`Boid3D extends Boid`), `vehicle3d.js` (`Vehicle3D`),
  `predator3d.js` (`Predator3D`), `food3d.js`, `obstacle3d.js`.
- These mirror the GPU data layout and act as a JS reference oracle; the live
  simulation does not use them.

## Do not modify
- `boids.js` and `vehicle.js` — original files, deliberately untouched; all 3D
  behaviour lives in the subclasses under `src/legacy/`.
- `libraries/p5.min.js` — vendored p5.js **v1.7.0**. Do not replace/reformat.
- French comments/identifiers in `boids.js` are intentional; keep that
  convention in the subclasses.

## Code constraints
- `boids.js` is written against p5 **global mode** (`createVector`, `dist`,
  `random`, `width`, `height`, …). It will not run in Node without `p5shim.js`.
- `Boid.debug` (static boolean) gates the debug drawing inside the original
  `avoid`, `wander`, `boundaries`.
- The WGSL `Boid3D.avoid()` look-ahead is scaled to `maxSpeed * 0.6` (the
  original `vel * 30` is only valid at 2D speeds), and it pushes **away from
  the obstacle centre**.
- `vehicle.js` was empty, so `Vehicle3D` is the 3D base that `Predator3D`
  extends.
- WebGPU limits bit us before: only **8 storage buffers per stage** (all grid
  scratch is packed into one `grid` buffer; render uses 6 storage + 1 uniform
  binding, plus a `grid` read for the hash-cell overlay), and `var<workgroup>`
  must be module-scope. `meta` and `target` are reserved WGSL words (renamed
  `extra` / `goal`). WGSL also requires explicit parentheses when mixing
  `*`/`^` or `^`/`+` in the same expression.
- A `drawIndexed` call requires a real index buffer; the models shader instead
  uses plain `draw(indexCount, instances)` and fetches indices from the pool.
- The spatial-hash neighbour search scans the cells within the **cohesion**
  radius (`2 * perception`) in each direction (`boidsUpdate` in
  `shaders/sim.wgsl`), clamped to 4 rings so a tiny cell size cannot become an
  unbounded device-losing sweep. It must be adaptive: a fixed 3×3×3 neighborhood
  silently misses neighbours whenever the cell is smaller than the perception
  radius, which stops flocking from working. The "Cell size" slider (5–400)
  trades scan cost against grid resolution.
- `cohesion()` uses **twice** the perception radius while `align()`/`separation()`
  use the plain radius — that asymmetry is in the original `boids.js`, and the
  GPU port must match it (there is a regression test for this).
- GLB model orientation is a guess: `orthoBasis(vel)` maps the model's local
  `+Z` to the velocity, which may not be each asset's authored forward. The
  "Boid/Predator model yaw (deg)" sliders correct this live.
- Textured models sample two shared atlases — base colour and emissive
  (`shaders/render/models.wgsl`, bindings 9/10 and 11/12). If either atlas fails
  to build the renderer falls back to a 1x1 texel (grey base, black emissive),
  so models stay tinted/unglowing rather than vanishing.
- `bakeModel` scales each mesh's vertex positions into **unit-radius** space
  before packing (uniform scale survives skinning: `joint * (s*p) = s*(joint*p)`),
  so the vertex shaders draw a model at exactly its physics radius and the drawn
  mesh lines up with its hitbox. "Boid/Predator radius" and "… model scale"
  therefore scale the model and its hitbox together.
- **Alpha cut-out restores the phoenix's feather gaps.** Both predator materials
  declare `alphaMode: BLEND` and `doubleSided: true`, and the base-colour PNGs are
  real RGBA: roughly 9–18% of each is fully transparent and another 3–7% partial —
  that transparency *is* the space between the feathers. `bakeModel` flattens the
  glTF `alphaMode`/`alphaCutoff` into the **previously-unused vertex `.w` slots**
  (mode ordinal at word `+3`, cutoff at `+7`), and `fsModel` `discard`s fragments
  whose sampled alpha falls below the cutoff whenever the mode is MASK or BLEND
  (BLEND uses the same threshold; a true blend would need sorted overdraw and a
  pipeline blend state). OPAQUE models — boid and obstacle — never cut out.
- The emissive atlas drives the predator/phoenix glow. It is added on top of the
  lit colour (so it reads at night) but treated as **lighting-independent per
  glTF**, and multiplied by the material's own `emissiveFactor` (carried in
  `ModelInfo.mat.xyz`) before the "Model glow strength" slider (`ModelInfo.extra.y`)
  scales it. The albedo is the sampled texture times the GLB `baseColorFactor`
  (`ModelInfo.mat.w`), so the model's real colours show; the old hard-coded
  `PREDATOR_TINT`/`* 1.25` boost is gone and the type tint is only a 8% nudge for
  readability.
- Obstacles are generated at **varied sizes** (each 0.4×–1.8× the "Obstacle
  radius" slider, stored per-instance in `Point.posR.w`). The obstacles get their
  own hitbox ring in the `hitbox` debug overlay.
- Food clusters are scattered across the whole world with a **central dead-zone**
  (18% of the smallest world dimension) so they never all pile up in the middle.
  Reaching a cluster is a **flock-wide** effect: it lifts the health of *every*
  living boid by the "Feed heals flock (%)" value (default **5%**, clamped at
  100), so the population average rises by that much. `boidsUpdate` only counts
  feed events into `grid[O_SPARE]`; the `flockHealUpdate` kernel then applies the
  boost to all boids and `gridClear` resets the counter next frame.
- **Population dynamics (deaths are spread out, not synchronised).** Every boid
  gets a stable per-slot *vitality* (0.5×–1.5× `healthDecayPerSec`) and
  *resilience* (0.6×–2.4× `hitsToKill`), derived from `hash1(slot)`, and spawns /
  respawns with **randomised health** (35–100). Without that variance every boid
  crossed zero on the same frame and the whole flock wiped and respawned in
  lock-step; with it, starvation and predation cull the flock a *fraction* at a
  time. The food/decay feedback then settles the living count in a band (about
  65–100% with the defaults) that the "Health decay /s", "Predator damage / hit"
  and "Feed heals flock (%)" sliders move up or down. The regression test
  `boid deaths spread out instead of wiping the flock at once` pins this.
- The skinned vertex path scales the **joint matrices** (left-multiplied by the
  unit-radius scale) rather than the input positions. Pre-scaling positions only
  scales the linear part of `joint * p` and leaves the joint translation at raw
  scale, which tears skinned meshes (predator/boid) into shreds; static meshes
  have no skin transform and are scaled from their stored positions.

## Known limits (verify before relying on them)
- In this container there is **no GPU adapter**, so WebGPU cannot be visually
  verified here. Correctness is covered by `npm test`, which includes an
  offscreen render-to-texture readback that asserts visible pixels, plus GLB
  bake tests. **Visual confirmation of the night sky, model textures and
  orientation, and the spatial-hash overlay must be done in a real browser.**
- Skinned boid draws are capped (`BOID_DRAW_CAP = 800` in `src/main.js`) for
  performance: the simulation still updates every boid, but only that many are
  rendered with the animated model.
- Original `boids.js` gaps, inherited by `Boid3D`: `avoid()`'s vehicle-vs-vehicle
  branch is commented out but still references the undefined
  `vehiculeLePlusProche`; `avoid()` reads `.pos`/`.r` from
  `getObstacleLePlusProche(obstacles)` and throws on an empty `obstacles` array;
  `boundaries()` computes `force` but returns `vitesseDesiree` (likely
  unintended).
