# specs.md — Functional specification

What the project *is*, what it *does*, and how each piece behaves. For the
development narrative (the problems and how they were solved) see `challenges.md`.
For day-to-day orientation and invariants see `AGENTS.md`.

---

## 1. What it is

A 3D artificial-life simulation rendered with **WebGPU**. A population of
flocking agents ("boids") drifts through a bounded box, feeds on scattered food
clusters, avoids obstacles and flees predators. Predators hunt the boids. The
whole scene sits under a procedural night sky with stars and a few debug
overlays you can toggle individually.

The defining constraint: **all simulation maths runs in WGSL compute shaders**,
and the app is **plain static ES modules** — no bundler, no build step, no
transpiler. `index.html` loads classic scripts (for the legacy reference classes)
and then `src/main.js` as a module. `package.json` exists only for the dev server
and the headless test harness.

### Non-goals / hard constraints

- `boids.js` and `vehicle.js` are **original files and must not be modified**;
  all 3D behaviour lives in subclasses under `src/legacy/`.
- `libraries/p5.min.js` is a vendored p5.js **v1.7.0** and must not be replaced or
  reformatted.
- French comments/identifiers in `boids.js` are intentional and are kept in the
  subclasses.
- `node_modules` must never be tracked in a commit.

---

## 2. How to run

```bash
npm run serve     # serves the repo at http://localhost:8080
npm test          # headless correctness suite (Dawn / `webgpu` bindings)
```

There is **no** build or lint step. WebGPU needs a secure context; `localhost`
qualifies. If WebGPU is unavailable (older browser, no adapter) the app shows a
human-readable message and a hint instead of a blank canvas
(`src/gpu/device.js` throws an error carrying a `userMessage`).

A real GPU adapter is not present in every environment (e.g. a headless CI
container), so the night sky, model textures/orientation and the spatial-hash
overlay are best confirmed in a WebGPU-capable browser. Correctness is covered by
`npm test`, which includes an offscreen render-to-texture readback.

---

## 3. Entities and world

The world is an axis-aligned box centred on the origin: bounds are
`[-half, +half]` on each axis, half-extents from the `worldX/Y/Z` sliders
(default **1000 × 1000 × 1000**). Positions are clamped into the box every step.

| Entity | Buffer layout | Notes |
| --- | --- | --- |
| Boid | `Agent` (3 × `vec4`) | flocking agent, 4096 capacity |
| Predator | `Agent` | hunter, 128 capacity |
| Food cluster | `Point` (2 × `vec4`) | has an inner radius, 160 capacity |
| Obstacle | `Point` | each has its **own** per-instance radius, 128 capacity |

`Agent` (in `shaders/common.wgsl`):

```
posH : vec4<f32>   // xyz = position, w = health [0..100]
velD : vec4<f32>   // xyz = velocity, w = boids: hits taken / predators: wander angle
info : vec4<f32>   // (alive, age_seconds, wanderTheta / attackCooldown, radius)
```

`Point`:

```
posR  : vec4<f32>  // xyz = position, w = radius (obstacle) / inner radius (food)
extra : vec4<f32>  // (active, timer, seed, energy)
```

All parameter values live in `src/sim/params.js` (`DEFAULTS` + `SCHEMA`) and are
packed by `ParamPacker` into a `Params` uniform of 15 × `vec4` = **240 bytes**,
mirroring the WGSL struct exactly.

---

## 4. Simulation behaviour

Everything below is implemented in `shaders/sim.wgsl` as **11 compute entry
points**, dispatched in order each step from `src/sim/world.js`:

```
gridClear → gridCount → gridScanLocal → gridScanPartial → gridScanAdd
→ gridCursorInit → gridScatter → boidsUpdate → predatorsUpdate
→ foodUpdate → flockHealUpdate
```

### 4.1 Spatial hash (neighbour queries)

A GPU **counting sort** over a uniform grid, with a two-level prefix scan:

1. `gridClear` zeroes the cell counters (and resets the feed-event counter).
2. `gridCount` counts boids per cell.
3. `gridScanLocal` / `gridScanPartial` / `gridScanAdd` build the inclusive prefix
   sums (a workgroup-level scan plus block offsets).
4. `gridCursorInit` turns the prefix sums into scatter cursors.
5. `gridScatter` writes each boid's index into its cell's slot.

All scratch lives in a **single** `grid` storage buffer (offsets `O_STARTS`,
`O_CURSOR`, `O_PARTIAL`, `O_ITEMS`, `O_SPARE`) so the whole simulation stays
within WebGPU's **8 storage buffers per stage** limit.

Neighbour search is **adaptive**: `boidsUpdate` scans the cells within the
**cohesion** radius (`2 × perception`) in each direction, clamped to **4 rings**
so a tiny cell size cannot become an unbounded, device-losing sweep. A fixed
3×3×3 neighbourhood would silently miss neighbours whenever the cell is smaller
than the perception radius and flocking would break.

### 4.2 Boids — the flocking rules

A 3D port of the original 2D `boids.js` behaviours, matching its specification
(there is a regression test for the one asymmetry below):

| Steering | Radius used | Meaning |
| --- | --- | --- |
| `align` | perception | steer toward neighbours' average heading |
| `separation` | perception | steer away from close neighbours |
| `cohesion` | **twice** perception | steer toward neighbours' centroid |
| `boundaries` | edge-awareness margin | steer back inside the box |
| `avoid` | avoid margin | steer around obstacles |
| `seekFood` | — | steer toward the nearest active cluster (stronger when hungry) |
| `fleePredator` | predator vision | steer away from the nearest predator within range |
| `wander` | wander sphere | random walk on a sphere perpendicular to the heading |

`cohesion()` uses **twice** the perception radius while `align()`/`separation()`
use the plain radius — that asymmetry is in the original `boids.js` and the GPU
port must match it. The obstacle look-ahead is scaled to `maxSpeed * 0.6` (the
original `vel * 30` is only valid at 2D speeds) and pushes **away from the
obstacle centre**.

Integration order matches `Boid.update` in the original: `pos += vel`, then
`vel += acc` (acceleration clamped by `limitLen` to `maxForce`), then clamp the
position into the box.

### 4.3 Health, death and respawn

- Every boid has health in `[0..100]`; it decays by `healthDecayPerSec`.
- A **stable per-slot vitality** (0.5×–1.5× decay) and **resilience**
  (0.6×–2.4× `hitsToKill`) are derived from `hash1(slot)`, and a boid spawns /
  respawns with **randomised health** (35–100). Without that variance every boid
  crossed zero on the same frame and the whole flock wiped and respawned in
  lock-step; with it, deaths are a spread-out stochastic process and the living
  count settles in a band (roughly 65–100% with the defaults).
- A predator within attack range deals `predatorDamage` and counts a hit; after
  `hitsToKill` hits the boid dies (cooldown `attackCooldown` between hits).
- A dead boid waits `boidRespawnDelay` seconds, then respawns at a random
  position with fresh randomised health and a **non-zero spawn velocity** (see
  §4.6).
- Predators likewise die and respawn on a timer.

### 4.4 Food and flock-wide feeding

- Food clusters are scattered across the whole world with a **central
  dead-zone** (18% of the smallest world dimension) so they never all pile up in
  the middle.
- Reaching a cluster is a **flock-wide** effect: `boidsUpdate` records a feed
  event in `grid[O_SPARE]` and consumes the cluster; the `flockHealUpdate` kernel
  then lifts the health of **every living boid** by `foodHeal` (default **5%**,
  clamped at 100). So the population *average* rises by that amount, rather than
  one boid being healed.
- `gridClear` resets the counter next frame.
- An eaten cluster regrows after `foodRespawnDelay`.

### 4.5 Predators — hunting

- Each frame a predator scans all boids and targets the **nearest one that is
  ahead of it** (positive dot product with its own heading). It only targets prey
  that is genuinely in front of it.
- If a target is within `predatorVision * 3`, the predator steers toward it
  (arrival steering: `setMagSafe(goal - pos, speed) - vel`, force clamped to
  `predatorForce`).
- Otherwise it **wanders**, using a *persistent* angle nudged a little each frame
  (the same scheme the boids use), so its heading drifts instead of jittering.
- When a predator is inside the **edge margin**, it grows a purely inward
  acceleration that scales with how deep it is into the margin *and* how much of
  its velocity points out of the box. The in-bounds turn therefore **eases in**
  rather than snapping around. See `challenges.md` §4 for why this matters.
- Predators never accumulate hits (that slot carries the wander angle), and they
  are drawn with an emissive glow.

### 4.6 Orientation: models follow their heading

The model's facing is taken from `velD.xyz` and mapped to a **continuous**
orientation:

- `orthoBasis(vel)` (in `shaders/render_common.wgsl`) maps the model's local
  `+Z` onto the heading as a **minimum rotation** (Rodrigues shortest arc) from
  world `+Z`. The roll reference varies smoothly in every direction; an earlier
  version hard-switched its reference up at `|fwd.y| > 0.9`, which snapped the
  roll — a visible flip. A regression test sweeps headings across that threshold
  and fails if the basis jumps.
- Because each asset's authored forward differs from `+Z`, a per-model yaw/pitch
  correction (`applyModelYawPitch`) rotates the mesh onto local `+Z`, exposed
  live as sliders:
  - **boid**: body runs along `+Y`, so default **pitch 90°**;
  - **predator (phoenix)**: a Sketchfab/FBX rig whose head bone (`b_Head_06`) is
    at the extreme `+X` end and whose tail chain (`B_Tail_5_034`) trails to
    `-X` — it faces `+X`, so default **yaw -90°**, pitch 0.
- **Spawns carry a real heading.** An agent born with a zero velocity would be
  drawn at the fixed `+Z` fallback and then *snap* to its true heading as it
  accelerated, so both respawn paths spawn heading toward the world centre at a
  fraction of max speed.

---

## 5. Rendering

Every pipeline is a **single instanced / procedural draw** — the models shader
reads vertices, indices and joint frames from one storage buffer, so no
vertex/index buffer is ever bound.

### 5.1 Scene

- **Night sky** (`render/sky.wgsl`): a full-screen gradient from deep navy at the
  horizon to near-black at the zenith, plus a deterministic star field with a few
  slowly twinkling stars.
- **Models** (`render/models.wgsl`): instanced skinned GLB rendering + texture
  atlas sampling. Skinning blends two bracketing baked animation frames; each
  instance gets a random phase offset so the flock does not flap in unison.
- **Food**, **bounding box**, **debug lines** (heading / vision / hitboxes /
  bboxes), and **spatial-hash cells**.

### 5.2 GLB pipeline

- `src/io/gltf.js` — a small dependency-free GLB/glTF reader (accessors, node
  hierarchy, animation sampler evaluation, embedded image bytes, base-colour /
  emissive material lookup).
- `src/render/models.js` — bakes a GLB into the flat **model pool**: flattens
  mesh + skin data, evaluates the animation at N evenly-spaced times, stores each
  joint's `worldJoint × inverseBind` matrix per frame, then packs verts / skin /
  indices / joint-frames into one `Float32Array`. Each mesh is normalised to
  **radius 1** so the shader scales it by a physics radius and the drawn mesh
  lines up with its hitbox. `collectModelImages` pulls the base-colour PNGs back
  out; `applyAtlas` rewrites each vertex's UV to sample its cell in the atlas.
- `src/render/textures.js` — packs every GLB base-colour image **and** its
  emissive ("glow") map into two bilinear texture atlases (512 px cells), so a
  multi-material mesh still draws in a single call. If an atlas fails to build,
  the renderer falls back to a 1×1 texel (grey base, black emissive) so models
  stay tinted/unglowing rather than vanishing.

**Skinned meshes are scaled via the joint matrices**, left-multiplied by the
unit-radius scale. Pre-scaling input positions only scales the linear part of
`joint * p` and leaves the joint translation at raw scale, which tears skinned
meshes into shreds; static meshes have no skin transform and are scaled from
their stored positions.

**Alpha cut-out** restores the phoenix's feather gaps. Both predator materials
declare `alphaMode: BLEND` and `doubleSided: true`, and the base-colour PNGs are
real RGBA (roughly 9–18% fully transparent). That transparency *is* the space
between the feathers. `bakeModel` flattens the glTF `alphaMode`/`alphaCutoff`
into previously-unused vertex `.w` slots, and `fsModel` `discard`s fragments
below the cutoff for MASK/BLEND (opaque models never cut out). A true blend would
need sorted overdraw and a pipeline blend state, which is out of scope.

**Glow and albedo honour the GLB.** The emissive atlas is added on top of the lit
colour (so it reads at night) but treated as **lighting-independent per glTF**,
multiplied by the material's own `emissiveFactor`; the albedo is the sampled
texture times the `baseColorFactor`. There is no hard-coded predator tint beyond
an 8% readability nudge.

### 5.3 HDR bloom — a hazy mist, not a streak

- The scene renders into an **rgba16float** target so emissive can exceed 1.0 and
  the bright pass has headroom.
- `render/bloom.wgsl` runs three fragment passes:
  - `fsBright` — soft-knee threshold + half-res downsample. It also **dilates**
    the source with a 5-tap cross (±1 bloom texel) so the blur has a source to
    grab: a bright pixel usually lands in a single half-res texel, and a kernel
    whose first off-centre tap is a whole texel away would sample past it and
    get nothing back.
  - `fsBlur` — a separable **two-scale** kernel, run horizontally then vertically
    by ping-ponging two half-res textures:
    - a **core** — a tight Gaussian (σ ≈ 1.2 texels) that keeps the source
      reading as a glow;
    - a **veil** — a much wider Gaussian (σ = 3 + 6 × `bloomRadius` texels) that
      supplies the soft haze.
    The two are summed (40 % core / 60 % veil). A single-scale blur can only be
    tight (a hot blob) or broad (a washed-out smear); the pairing is what reads
    as suspended mist.
    The taps are **contiguous** — tap *k* sits at *(k − 0.5)* texel steps — so
    the kernel never skips a position between the centre and the far edge.
  - `fsComposite` — scene + bloom → swapchain. It only *adds* the glow; it
    deliberately does **not** tonemap, so enabling bloom does not change how the
    rest of the scene looks. The glow is first **desaturated** toward its own
    luminance (35 %) so it reads as mist rather than a coloured streak.
- A texture may not be sampled and used as a colour target in the same pass, so
  each bloom pass binds the scene into the unused second texture slot.
- Bloom keying is **relative to the scene's real range**: the night sky and most
  lit geometry sit near 0–0.3; only the emissive glow and the brightest
  highlights go higher. `fsModel` scales the emissive term by `EMISSIVE_HDR`
  (**2.5**) to push glowing materials into genuine HDR; the default
  `bloomThreshold` is therefore **0.4** (at 0.85+ so few pixels qualify that the
  blur has no energy to spread and bloom is invisible).
- **Each post pass needs its own uniform buffer** (`bright`, `blurH`, `blurV`,
  `composite`): all `queue.writeBuffer` calls are queued before the single
  `submit`, so a shared buffer would let every pass read only the last-written
  value.

### 5.4 Debug / effect overlays

Six independent toggles (`src/ui/panel.js`): heading vectors, vision radius,
hitboxes, bounding boxes, **spatial-hash cells** (the occupied cells of the
neighbour grid), and **bloom**. Only the enabled ones are drawn. Obstacles get
their own hitbox ring.

---

## 6. User interface

- **Camera**: drag to orbit, scroll to zoom, shift/middle-drag to pan
  (`src/render/camera.js`, an orbit camera producing a 256-byte `Camera` uniform).
- **Control panel** (`src/ui/panel.js`) is generated from `SCHEMA` in
  `src/sim/params.js` and exposes every tunable (population, world size, motion,
  weights, health, feeding, predator, spatial hash, scale/orientation/glow,
  bloom, time scale), plus Pause / Reset / Randomize-seed.
- **Hide-UI button**: a floating `#ui-toggle` button sits outside the panel (and
  outside the stats overlay) so it stays reachable when they are hidden. It
  toggles `body.ui-hidden`, which slides the panel out to the right and fades
  the stats overlay; the button relabels itself "Hide UI" / "Show UI" and
  carries `aria-expanded`. The **H** key does the same, except while a slider,
  number field or checkbox has focus.
- **Stats overlay** (`src/ui/stats.js`) reads the boid buffer back at ~250 ms
  intervals and shows counts/FPS.

### Notable defaults

| Parameter | Default |
| --- | --- |
| World | 1000 × 1000 × 1000 |
| Boids / predators / food / obstacles | 600 / 6 / 28 / 16 |
| Boid max speed / force | 150 / 90 |
| Predator speed / force / vision | 210 / 120 / 260 |
| Health decay | 16 / s |
| Feed heals flock | 5 % |
| Predator damage / hits to kill | 40 / 4 |
| Grid cell size | 120 |
| Boid model scale / pitch | 2.2 / 90° |
| Predator model scale / yaw | 6.0 / -90° |
| Bloom threshold / strength / radius | 0.4 / 1.0 / 1.5 |

- Obstacles are generated at **varied sizes** (each 0.4×–1.8× the "Obstacle
  radius" slider, stored per-instance).
- Skinned boid draws are capped at `BOID_DRAW_CAP = 800` (`src/main.js`) for
  performance: the simulation still updates every boid, but only that many are
  drawn with the animated model.

---

## 7. Legacy 3D reference classes

`src/legacy/` are classic scripts (loaded by `index.html`, exposed on
`globalThis.AL`) that mirror the GPU data layout and act as a **JS reference
oracle**; the live simulation does not use them.

- `p5shim.js` — just enough global-mode p5 (`Vec3`, `createVector`, `dist`,
  `random`, no-op draw calls) for the untouched `boids.js` to load in Node.
- `boid3d.js` (`Boid3D extends Boid`), `vehicle3d.js` (`Vehicle3D`),
  `predator3d.js`, `food3d.js`, `obstacle3d.js`.

Known gaps inherited from the original `boids.js` (documented, not fixed, to keep
the original untouched): `avoid()`'s vehicle-vs-vehicle branch references an
undefined `vehiculeLePlusProche`; `avoid()` reads `.pos`/`.r` from
`getObstacleLePlusProche(obstacles)` and throws on an empty array;
`boundaries()` computes `force` but returns `vitesseDesiree`.

---

## 8. Testing

`test/harness.mjs` runs headlessly against the **Dawn** (`webgpu`) Node bindings
and checks:

- shader compilation for the compute module and every render module;
- the spatial-hash sort / two-level scan;
- boid behaviour: bounds, health decay, feeding (flock-wide), obstacle
  avoidance, predator hits, spread-out deaths;
- flocking correctness against the original `boids.js` (including the cohesion
  radius asymmetry);
- GLB parsing and animation baking;
- the JS 3D reference classes against the WGSL behaviour;
- **orientation**: `orthoBasis` is orthonormal and continuous, and the predator's
  default orientation points its head along its heading;
- **steering**: a respawned predator has a well-defined heading, and a predator
  turns smoothly at the boundary rather than snapping around;
- an **offscreen render-to-texture readback** asserting visible pixels, plus
  models-only rendering, HDR headroom above the bloom threshold, and a **bloom
  shape** check: the glow must reach pixels that were dark before, the radius
  slider must widen it (the bloom layer's rms radius grows), and the glow must
  be **round** rather than a streak.

Currently **44 tests pass**. There is no GPU adapter in a headless container, so
visual confirmation of the night sky, textures/orientation and overlays must be
done in a real browser.

---

## 9. Deployment

Hosted as static files (GitHub Pages is the target). `node_modules` is ignored
via `.gitignore`, and `.nojekyll` is present so Pages serves the `src/` and
`shaders/` directories verbatim. The visitor's browser must support WebGPU
(desktop Chrome/Edge 113+, Chrome on Android); other browsers show the graceful
"No WebGPU adapter" message.
