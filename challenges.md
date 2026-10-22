# challenges.md — Development challenges and how they were solved

A record of the non-obvious problems hit while building this, why they happened,
and what fixed them. Useful both as a "don't repeat this" list and as an
explanation of why parts of the code look the way they do. For the functional
spec see `specs.md`; for invariants see `AGENTS.md`.

---

## 1. Porting 2D boids to 3D without touching the original

**Challenge.** The original `boids.js` is 2D and had to stay byte-for-byte
untouched, along with a vendored p5.js. Yet the simulation had to become fully
3D.

**Approach.** Peeling the behaviours into subclasses only:
`Boid3D extends Boid`, then `Vehicle3D` as a 3D base, then `Predator3D`,
`Food3D`, `Obstacle3D`. `boids.js` is written against p5 **global mode**, so a
tiny `p5shim.js` supplies `Vec3`, `createVector`, `dist`, `random` and no-op draw
calls — just enough to load it in Node for the test harness.

**Subtleties that bit us.**

- The original `Boid3D.avoid()` look-ahead is `vel * 30`, which is only a sane
  distance at 2D speeds. In 3D it was scaled to `maxSpeed * 0.6`, and it must
  push **away from the obstacle centre** (using the raw position rather than the
  look-ahead point, which can land *inside* the obstacle).
- `cohesion()` uses **twice** the perception radius while `align()` and
  `separation()` use the plain radius. That asymmetry is easy to "fix" by
  accident and is pinned by a regression test.
- `boundaries()` in the original computes `force` but returns
  `vitesseDesiree` — a likely bug, deliberately preserved.

---

## 2. Living inside WebGPU's limits

**Challenge.** WebGPU imposes several limits that a naive port walks straight
into.

**What we hit.**

- **Only 8 storage buffers per stage.** The spatial hash needs several scratch
  arrays (cell starts, cursors, block sums, sorted indices). Rather than give each
  its own buffer, they are all packed into **one** `grid` storage buffer at fixed
  offsets (`O_STARTS`, `O_CURSOR`, `O_PARTIAL`, `O_ITEMS`, `O_SPARE`). The render
  side likewise uses 6 storage + 1 uniform binding, plus a `grid` read for the
  hash-cell overlay.
- **`var<workgroup>` must be module-scope.** No block-local workgroup variables.
- **Reserved words.** `meta` and `target` are reserved in WGSL and had to be
  renamed (`extra` / `goal`).
- **Explicit parentheses** are required when mixing `*`/`^` or `^`/`+` in one
  expression.
- **`drawIndexed` needs a real index buffer.** There isn't one: the models shader
  instead uses plain `draw(indexCount, instances)` and fetches indices from the
  packed pool itself.

**Grid sizing.** The two-level prefix scan must stay within a single workgroup of
block sums, so the cell count is capped at **65536**; `computeGridDims` shrinks
the resolution uniformly if a small `gridCellSize` would exceed that.

---

## 3. The spatial-hash neighbourhood must be adaptive

**Challenge.** The neighbour search originally scanned a fixed **3×3×3** cell
neighbourhood. Whenever the cell was smaller than the perception radius, this
silently missed neighbours — and because missed neighbours mean no
alignment/cohesion, **flocking simply stopped working**. It looked like a physics
bug but was a grid bug.

**Fix.** Scan the cells within the **cohesion** radius (`2 × perception`) in each
direction, clamped to **4 rings** so a pathologically tiny cell size cannot become
an unbounded, device-losing sweep. The "Cell size" slider (5–400) then trades scan
cost against grid resolution.

---

## 4. Predator orientation: "the face doesn't follow the heading"

This was the longest-running and most misleading bug. It turned out to be
**three** separate defects stacked on top of each other.

### 4a. The basis *was* correct — the asset's axis wasn't

**Symptom.** Predator models seemed to "rotate in space sporadically" instead of
facing their direction of travel.

**False lead.** The suspicion fell on `orthoBasis`, the function that maps the
model's local `+Z` onto the heading. Numerically it was fine: across 14 400
sampled headings `B · (+Z)` equalled the heading to `0.000000`, and the roll
reference was continuous across the old `|fwd.y| > 0.9` threshold (a change of
`0.000082`). An early hand-written check appeared to show a `2.0` error, but that
was a **column-indexing mistake in the check itself** (WGSL `mat3x3(a,b,c)` is
column-major, so `m*v = a*v.x + b*v.y + c*v.z`). The basis was never the bug.

**The real cause.** Each asset's authored forward is not `+Z`:

- the **boid's** body runs along `+Y`, so it needs `boidModelPitch = 90°`;
- the **phoenix** is a different Sketchfab/FBX rig. Its *skeleton* is the ground
  truth: the head bone `b_Head_06` sits at the extreme **`+X`** end and the tail
  chain `B_Tail_5_034` trails to `-X`, so the bird faces **`+X`** and needs
  `predatorModelYaw = -90°` with pitch 0.

**Why it took so long.** Guessing the facing from raw geometry is a trap. The
predator's `+Y` spike looks like a tail but is actually the **crest**
(`B_Hair_*` bones); reading it as a tail points you at the wrong axis entirely. A
first attempt at a fix set `predatorModelPitch = -90` on exactly that mistaken
reading. The reliable method was to read the **bone names and their world
positions** out of the glTF and anchor the assertion to head-vs-tail, which is
what the regression test now does.

A useful side lesson: two diagnostic scripts that appeared to disagree did so
because one had sign-convention drift in its re-implementation of
`applyModelYawPitch`. Re-deriving from the WGSL source text settled it.

### 4b. Zero-velocity respawns

**Symptom.** Even with the axis corrected, a predator would occasionally snap its
orientation.

**Cause.** `makeAgent` created agents with `velD = 0`. The renderer takes the
facing from `velD.xyz`, and `safeNormalize` falls back to a fixed `+Z` for a
near-zero vector — so a freshly respawned predator was drawn at an arbitrary
orientation and then *snapped* to its true heading as it accelerated.

**Fix.** `makeAgent` now takes a `spawnVel`; both respawn paths spawn heading
toward the world centre at a fraction of max speed, so a spawn renders coherently
from its first frame.

### 4c. Sporadic heading jitter (see §5)

---

## 5. Predator boundary turns were instant about-faces

**Symptom.** Predators "turned the opposite direction too abruptly" at the box
boundary, and sometimes kept heading into the wall even with no boids in front of
them.

**Causes — two of them.**

1. **Hard-swapped boundary steer.** On entering the edge margin, the code
   *overwrote* `desired` with a velocity pointing straight off that wall and
   applied it with a large weight. A predator chasing prey at a wall therefore
   made a **near-180° turn in a single frame** (measured at **139.5°/frame**).
2. **Wander re-rolled every frame.** The predator's wander target was computed
   from `hash2(i, frame)`, which changes every tick — so the target jumped to a
   random point on the perpendicular circle each frame and the heading jittered
   constantly. The boids never had this problem because they carry a *persistent*
   `theta` in `info.z` and nudge it only slightly per frame. The predator was
   re-rolling the angle from scratch each time.
3. **Targeting prey behind it.** A predator locked onto the *nearest* boid even
   when that boid was behind it, so it would drive toward the wall behind itself
   instead of turning.

**Fixes.**

- The boundary steer now grows a purely inward acceleration scaled by **how deep
  into the margin** the predator is *and* **how much of its velocity points out of
  the box**, applied additively like every other steering term. The turn eases in:
  measured max turn dropped from **139.5°/frame** to **~13°/frame** (well under a
  30° guard).
- The wander angle is now **persistent**, carried in `velD.w` — a slot that is
  free for predators because they never accumulate hits (boids use it for hits).
- A predator only targets prey whose direction has a **positive dot** with its own
  heading, so it never chases something behind it into a wall.

The regression test reverted against the old code reproduces the 139.5° snap, so
it genuinely pins the fix rather than restating the new logic.

---

## 6. Skinned meshes tear when scaled naively

**Challenge.** Each mesh is normalised to radius 1 so the shader can scale it by
the physics radius and line the drawn mesh up with its hitbox. For static meshes
you just scale the positions. For **skinned** meshes that tears the model into
shreds.

**Cause.** A skinned vertex is `sum_k w_k · (joint_k · p)`. Pre-scaling `p` only
scales the *linear* part of `joint · p` and leaves the joint **translation** at
raw scale, so the parts no longer meet.

**Fix.** Scale the **joint matrices** on the left (`S · joint`) instead. That
scales the whole skinned output — linear part and translation together — so the
mesh stays intact, and uniform scale survives skinning because
`joint · (s·p) = s·(joint · p)`.

---

## 7. The phoenix looked "webbed" instead of feathered

**Challenge.** The phoenix's feather gaps looked filled in, unlike the original
asset.

**Cause.** Both predator materials declare `alphaMode: BLEND` and
`doubleSided: true`, and the base-colour PNGs are real RGBA — roughly 9–18% fully
transparent and another 3–7% partial. **That transparency *is* the space between
the feathers.** Rendering it opaquely webbed the gaps closed.

**Fix.** Alpha **cut-out**: `bakeModel` flattens the glTF `alphaMode`/`alphaCutoff`
into previously-unused vertex `.w` slots, and `fsModel` `discard`s fragments below
the cutoff for MASK/BLEND. Opaque models (boid, obstacle) never cut out. A true
blend would need sorted overdraw and a pipeline blend state, which was judged out
of scope; the cut-out reproduces the look.

---

## 8. Glow and colour fidelity

**Challenge.** The predator's glow and colours did not match the original asset;
an early version used hard-coded tints and a `* 1.25` boost.

**Fix.** Honour the GLB exactly:

- the emissive atlas is added on top of the lit colour so it reads at night, but
  treated as **lighting-independent per glTF**, multiplied by the material's own
  `emissiveFactor` (carried in `ModelInfo.mat.xyz`);
- the albedo is the sampled texture times the `baseColorFactor`
  (`ModelInfo.mat.w`);
- the hard-coded `PREDATOR_TINT` / `* 1.25` boost was removed; the type tint is
  only an 8% readability nudge.

---

## 9. Bloom: "radius does nothing; the other sliders just brighten pixels"

**Challenge.** The bloom sliders misbehaved in a very specific way: **radius had
no effect at all**, while threshold and strength just seemed to make pixels
brighter rather than spread a glow.

**Cause 1 — a shared uniform buffer across passes.** All four post passes shared
one `postUniform`. Every `queue.writeBuffer` is queued *before* the single
`submit`, so by the time the GPU ran the passes they all read the **last** value
written. The blur therefore sampled with `texelStep = 0` — all 13 taps landed on
one texel, producing no blur, so radius did nothing and the composite simply
re-added the already-bright pixels.

**Cause 2 — no HDR headroom.** Even with the blur fixed, the scene's peak
luminance was **0.929** with only ~12 pixels above 0.85, so the bright pass
selected almost nothing (measured bloom energy ≈ 0.1). The night sky and lit
geometry sit near 0–0.3; only the emissive glow goes higher, and `shade()` alone
tops out near 1.0.

**Fixes.**

- Each post pass now has its **own** uniform buffer (`bright`, `blurH`, `blurV`,
  `composite`), each with its own bind group.
- `fsModel` scales the emissive term by `EMISSIVE_HDR = 2.5`, pushing glowing
  materials into genuine HDR (peak 0.929 → 1.447; 77 pixels above 1.0).
- The default `bloomThreshold` was lowered from 0.85 to **0.4**, chosen from a
  sweep on the real assets: 0.85 produced no visible effect, 0.4 a clear glow,
  0.2 an over-hazed image.

**Verification.** With real predator textures, raising the radius from 1 to 4 grew
the halo (7326 → 8668) while *lowering* the saturated core (122 → 89) — i.e. the
energy spread outward rather than just brightening, which is exactly what a
working blur does. The regression test now measures halo vs core and fails with
the shared-uniform bug (`halo 46231 … vs 46231`).

Two further subtleties: the blur offsets scale with `bloomRadius` so the slider
widens the halo, and half-float readback (`rgba16float`) must be decoded as
`Uint16Array` — reading it as `Float32Array` yields wrong near-zero values.

---

## 10. The bloom read as a light *streak*, not a mist

**Challenge.** With the blur working, the glow still looked like a hard little
streak rather than haze. The radial profile of the pure bloom layer (measured
by rendering the scene with and without bloom and subtracting) made the cause
obvious: the profile was **not monotone**. Around the source it fell from 85.6
to 27.4 at 4 px and then *rose again* to 29.0 at 8 px — a ring of secondary
peaks.

**Cause — a gapped kernel.** The blur used 6 taps spaced at `k^1.35` texels, so
the taps sat at roughly 2, 5, 9, 14, 19 and 25 screen pixels. The kernel
therefore did not blur the source at all: it **copied** it to a handful of
discrete positions. What the eye reads as "a streak" is exactly that row of
ghost copies. On top of that the bright pass point-sampled a single
half-resolution texel, so the kernel had almost nothing to smear.

**Fix — a two-scale, contiguous kernel.**

- The bright pass now **dilates** the source with a 5-tap cross (±1 bloom
  texel). Without it the blur's nearest tap lands a whole texel away and can
  sample *past* the one-texel-wide source.
- The blur's taps are **contiguous** — tap *k* at *(k − 0.5)* texel steps — so
  every position between the centre and the far edge is sampled. The kernel is
  a sum of two Gaussians: a tight **core** (σ ≈ 1.2 texels) and a wide **veil**
  (σ = 3 + 6 × `bloomRadius` texels), weighted 40 / 60. The core keeps the
  source reading as a glow; the veil supplies the haze. A single-scale blur can
  only be one or the other — tight reads as a hot blob, broad as a smear.
- The composite **desaturates** the glow toward its own luminance (35 %) so it
  reads as suspended mist rather than a coloured streak. An earlier attempt
  mixed the glow toward *white* by a constant 0.14, which added 0.14 to
  **every** pixel of the frame — a full-screen white wash, not bloom. Mixing
  toward the glow's own luminance is luminance-preserving and does nothing to
  the dark background.

**Result** (same scene, radius 1): the radial profile is now monotone
(70 → 22 → 21 → 6 → 3), the anisotropy improves from 0.874 to 0.935 (and to
0.993 at radius 2), and the glow reaches further out.

**Measurement lesson.** The first attempt at a wider kernel hung the headless
Dawn run — and the *original* shader hung just as often. The flake was in the
container's software backend, not in the shader. Chasing it as a shader bug
would have been a wild goose chase; the fix was to run one radius per process.
See also §15.

---

## 11. Hiding the control UI

**Challenge.** The control panel covered the right-hand side of the scene, and
there was no way to get an unobstructed view of the simulation.

**Fix.** A floating `#ui-toggle` button lives **outside** the panel (and
outside the stats overlay) in `index.html`, so it stays reachable once they
are hidden. Clicking it toggles `body.ui-hidden`; the CSS then slides the panel
out to the right and fades the stats overlay, while the button relabels itself
"Hide UI" / "Show UI" and carries `aria-expanded` for screen readers. The
button shifts left while the panel is open so it never covers it.

The **H** key does the same, except while a slider, number field or checkbox
has focus — otherwise typing in a control would dismiss the panel the user was
in the middle of using.

---

## 12. Population dynamics wiped the flock in lock-step

**Challenge.** The population didn't vary naturally: **every boid crossed zero
health on the same frame**, so the whole flock died and respawned at once.

**Fix.** Give every boid a stable per-slot **vitality** (0.5×–1.5×
`healthDecayPerSec`) and **resilience** (0.6×–2.4× `hitsToKill`) derived from
`hash1(slot)`, and spawn/respawn with **randomised health** (35–100). Starvation
and predation then cull the flock a *fraction* at a time; the food/decay feedback
settles the living count in a band (about 65–100% with the defaults). The test
`boid deaths spread out instead of wiping the flock at once` pins it.

---

## 13. Headless verification without a GPU

**Challenge.** The container has **no GPU adapter**, so WebGPU cannot be verified
visually. Any bug whose only symptom is "it looks wrong" is hard to catch.

**Mitigations.**

- `npm test` runs against the **Dawn** (`webgpu`) Node bindings and includes an
  **offscreen render-to-texture readback** that asserts visible pixels, plus
  models-only rendering, HDR headroom above the threshold, and a **bloom shape**
  check (the glow must reach dark pixels, the radius must widen it, and it must
  be round rather than a streak).
- Behavioural bugs (orientation, smooth turning, spread-out deaths, flocking
  rules) are pinned by numeric regression tests that fail against the old code.
- Where a visual judgement is unavoidable (night sky, texture fidelity, overlay
  appearance) a real browser is required, and that is called out explicitly in
  `specs.md` / `README.md`.

Diagnostic scripts must live **inside the repo** to resolve the `webgpu` npm
package; they were written, run, and cleaned up per investigation.

---

## 14. Packaging and deployment

**Challenge.** The project is static ES modules with no build step, but
`node_modules` (the test bindings) must never be committed, and GitHub Pages must
serve the shader sources correctly.

**Fix.** `.gitignore` excludes `node_modules`, and `.nojekyll` disables Jekyll so
Pages serves files beginning with `_` and the `src/`/`shaders/` directories
verbatim. Git history was rewritten once to purge an accidentally committed
`node_modules` tree.

---

## 15. A note on misleading measurements

Several investigations were slowed by **measurement** bugs rather than product
bugs, and it's worth naming the pattern:

- a hand-rolled mat3 multiply used the wrong column convention and "proved" the
  basis was broken when it wasn't (§4a);
- a diagnostic passed the camera yaw where the model yaw was expected, so two
  candidate orientations produced byte-identical images;
- reading half-float textures as `Float32Array` gave near-zero values (§9);
- the headless Dawn runs **hang intermittently** in this container — and the
  *unmodified* shader hung just as often as the rewritten one, which is what
  finally showed the flake was in the software backend rather than in the
  bloom code (§10). One radius per process made the measurements reliable.

The habit that resolved all three: derive the expected maths from the **shipped
WGSL source**, and assert against something the code cannot fake — bone
positions, halo-vs-core energy, per-frame turn deltas — rather than a plausible
screenshot.
