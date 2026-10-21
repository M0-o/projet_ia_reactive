// common.wgsl
// Shared structs and helpers for the artificial-life simulation.
// Composed with a bindings block and a kernel/render module at load time.

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;
const EPS: f32 = 1e-5;

// Spatial-hash scratch layout inside the single `grid` storage buffer.
// Keep in sync with GRID_WORDS in src/sim/bindings.js.
const MAX_CELLS: u32 = 65536u;
const SCAN_BLOCKS: u32 = 256u;          // MAX_CELLS / workgroup size (256)
const BOID_CAP: u32 = 4096u;
const O_COUNTS: u32 = 0u;
const O_STARTS: u32 = MAX_CELLS;                        // inclusive prefix sums
const O_CURSOR: u32 = 2u * MAX_CELLS;                   // scatter cursors
const O_PARTIAL: u32 = 3u * MAX_CELLS;                  // block sums
const O_ITEMS: u32 = 3u * MAX_CELLS + SCAN_BLOCKS;      // sorted boid indices
const O_SPARE: u32 = O_ITEMS + BOID_CAP;                // scratch cursor area
const GRID_WORDS: u32 = O_SPARE + 8u;

// One moving agent (boid or predator).
//   posH.w = health [0..100]
//   velD.w = accumulated hits taken
//   info  = (alive, age_seconds, wanderTheta, radius)
struct Agent {
  posH : vec4<f32>,
  velD : vec4<f32>,
  info : vec4<f32>,
};

// A static-ish point: food cluster or obstacle.
//   posR.w = radius (obstacle) / inner radius (food)
//   extra   = (active, timer, seed, energy)
struct Point {
  posR : vec4<f32>,
  extra : vec4<f32>,
};

// All tunable simulation parameters. Every field is a vec4 so that the
// WGSL uniform layout is trivial to mirror from JavaScript.
struct Params {
  counts    : vec4<u32>, // boids, predators, food, obstacles
  timing    : vec4<f32>, // dt, time, frame, timeScale
  motion    : vec4<f32>, // maxSpeed, maxForce, perception, separationRadius
  weights1  : vec4<f32>, // align, cohesion, separation, boundaries
  weights2  : vec4<f32>, // avoidObstacle, seekFood, fleePredator, wander
  boundsMin : vec4<f32>,
  boundsSize: vec4<f32>,
  health    : vec4<f32>, // decayPerSec, foodHeal, predatorDamage, hitsToKill
  eating    : vec4<f32>, // eatRadius, foodRespawnDelay, foodEnergy, boidRespawnDelay
  predator  : vec4<f32>, // maxSpeed, maxForce, vision, attackCooldown
  grid      : vec4<f32>, // cellSize, dimX, dimY, dimZ
  wanderP   : vec4<f32>, // distance, radius, displace, strength
  aux       : vec4<f32>, // reserved, reserved, boidRadius, predatorRadius
  aux2      : vec4<f32>, // foodRadius, obstacleRadius, attackRange, avoidWidth
  debug     : vec4<f32>, // headingScale, bboxScale, visionScale, hitboxScale
};

// ---------------------------------------------------------------------------
// Hashing / RNG (deterministic, per-invocation)
// ---------------------------------------------------------------------------

fn hash1(x0: u32) -> f32 {
  var x = x0;
  x = (x ^ 61u) ^ (x >> 16u);
  x = x + (x << 3u);
  x = x ^ (x >> 4u);
  x = x * 0x27d4eb2du;
  x = x ^ (x >> 15u);
  return f32(x) / 4294967295.0;
}

fn hash2(a: u32, b: u32) -> f32 {
  return hash1(a ^ (b * 0x9e3779b9u));
}

// vec3 hash for integer lattice cells (used by the star field).
fn hash3(c: vec3<f32>) -> vec3<f32> {
  let ix = bitcast<u32>(i32(c.x));
  let iy = bitcast<u32>(i32(c.y));
  let iz = bitcast<u32>(i32(c.z));
  return vec3<f32>(
    hash1((ix * 73856093u) ^ (iy * 19349663u) ^ (iz * 83492791u)),
    hash1(((ix * 19349663u) ^ (iy * 83492791u) ^ (iz * 73856093u)) + 1u),
    hash1(((ix * 83492791u) ^ (iy * 73856093u) ^ (iz * 19349663u)) + 2u),
  );
}

fn rand3(seed: u32) -> vec3<f32> {
  return vec3<f32>(hash1(seed * 3u + 1u), hash1(seed * 3u + 2u), hash1(seed * 3u + 3u));
}

fn randUnit(seed: u32) -> vec3<f32> {
  // Uniformly distributed direction on the unit sphere.
  let u = hash1(seed * 5u + 1u) * 2.0 - 1.0;
  let t = hash1(seed * 5u + 2u) * TAU;
  let r = sqrt(max(0.0, 1.0 - u * u));
  return vec3<f32>(r * cos(t), r * sin(t), u);
}

// ---------------------------------------------------------------------------
// Vector helpers
// ---------------------------------------------------------------------------

fn safeNormalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
  let l = length(v);
  if (l < EPS) { return fallback; }
  return v / l;
}

fn limitLen(v: vec3<f32>, m: f32) -> vec3<f32> {
  let l = length(v);
  if (l > m && l > EPS) { return v * (m / l); }
  return v;
}

fn setMagSafe(v: vec3<f32>, m: f32) -> vec3<f32> {
  let l = length(v);
  if (l < EPS) { return vec3<f32>(0.0); }
  return v * (m / l);
}

// `vel` is the spawn velocity. A zero velocity is a problem for the renderer:
// the model's facing is taken from `velD.xyz`, so a just-spawned agent would be
// drawn at the fixed +Z fallback heading and then snap to its real heading once
// it accelerates — a visible orientation pop. Spawns therefore get a real
// (if arbitrary) heading so they render coherently from their first frame.
fn makeAgent(pos: vec3<f32>, radius: f32, vel: vec3<f32>) -> Agent {
  var a: Agent;
  a.posH = vec4<f32>(pos, 100.0);
  a.velD = vec4<f32>(vel, 0.0);
  a.info = vec4<f32>(1.0, 0.0, 0.0, radius);
  return a;
}

fn makeDeadAgent(radius: f32) -> Agent {
  var a: Agent;
  a.posH = vec4<f32>(0.0);
  a.velD = vec4<f32>(0.0);
  a.info = vec4<f32>(0.0, 0.0, 0.0, radius);
  return a;
}

// ---------------------------------------------------------------------------
// Uniform-grid (spatial hash) helpers
// ---------------------------------------------------------------------------

fn gridDims() -> vec3<i32> {
  return vec3<i32>(i32(P.grid.y), i32(P.grid.z), i32(P.grid.w));
}

fn cellCoord(p: vec3<f32>) -> vec3<i32> {
  let rel = (p - P.boundsMin.xyz) / max(P.grid.x, EPS);
  let dim = gridDims();
  var c = vec3<i32>(floor(rel));
  c = min(max(c, vec3<i32>(0)), dim - vec3<i32>(1));
  return c;
}

fn cellLinear(c: vec3<i32>) -> u32 {
  let dim = gridDims();
  return u32(c.x) + u32(c.y) * u32(dim.x) + u32(c.z) * u32(dim.x * dim.y);
}

fn cellIndex(p: vec3<f32>) -> u32 {
  return cellLinear(cellCoord(p));
}
