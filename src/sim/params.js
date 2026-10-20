// params.js
// The single source of truth for every tunable value. The control panel is
// generated from SCHEMA, and packParams() mirrors the WGSL `Params` struct in
// shaders/common.wgsl (15 x vec4 = 240 bytes) exactly.

export const DEFAULTS = {
  // population
  boidCount: 600,
  predatorCount: 6,
  foodCount: 28,
  obstacleCount: 16,

  // world box (x, y, z)
  worldX: 1000,
  worldY: 1000,
  worldZ: 1000,

  // motion
  maxSpeed: 150,
  maxForce: 90,
  perception: 90,
  boundariesDistance: 120,

  // flocking / steering weights
  alignWeight: 1.0,
  cohesionWeight: 0.9,
  separationWeight: 1.7,
  boundariesWeight: 9.0,
  avoidWeight: 7.0,
  seekFoodWeight: 1.3,
  fleePredatorWeight: 3.2,
  wanderWeight: 0.35,

  // health & damage
  healthDecayPerSec: 1.6,
  foodHeal: 5,
  predatorDamage: 22,
  hitsToKill: 4,

  // feeding
  eatRadius: 22,
  foodRespawnDelay: 8,
  boidRespawnDelay: 4,

  // predator
  predatorSpeed: 210,
  predatorForce: 120,
  predatorVision: 260,
  attackCooldown: 0.8,

  // spatial hash
  gridCellSize: 120,

  // wander
  wanderDistance: 150,
  wanderRadius: 60,
  wanderDisplace: 0.5,

  // radii
  boidRadius: 12,
  predatorRadius: 18,
  foodRadius: 14,
  obstacleRadius: 45,
  attackRange: 6,
  avoidWidth: 80,

  // model appearance (multipliers on the baked GLB base scale)
  boidModelScale: 2.2,
  predatorModelScale: 6.0,
  // model orientation correction: rotates the mesh about its own up axis so a
  // model whose authored forward is not +Z flies nose-first.
  boidModelYaw: 0,
  predatorModelYaw: 0,
  // multiplier on the emissive (glow) map; 0 disables it
  modelGlow: 1.0,

  // simulation
  timeScale: 1.0,

  // debug overlay scales
  headingScale: 1.4,
  bboxScale: 1.8,
  visionScale: 1.0,
  hitboxScale: 1.0,
};

// Each entry: [groupLabel, key, label, min, max, step]
export const SCHEMA = [
  ["Population", "boidCount", "Boids", 0, 4000, 1],
  ["Population", "predatorCount", "Predators", 0, 60, 1],
  ["Population", "foodCount", "Food clusters", 0, 120, 1],
  ["Population", "obstacleCount", "Obstacles", 0, 80, 1],

  ["World", "worldX", "Size X", 60, 2000, 1],
  ["World", "worldY", "Size Y", 40, 2000, 1],
  ["World", "worldZ", "Size Z", 60, 2000, 1],

  ["Motion", "maxSpeed", "Boid max speed", 5, 600, 1],
  ["Motion", "maxForce", "Boid max force", 1, 400, 1],
  ["Motion", "perception", "Perception radius", 5, 300, 1],
  ["Motion", "boundariesDistance", "Edge awareness", 10, 500, 1],

  ["Weights", "alignWeight", "Alignment", 0, 4, 0.05],
  ["Weights", "cohesionWeight", "Cohesion", 0, 4, 0.05],
  ["Weights", "separationWeight", "Separation", 0, 6, 0.05],
  ["Weights", "boundariesWeight", "Boundaries", 0, 30, 0.1],
  ["Weights", "avoidWeight", "Avoid obstacles", 0, 30, 0.1],
  ["Weights", "seekFoodWeight", "Seek food", 0, 8, 0.05],
  ["Weights", "fleePredatorWeight", "Flee predator", 0, 12, 0.05],
  ["Weights", "wanderWeight", "Wander", 0, 3, 0.01],

  ["Health", "healthDecayPerSec", "Health decay /s", 0, 30, 0.1],
  ["Health", "foodHeal", "Feed heals flock (%)", 1, 100, 1],
  ["Health", "predatorDamage", "Predator damage / hit", 1, 100, 1],
  ["Health", "hitsToKill", "Hits to kill", 1, 20, 1],
  ["Health", "boidRespawnDelay", "Respawn delay (s)", 0, 20, 0.1],

  ["Feeding", "eatRadius", "Eat radius", 1, 120, 0.5],
  ["Feeding", "foodRespawnDelay", "Food regrow (s)", 0, 60, 0.5],

  ["Predator", "predatorSpeed", "Predator max speed", 5, 800, 1],
  ["Predator", "predatorForce", "Predator max force", 1, 600, 1],
  ["Predator", "predatorVision", "Predator vision", 10, 1200, 1],
  ["Predator", "attackCooldown", "Attack cooldown (s)", 0, 5, 0.05],

  ["Spatial hash", "gridCellSize", "Cell size", 5, 400, 1],

  ["Scale", "boidRadius", "Boid radius", 0.5, 60, 0.5],
  ["Scale", "predatorRadius", "Predator radius", 0.5, 80, 0.5],
  ["Scale", "foodRadius", "Food radius", 0.5, 80, 0.5],
  ["Scale", "obstacleRadius", "Obstacle radius", 1, 200, 0.5],
  ["Scale", "avoidWidth", "Avoid margin", 0, 400, 1],
  ["Scale", "boidModelScale", "Boid model scale", 0.1, 6, 0.05],
  ["Scale", "predatorModelScale", "Predator model scale", 0.1, 6, 0.05],
  ["Scale", "boidModelYaw", "Boid model yaw (deg)", -180, 180, 5],
  ["Scale", "predatorModelYaw", "Predator model yaw (deg)", -180, 180, 5],
  ["Scale", "modelGlow", "Model glow strength", 0, 4, 0.05],

  ["Simulation", "timeScale", "Time scale", 0, 4, 0.05],
];

export function makeParams() {
  return { ...DEFAULTS };
}

// Grid resolution for the spatial hash: cells roughly `gridCellSize` wide,
// capped so the two-level prefix scan stays within a single workgroup of
// block sums (<= 65536 cells total).
export function computeGridDims(params, maxCells = 65536) {
  const cell = Math.max(1, params.gridCellSize);
  let dx = Math.max(1, Math.ceil(params.worldX / cell));
  let dy = Math.max(1, Math.ceil(params.worldY / cell));
  let dz = Math.max(1, Math.ceil(params.worldZ / cell));
  // Shrink uniformly if we exceed the cap.
  while (dx * dy * dz > maxCells) {
    const s = Math.cbrt(maxCells / (dx * dy * dz));
    dx = Math.max(1, Math.floor(dx * s));
    dy = Math.max(1, Math.floor(dy * s));
    dz = Math.max(1, Math.floor(dz * s));
    if (dx * dy * dz <= maxCells) break;
    if (dz > 1) dz--; else if (dy > 1) dy--; else if (dx > 1) dx--; else break;
  }
  return { dx, dy, dz, cells: dx * dy * dz };
}

const FIELDS = 15; // vec4 fields in the Params uniform

export class ParamPacker {
  constructor() {
    this.buffer = new ArrayBuffer(FIELDS * 16);
    this.f32 = new Float32Array(this.buffer);
    this.u32 = new Uint32Array(this.buffer);
  }

  // fields: array in the exact order of the WGSL struct
  write(params, dims, dt, time, frame) {
    const f = this.f32;
    const u = this.u32;
    const p = params;

    // counts (u32)
    u[0] = p.boidCount | 0;
    u[1] = p.predatorCount | 0;
    u[2] = p.foodCount | 0;
    u[3] = p.obstacleCount | 0;

    f[4] = dt;
    f[5] = time;
    f[6] = frame;
    f[7] = p.timeScale;

    f[8] = p.maxSpeed;
    f[9] = p.maxForce;
    f[10] = p.perception;
    f[11] = p.boidRadius * 3; // separationScale (kept near perception)

    f[12] = p.alignWeight;
    f[13] = p.cohesionWeight;
    f[14] = p.separationWeight;
    f[15] = p.boundariesWeight;

    f[16] = p.avoidWeight;
    f[17] = p.seekFoodWeight;
    f[18] = p.fleePredatorWeight;
    f[19] = p.wanderWeight;

    f[20] = -p.worldX / 2;
    f[21] = -p.worldY / 2;
    f[22] = -p.worldZ / 2;
    f[23] = 0;

    f[24] = p.worldX;
    f[25] = p.worldY;
    f[26] = p.worldZ;
    f[27] = 0;

    f[28] = p.healthDecayPerSec;
    f[29] = p.foodHeal;
    f[30] = p.predatorDamage;
    f[31] = p.hitsToKill;

    f[32] = p.eatRadius;
    f[33] = p.foodRespawnDelay;
    f[34] = 100;
    f[35] = p.boidRespawnDelay;

    f[36] = p.predatorSpeed;
    f[37] = p.predatorForce;
    f[38] = p.predatorVision;
    f[39] = p.attackCooldown;

    f[40] = p.gridCellSize;
    f[41] = dims.dx;
    f[42] = dims.dy;
    f[43] = dims.dz;

    f[44] = p.wanderDistance;
    f[45] = p.wanderRadius;
    f[46] = p.wanderDisplace;
    f[47] = 1;

    // aux.x / aux.y are reserved (model appearance is render-only metadata).
    f[48] = 0;
    f[49] = 0;
    f[50] = p.boidRadius;
    f[51] = p.predatorRadius;

    f[52] = p.foodRadius;
    f[53] = p.obstacleRadius;
    f[54] = p.attackRange;
    f[55] = p.avoidWidth;

    f[56] = p.headingScale;
    f[57] = p.bboxScale;
    f[58] = p.visionScale;
    f[59] = p.hitboxScale;

    return this.buffer;
  }
}
