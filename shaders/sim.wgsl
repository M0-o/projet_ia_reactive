// sim.wgsl
// All compute entry points for the simulation. Composed as:
//   common.wgsl + compute_bindings.wgsl + sim.wgsl
// One shader module -> many compute pipelines.

const SCAN_WG: u32 = 256u;

// Workgroup-scope scratch for the prefix scan. Declared at module scope as
// required by WGSL; each dispatch uses it independently.
var<workgroup> sScanLocal: array<u32, 256>;
var<workgroup> sScanPartial: array<u32, 256>;

fn numCells() -> u32 {
  return u32(P.grid.y) * u32(P.grid.z) * u32(P.grid.w);
}

// Exclusive/inclusive bounds of a cell's slice inside grid[O_ITEMS].
fn cellExclusiveStart(c: u32) -> u32 {
  return select(0u, atomicLoad(&grid[O_STARTS + c - 1u]), c > 0u);
}

fn cellInclusiveEnd(c: u32) -> u32 {
  return atomicLoad(&grid[O_STARTS + c]);
}

// ---------------------------------------------------------------------------
// Spatial hash: counting sort over a uniform grid.
// grid[O_COUNTS] -> grid[O_STARTS] (prefix sum) -> grid[O_ITEMS] (sorted).
// ---------------------------------------------------------------------------

@compute @workgroup_size(64)
fn gridClear(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= numCells()) { return; }
  atomicStore(&grid[O_COUNTS + i], 0u);
  // Slot 0 also resets the previous frame's flock-wide feed-event counter,
  // which `flockHealUpdate` drains at the end of the frame.
  if (i == 0u) { atomicStore(&grid[O_SPARE], 0u); }
}

@compute @workgroup_size(64)
fn gridCount(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.counts.x) { return; }
  let b = boids[i];
  if (b.info.x < 0.5) { return; }
  atomicAdd(&grid[O_COUNTS + cellIndex(b.posH.xyz)], 1u);
}

// Hillis-Steele inclusive scan of each 256-element block into O_STARTS.
// Block totals go to O_PARTIAL.
@compute @workgroup_size(256)
fn gridScanLocal(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let n = numCells();
  let i = wid.x * SCAN_WG + lid.x;

  var v = 0u;
  if (i < n) { v = atomicLoad(&grid[O_COUNTS + i]); }
  sScanLocal[lid.x] = v;
  workgroupBarrier();

  var offset = 1u;
  loop {
    if (offset >= SCAN_WG) { break; }
    var t = 0u;
    if (lid.x >= offset) { t = sScanLocal[lid.x - offset]; }
    workgroupBarrier();
    sScanLocal[lid.x] = sScanLocal[lid.x] + t;
    workgroupBarrier();
    offset = offset * 2u;
  }

  if (i < n) { atomicStore(&grid[O_STARTS + i], sScanLocal[lid.x]); }
  if (lid.x == SCAN_WG - 1u) { atomicStore(&grid[O_PARTIAL + wid.x], sScanLocal[SCAN_WG - 1u]); }
}

// Exclusive scan of block totals (single workgroup; <= 256 blocks).
@compute @workgroup_size(256)
fn gridScanPartial(@builtin(local_invocation_id) lid: vec3<u32>) {
  let n = numCells();
  let nb = (n + SCAN_WG - 1u) / SCAN_WG;

  var v = 0u;
  if (lid.x < nb) { v = atomicLoad(&grid[O_PARTIAL + lid.x]); }
  sScanPartial[lid.x] = v;
  workgroupBarrier();

  var offset = 1u;
  loop {
    if (offset >= SCAN_WG) { break; }
    var t = 0u;
    if (lid.x >= offset) { t = sScanPartial[lid.x - offset]; }
    workgroupBarrier();
    sScanPartial[lid.x] = sScanPartial[lid.x] + t;
    workgroupBarrier();
    offset = offset * 2u;
  }

  if (lid.x < nb) { atomicStore(&grid[O_PARTIAL + lid.x], sScanPartial[lid.x] - v); }
}

@compute @workgroup_size(256)
fn gridScanAdd(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let i = wid.x * SCAN_WG + lid.x;
  if (i >= numCells()) { return; }
  let base = atomicLoad(&grid[O_STARTS + i]);
  let add = atomicLoad(&grid[O_PARTIAL + i / SCAN_WG]);
  atomicStore(&grid[O_STARTS + i], base + add);
}

@compute @workgroup_size(64)
fn gridCursorInit(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= numCells()) { return; }
  atomicStore(&grid[O_CURSOR + i], cellExclusiveStart(i));
}

@compute @workgroup_size(64)
fn gridScatter(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.counts.x) { return; }
  let b = boids[i];
  if (b.info.x < 0.5) { return; }
  let c = cellIndex(b.posH.xyz);
  let slot = atomicAdd(&grid[O_CURSOR + c], 1u);
  atomicStore(&grid[O_ITEMS + slot], i);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn spawnPos(seed: u32) -> vec3<f32> {
  let r = rand3(seed);
  let margin = max(P.aux.z * 2.0, 1.0);
  let minB = P.boundsMin.xyz + vec3<f32>(margin);
  let maxB = P.boundsMin.xyz + P.boundsSize.xyz - vec3<f32>(margin);
  return minB + r * (maxB - minB);
}

// ---------------------------------------------------------------------------
// Boids: 3D port of boids.js (align/cohesion/separation/seek/flee/wander/
// boundaries/avoid) plus health, feeding and death.
// ---------------------------------------------------------------------------

@compute @workgroup_size(64)
fn boidsUpdate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.counts.x) { return; }
  let dt = P.timing.x;
  var b = boids[i];

  if (b.info.x < 0.5) {
    b.info.y = b.info.y + dt;
    if (b.info.y >= P.eating.w) {
      let seed = i * 2654435761u + u32(P.timing.z) + 17u;
      boids[i] = makeAgent(spawnPos(seed), b.info.w);
      return;
    }
    boids[i] = b;
    return;
  }

  let pos = b.posH.xyz;
  let vel = b.velD.xyz;
  let minB = P.boundsMin.xyz;
  let maxB = minB + P.boundsSize.xyz;
  let perception = P.motion.z;

  var alignSum = vec3<f32>(0.0);
  var alignN = 0.0;
  var cohSum = vec3<f32>(0.0);
  var cohN = 0.0;
  var sepSum = vec3<f32>(0.0);
  var sepN = 0.0;

  let cc = cellCoord(pos);
  let dim = gridDims();
  // Scan every cell that could hold a neighbour. The original 2D Boid uses
  // `perceptionRadius` for alignment/separation but **2x** that for cohesion
  // (see boids.js `cohesion()`), so the scan must cover the larger radius or
  // cohesion silently loses neighbours. A fixed 3x3x3 neighborhood also missed
  // neighbours whenever the cell was smaller than perception, which stopped
  // flocking from working. The reach is clamped so a tiny cell size cannot turn
  // the scan into an unbounded (device-losing) 3D sweep: 4 rings = 9x9x9 cells
  // per boid, which still covers the cohesion radius down to an eighth of the
  // cell size.
  let cohRadius = 2.0 * perception;
  let want = i32(ceil(cohRadius / max(P.grid.x, EPS)));
  let reach = clamp(want, 1, 4);
  for (var dz = -reach; dz <= reach; dz = dz + 1) {
    for (var dy = -reach; dy <= reach; dy = dy + 1) {
      for (var dx = -reach; dx <= reach; dx = dx + 1) {
        let ncc = cc + vec3<i32>(dx, dy, dz);
        if (ncc.x < 0 || ncc.y < 0 || ncc.z < 0) { continue; }
        if (ncc.x >= dim.x || ncc.y >= dim.y || ncc.z >= dim.z) { continue; }
        let nc = cellLinear(ncc);
        let s = cellExclusiveStart(nc);
        let e = cellInclusiveEnd(nc);
        for (var k = s; k < e; k = k + 1u) {
          let j = atomicLoad(&grid[O_ITEMS + k]);
          if (j == i) { continue; }
          let o = boids[j];
          if (o.info.x < 0.5) { continue; }
          let d = distance(pos, o.posH.xyz);
          if (d <= EPS) { continue; }
          // align + separation use `perception`...
          if (d < perception) {
            alignSum = alignSum + o.velD.xyz;
            alignN = alignN + 1.0;
            sepSum = sepSum + (pos - o.posH.xyz) / (d * d);
            sepN = sepN + 1.0;
          }
          // ...cohesion uses twice that, exactly as in the 2D original.
          if (d < cohRadius) {
            cohSum = cohSum + o.posH.xyz;
            cohN = cohN + 1.0;
          }
        }
      }
    }
  }

  var force = vec3<f32>(0.0);

  if (alignN > 0.0) {
    let v = limitLen(setMagSafe(alignSum / alignN, P.motion.x) - vel, P.motion.y);
    force = force + v * P.weights1.x;
  }
  if (cohN > 0.0) {
    let v = limitLen(setMagSafe(cohSum / cohN - pos, P.motion.x) - vel, P.motion.y);
    force = force + v * P.weights1.y;
  }
  if (sepN > 0.0) {
    let v = limitLen(setMagSafe(sepSum / sepN, P.motion.x) - vel, P.motion.y);
    force = force + v * P.weights1.z;
  }

  // boundaries (3D port of Boid.boundaries, returning the intended force)
  let bd = P.motion.w;
  var desired = vec3<f32>(0.0);
  var nearEdge = false;
  if (pos.x < minB.x + bd) { desired = vec3<f32>(P.motion.x, vel.y, vel.z); nearEdge = true; }
  else if (pos.x > maxB.x - bd) { desired = vec3<f32>(-P.motion.x, vel.y, vel.z); nearEdge = true; }
  if (pos.y < minB.y + bd) { desired = vec3<f32>(vel.x, P.motion.x, vel.z); nearEdge = true; }
  else if (pos.y > maxB.y - bd) { desired = vec3<f32>(vel.x, -P.motion.x, vel.z); nearEdge = true; }
  if (pos.z < minB.z + bd) { desired = vec3<f32>(vel.x, vel.y, P.motion.x); nearEdge = true; }
  else if (pos.z > maxB.z - bd) { desired = vec3<f32>(vel.x, vel.y, -P.motion.x); nearEdge = true; }
  if (nearEdge) {
    let dv = limitLen(setMagSafe(desired, P.motion.x) - vel, P.motion.y);
    force = force + dv * P.weights1.w;
  }

  // avoid obstacles (3D port of Boid.avoid with spherical obstacles).
  // The original used `vel * 30` as look-ahead, which only made sense at 2D
  // speeds (~2-4 px/frame). Here the look-ahead is a fixed fraction of
  // maxSpeed so it stays meaningful at simulation speeds.
  var avoidF = vec3<f32>(0.0);
  if (P.counts.w > 0u && P.weights2.x > 0.0) {
    let fwd = safeNormalize(vel, vec3<f32>(0.0, 0.0, 1.0));
    let lookAhead = max(P.motion.x * 0.6, 1.0);
    let p1 = pos + fwd * lookAhead;
    let p2 = pos + fwd * lookAhead * 0.5;
    var bestSurf = 1.0e9;
    var bestCenter = pos;
    var found = false;
    for (var oi = 0u; oi < P.counts.w; oi = oi + 1u) {
      let obs = obstacles[oi];
      if (obs.extra.x < 0.5) { continue; }
      let oc = obs.posR.xyz;
      var dm = distance(pos, oc);
      let d1 = distance(p1, oc);
      if (d1 < dm) { dm = d1; }
      let d2 = distance(p2, oc);
      if (d2 < dm) { dm = d2; }
      let surf = dm - obs.posR.w;
      if (surf < bestSurf) { bestSurf = surf; bestCenter = oc; found = true; }
    }
    if (found && bestSurf < P.aux2.w) {
      // Steer radially away from the obstacle centre. Using `pos` (rather than
      // the closest look-ahead point) keeps the direction correct even when the
      // look-ahead point lands inside the obstacle.
      avoidF = setMagSafe(safeNormalize(pos - bestCenter, vec3<f32>(0.0, 1.0, 0.0)), P.motion.y * 0.5);
    }
  }
  force = force + avoidF * P.weights2.x;

  // seek nearest active food cluster
  var seekF = vec3<f32>(0.0);
  if (P.counts.z > 0u && P.weights2.y > 0.0) {
    var bestD = 1.0e9;
    var bestIdx = -1;
    for (var fi = 0u; fi < P.counts.z; fi = fi + 1u) {
      let fd = food[fi];
      if (fd.extra.x < 0.5) { continue; }
      let dd = distance(pos, fd.posR.xyz);
      if (dd < bestD) { bestD = dd; bestIdx = i32(fi); }
    }
    if (bestIdx >= 0) {
      let goal = food[u32(bestIdx)].posR.xyz;
      seekF = limitLen(setMagSafe(goal - pos, P.motion.x) - vel, P.motion.y);
      seekF = seekF * (1.0 + 2.0 * clamp(1.0 - b.posH.w / 100.0, 0.0, 1.0));
    }
  }
  force = force + seekF * P.weights2.y;

  // flee nearest predator within its vision range
  var fleeF = vec3<f32>(0.0);
  if (P.counts.y > 0u && P.weights2.z > 0.0) {
    var bestD = 1.0e9;
    var bestIdx = -1;
    for (var pi = 0u; pi < P.counts.y; pi = pi + 1u) {
      let pr = predators[pi];
      if (pr.info.x < 0.5) { continue; }
      let dd = distance(pos, pr.posH.xyz);
      if (dd < bestD) { bestD = dd; bestIdx = i32(pi); }
    }
    if (bestIdx >= 0) {
      let fleeR = P.predator.z;
      if (bestD < fleeR) {
        let urgency = 1.0 - bestD / max(fleeR, EPS);
        let away = setMagSafe(pos - predators[u32(bestIdx)].posH.xyz, P.motion.x) - vel;
        fleeF = limitLen(away, P.motion.y) * (0.5 + urgency * 1.5);
      }
    }
  }
  force = force + fleeF * P.weights2.z;

  // wander (3D, on a sphere perpendicular to the heading)
  var wanderF = vec3<f32>(0.0);
  var theta = b.info.z;
  if (P.weights2.w > 0.0) {
    let fwd = safeNormalize(vel, vec3<f32>(0.0, 0.0, 1.0));
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(fwd.y) > 0.9);
    let tangent = safeNormalize(cross(up, fwd), vec3<f32>(1.0, 0.0, 0.0));
    let bitangent = cross(fwd, tangent);
    let centre = pos + fwd * P.wanderP.x;
    let pt = centre + (tangent * cos(theta) + bitangent * sin(theta)) * P.wanderP.y;
    wanderF = limitLen(setMagSafe(pt - pos, P.motion.x) - vel, P.motion.y);
    theta = theta + (hash2(i, u32(P.timing.z)) * 2.0 - 1.0) * P.wanderP.z;
  }
  force = force + wanderF * P.weights2.w;

  // integrate (pos += vel, then vel += acc, matching Boid.update order)
  var npos = pos + vel * dt;
  let nvel = limitLen(vel + force * dt, P.motion.x);
  npos = min(max(npos, minB), maxB);

  // feeding + health decay + death by starvation
  var health = b.posH.w;
  let hits = b.velD.w;
  if (P.counts.z > 0u) {
    for (var fi = 0u; fi < P.counts.z; fi = fi + 1u) {
      var fd = food[fi];
      if (fd.extra.x < 0.5) { continue; }
      if (distance(npos, fd.posR.xyz) < P.eating.x + fd.posR.w) {
        // Record a flock-wide feed event. The individual boid's own health is
        // left alone here; the population average is topped up by the
        // `flockHealUpdate` kernel later this frame.
        atomicAdd(&grid[O_SPARE], 1u);
        fd.extra.x = 0.0;
        fd.extra.y = P.eating.y;
        food[fi] = fd;
        break;
      }
    }
  }
  health = health - P.health.x * dt;

  if (health <= 0.0 || hits >= P.health.w) {
    boids[i] = makeDeadAgent(b.info.w);
    return;
  }

  b.posH = vec4<f32>(npos, health);
  b.velD = vec4<f32>(nvel, hits);
  b.info = vec4<f32>(1.0, b.info.y + dt, theta, b.info.w);
  boids[i] = b;
}

// ---------------------------------------------------------------------------
// Predators: seek the nearest boid, attack on contact, wander otherwise.
// ---------------------------------------------------------------------------

@compute @workgroup_size(64)
fn predatorsUpdate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.counts.y) { return; }
  let dt = P.timing.x;
  var p = predators[i];

  if (p.info.x < 0.5) {
    p.info.y = p.info.y + dt;
    if (p.info.y >= P.eating.w * 2.0) {
      let seed = i * 40503u + u32(P.timing.z) + 777u;
      predators[i] = makeAgent(spawnPos(seed), p.info.w);
      return;
    }
    predators[i] = p;
    return;
  }

  let pos = p.posH.xyz;
  let vel = p.velD.xyz;
  let minB = P.boundsMin.xyz;
  let maxB = minB + P.boundsSize.xyz;

  var bestD = 1.0e9;
  var bestIdx = -1;
  for (var bi = 0u; bi < P.counts.x; bi = bi + 1u) {
    let b = boids[bi];
    if (b.info.x < 0.5) { continue; }
    let d = distance(pos, b.posH.xyz);
    if (d < bestD) { bestD = d; bestIdx = i32(bi); }
  }

  var force = vec3<f32>(0.0);
  if (bestIdx >= 0 && bestD < P.predator.z * 3.0) {
    let goal = boids[u32(bestIdx)].posH.xyz;
    force = limitLen(setMagSafe(goal - pos, P.predator.x) - vel, P.predator.y);

    let range = P.aux2.z + p.info.w + boids[u32(bestIdx)].info.w;
    if (bestD < range && p.info.z <= 0.0) {
      var victim = boids[u32(bestIdx)];
      victim.velD.w = victim.velD.w + 1.0;
      victim.posH.w = victim.posH.w - P.health.z;
      boids[u32(bestIdx)] = victim;
      p.info.z = P.predator.w;
    }
  } else {
    let fwd = safeNormalize(vel, vec3<f32>(0.0, 0.0, 1.0));
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(fwd.y) > 0.9);
    let tangent = safeNormalize(cross(up, fwd), vec3<f32>(1.0, 0.0, 0.0));
    let bitangent = cross(fwd, tangent);
    let ang = hash2(i, u32(P.timing.z)) * TAU;
    let centre = pos + fwd * P.wanderP.x;
    let pt = centre + (tangent * cos(ang) + bitangent * sin(ang)) * P.wanderP.y;
    force = limitLen(setMagSafe(pt - pos, P.predator.x) - vel, P.predator.y);
  }

  let bd = P.motion.w;
  var desired = vec3<f32>(0.0);
  var nearEdge = false;
  if (pos.x < minB.x + bd) { desired = vec3<f32>(P.predator.x, vel.y, vel.z); nearEdge = true; }
  else if (pos.x > maxB.x - bd) { desired = vec3<f32>(-P.predator.x, vel.y, vel.z); nearEdge = true; }
  if (pos.y < minB.y + bd) { desired = vec3<f32>(vel.x, P.predator.x, vel.z); nearEdge = true; }
  else if (pos.y > maxB.y - bd) { desired = vec3<f32>(vel.x, -P.predator.x, vel.z); nearEdge = true; }
  if (pos.z < minB.z + bd) { desired = vec3<f32>(vel.x, vel.y, P.predator.x); nearEdge = true; }
  else if (pos.z > maxB.z - bd) { desired = vec3<f32>(vel.x, vel.y, -P.predator.x); nearEdge = true; }
  if (nearEdge) {
    force = force + limitLen(setMagSafe(desired, P.predator.x) - vel, P.predator.y) * P.weights1.w;
  }

  var npos = pos + vel * dt;
  let nvel = limitLen(vel + force * dt, P.predator.x);
  npos = min(max(npos, minB), maxB);

  p.posH = vec4<f32>(npos, 100.0);
  p.velD = vec4<f32>(nvel, p.velD.w);
  p.info = vec4<f32>(1.0, p.info.y + dt, max(0.0, p.info.z - dt), p.info.w);
  predators[i] = p;
}

// ---------------------------------------------------------------------------
// Food: regrow eaten clusters after a delay.
// ---------------------------------------------------------------------------

@compute @workgroup_size(64)
fn foodUpdate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.counts.z) { return; }
  var f = food[i];
  if (f.extra.x < 0.5) {
    // Eaten: count the respawn timer down and write it back every frame (a
    // local-only decrement would be discarded and the cluster would never
    // regrow).
    f.extra.y = f.extra.y - P.timing.x;
    if (f.extra.y <= 0.0) {
      f.extra.x = 1.0;
      f.extra.y = 0.0;
    }
    food[i] = f;
  }
}

// ---------------------------------------------------------------------------
// Flock-wide feeding: every time any boid reaches a food cluster, the *average*
// health of the whole population rises by the "Food heal (%)" value. Feed
// events are counted by `boidsUpdate` into grid[O_SPARE]; here each alive boid
// gets that many percentage points added to its own health (clamped at 100),
// which lifts the population average by the same amount.
// Runs after boidsUpdate and before the next frame's gridClear.
// ---------------------------------------------------------------------------

@compute @workgroup_size(64)
fn flockHealUpdate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.counts.x) { return; }
  let events = f32(atomicLoad(&grid[O_SPARE]));
  if (events <= 0.0) { return; }
  var b = boids[i];
  if (b.info.x < 0.5) { return; } // only living boids benefit
  b.posH.w = min(b.posH.w + P.health.y * events, 100.0);
  boids[i] = b;
}

