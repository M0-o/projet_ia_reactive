// src/legacy/boid3d.js
// Boid3D extends the untouched 2D `Boid` from boids.js and overrides every
// steering behaviour with a 3D version. This is the CPU reference/oracle for
// the WGSL compute port (shaders/sim.wgsl): the two implement the same
// behaviours, and the test harness checks the JS side for sanity.
//
// Nothing in boids.js is modified. Requires p5shim.js + boids.js first.

(function () {
  if (typeof Boid === "undefined") {
    throw new Error("Boid3D requires boids.js to be loaded first");
  }

  function basisFrom(fwd) {
    const f = fwd.copy().setMag(1);
    const up = Math.abs(f.y) > 0.9 ? createVector(1, 0, 0) : createVector(0, 1, 0);
    const right = up.cross(f).setMag(1);
    const top = f.cross(right);
    return { right, up: top, fwd: f };
  }

  class Boid3D extends Boid {
    constructor(x = 0, y = 0, z = 0, radius = 2.6) {
      super(x, y); // original constructor (fills pos/vel/acc + 2D defaults)
      this.pos.z = z;
      this.r = radius;
      this.radius = radius;
      this.vel = p5.Vector.random3D().setMag(random(2, 4));

      // health / damage (new in 3D version)
      this.health = 100;
      this.hits = 0;
      this.alive = true;
      this.age = 0;

      // 3D motion + behaviour parameters (mirror src/sim/params.js defaults)
      this.maxSpeed = 55;
      this.maxForce = 24;
      this.perceptionRadius = 26;
      this.alignWeight = 1.0;
      this.cohesionWeight = 0.9;
      this.separationWeight = 1.7;
      this.boundariesWeight = 9;
      this.avoidWeight = 7;
      this.seekFoodWeight = 1.3;
      this.fleePredatorWeight = 3.2;
      this.wanderWeight = 0.35;
      this.boundariesDistance = 30;
      this.boundsMin = createVector(-110, -75, -110);
      this.boundsSize = createVector(220, 150, 220);
      this.avoidWidth = 22;
      this.healthDecayPerSec = 1.6;
      this.eatRadius = 7;
      this.foodHeal = 5;

      // wander (spherical)
      this.distanceCercle = 40;
      this.wanderRadius = 18;
      this.displaceRange = 0.5;
      this.wanderTheta = 0;
    }

    // ---- flocking (3D overrides) -----------------------------------------
    align(boids) {
      const steering = createVector();
      let total = 0;
      for (const other of boids) {
        if (other === this) continue;
        if (this.pos.dist(other.pos) < this.perceptionRadius) {
          steering.add(other.vel);
          total++;
        }
      }
      if (total > 0) steering.div(total).setMag(this.maxSpeed).sub(this.vel).limit(this.maxForce);
      return steering;
    }

    cohesion(boids) {
      const perception = 2 * this.perceptionRadius;
      const steering = createVector();
      let total = 0;
      for (const other of boids) {
        if (other === this) continue;
        if (this.pos.dist(other.pos) < perception) {
          steering.add(other.pos);
          total++;
        }
      }
      if (total > 0) steering.div(total).sub(this.pos).setMag(this.maxSpeed).sub(this.vel).limit(this.maxForce);
      return steering;
    }

    separation(boids) {
      const steering = createVector();
      let total = 0;
      for (const other of boids) {
        if (other === this) continue;
        const d = this.pos.dist(other.pos);
        if (d < this.perceptionRadius && d > 1e-5) {
          const diff = p5.Vector.sub(this.pos, other.pos);
          diff.div(d * d);
          steering.add(diff);
          total++;
        }
      }
      if (total > 0) steering.div(total).setMag(this.maxSpeed).sub(this.vel).limit(this.maxForce);
      return steering;
    }

    // seek/flee are shared with the 2D version but now operate in 3D because
    // positions and targets carry a z component.
    seek(goal) {
      if (goal && goal.pos) goal = goal.pos;
      const desired = p5.Vector.sub(goal, this.pos).setMag(this.maxSpeed);
      return p5.Vector.sub(desired, this.vel).limit(this.maxForce);
    }

    flee(goal) {
      return this.seek(goal).mult(-1);
    }

    fleeWithTargetRadius(target) {
      const d = this.pos.dist(target.pos);
      const fleeRadius = target.r + target.radius + this.r;
      if (d >= fleeRadius) return createVector();
      return this.seek(target.pos).mult(-1).setMag(this.maxForce);
    }

    wander() {
      const fwd = this.vel.copy().setMag(1);
      const { right, up: top } = basisFrom(fwd);
      const centre = this.pos.copy().add(fwd.mult(this.distanceCercle));
      const pt = centre
        .copy()
        .add(right.mult(this.wanderRadius * cos(this.wanderTheta)))
        .add(top.mult(this.wanderRadius * sin(this.wanderTheta)));
      this.wanderTheta += random(-this.displaceRange, this.displaceRange);
      return p5.Vector.sub(pt, this.pos).setMag(this.maxSpeed).sub(this.vel).limit(this.maxForce);
    }

    boundaries(bx, by, bz, bw, bh, bd, d) {
      const minB = (bx === undefined) ? this.boundsMin : createVector(bx, by, bz);
      const size = (bx === undefined) ? this.boundsSize : createVector(bw, bh, bd);
      const dist = (d === undefined) ? this.boundariesDistance : d;
      const maxB = p5.Vector.add(minB, size);
      let desired = null;
      if (this.pos.x < minB.x + dist) desired = createVector(this.maxSpeed, this.vel.y, this.vel.z);
      else if (this.pos.x > maxB.x - dist) desired = createVector(-this.maxSpeed, this.vel.y, this.vel.z);
      if (this.pos.y < minB.y + dist) desired = createVector(this.vel.x, this.maxSpeed, this.vel.z);
      else if (this.pos.y > maxB.y - dist) desired = createVector(this.vel.x, -this.maxSpeed, this.vel.z);
      if (this.pos.z < minB.z + dist) desired = createVector(this.vel.x, this.vel.y, this.maxSpeed);
      else if (this.pos.z > maxB.z - dist) desired = createVector(this.vel.x, this.vel.y, -this.maxSpeed);
      if (desired === null) return createVector();
      return p5.Vector.sub(desired.setMag(this.maxSpeed), this.vel).limit(this.maxForce);
    }

    // 3D port of Boid.avoid against spherical obstacles. Look-ahead is a
    // fraction of maxSpeed (the original's `vel * 30` only made sense at 2D
    // speeds), and the push direction is away from the obstacle centre.
    avoid(obstacles) {
      if (!obstacles || obstacles.length === 0) return createVector();
      const fwd = this.vel.copy().setMag(1);
      const lookAhead = Math.max(this.maxSpeed * 0.6, 1);
      const p0 = this.pos;
      const p1 = p5.Vector.add(this.pos, fwd.copy().mult(lookAhead));
      const p2 = p5.Vector.add(this.pos, fwd.copy().mult(lookAhead * 0.5));
      let bestSurf = Infinity;
      let bestCenter = p0;
      for (const o of obstacles) {
        const oc = o.pos;
        let dm = p0.dist(oc);
        const d1 = p1.dist(oc); if (d1 < dm) dm = d1;
        const d2 = p2.dist(oc); if (d2 < dm) dm = d2;
        const surf = dm - o.r;
        if (surf < bestSurf) { bestSurf = surf; bestCenter = oc; }
      }
      if (bestSurf < this.avoidWidth) {
        return p5.Vector.sub(this.pos, bestCenter).setMag(this.maxForce * 0.5);
      }
      return createVector();
    }

    seekNearestFood(food) {
      let best = null; let bestD = Infinity;
      for (const f of food) {
        if (!f.active) continue;
        const d = this.pos.dist(f.pos);
        if (d < bestD) { bestD = d; best = f; }
      }
      if (!best) return createVector();
      const urgency = 1 + 2 * Math.min(Math.max(1 - this.health / 100, 0), 1);
      return this.seek(best.pos).mult(urgency);
    }

    fleeNearestPredator(predators, vision) {
      let best = null; let bestD = Infinity;
      for (const p of predators) {
        if (!p.alive) continue;
        const d = this.pos.dist(p.pos);
        if (d < bestD) { bestD = d; best = p; }
      }
      if (!best || bestD >= vision) return createVector();
      const urgency = 1 - bestD / vision;
      return this.flee(best.pos).mult(0.5 + urgency * 1.5);
    }

    applyForce(force) { this.acc.add(force); }

    // Overrides the 2D driver with a 3D one (same order as Boid.update).
    update() {
      this.pos.add(this.vel);
      this.vel.add(this.acc);
      this.vel.limit(this.maxSpeed);
      this.acc.mult(0);
    }

    edges() {
      const minB = this.boundsMin;
      const maxB = p5.Vector.add(this.boundsMin, this.boundsSize);
      this.pos.x = Math.min(Math.max(this.pos.x, minB.x), maxB.x);
      this.pos.y = Math.min(Math.max(this.pos.y, minB.y), maxB.y);
      this.pos.z = Math.min(Math.max(this.pos.z, minB.z), maxB.z);
    }

    // ---- aggregate step ---------------------------------------------------
    flock(boids, obstacles) {
      const align = this.align(boids).mult(this.alignWeight);
      const cohesion = this.cohesion(boids).mult(this.cohesionWeight);
      const separation = this.separation(boids).mult(this.separationWeight);
      const bounds = this.boundaries().mult(this.boundariesWeight);
      const avoid = this.avoid(obstacles).mult(this.avoidWeight);
      this.applyForce(align);
      this.applyForce(cohesion);
      this.applyForce(separation);
      this.applyForce(bounds);
      this.applyForce(avoid);
    }

    // Full reference tick: flock + seek food + flee predators + health.
    tick(dt, world) {
      if (!this.alive) return;
      const food = (world && world.food) || [];
      const predators = (world && world.predators) || [];
      const vision = (world && world.predatorVision) || 72;

      this.flock(world.boids || [], world.obstacles || []);
      this.applyForce(this.seekNearestFood(food).mult(this.seekFoodWeight));
      this.applyForce(this.fleeNearestPredator(predators, vision).mult(this.fleePredatorWeight));
      this.applyForce(this.wander().mult(this.wanderWeight));
      this.update();
      this.edges();

      for (const f of food) {
        if (f.active && this.pos.dist(f.pos) < this.eatRadius + f.r) {
          // Matches the compute shader: reaching a cluster lifts the *whole*
          // population's health by foodHeal percent (capped at 100), not just
          // the boid that ate.
          const flock = (world && world.boids) || [this];
          for (const b of flock) {
            if (b.alive !== false) b.health = Math.min(b.health + this.foodHeal, 100);
          }
          f.active = false;
          break;
        }
      }
      this.health -= this.healthDecayPerSec * dt;
      this.age += dt;
      if (this.health <= 0 || this.hits >= (world && world.hitsToKill ? world.hitsToKill : 4)) {
        this.alive = false;
      }
    }

    // Pack into the 12-float GPU Agent layout (see shaders/common.wgsl).
    toArray(out = new Float32Array(12)) {
      out[0] = this.pos.x; out[1] = this.pos.y; out[2] = this.pos.z; out[3] = this.health;
      out[4] = this.vel.x; out[5] = this.vel.y; out[6] = this.vel.z; out[7] = this.hits;
      out[8] = this.alive ? 1 : 0; out[9] = this.age; out[10] = this.wanderTheta; out[11] = this.r;
      return out;
    }
  }

  globalThis.AL.Boid = Boid; // expose the untouched 2D base for inspection
  globalThis.AL.Boid3D = Boid3D;
})();
