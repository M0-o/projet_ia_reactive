// src/legacy/predator3d.js
// Predator3D extends Vehicle3D (which extends Boid3D -> Boid). It hunts the
// nearest boid, attacks on contact with a cooldown, and otherwise wanders.
// CPU reference for the `predatorsUpdate` entry point in shaders/sim.wgsl.

(function () {
  if (typeof AL.Vehicle3D === "undefined") {
    throw new Error("Predator3D requires vehicle3d.js");
  }

  class Predator3D extends AL.Vehicle3D {
    constructor(x = 0, y = 0, z = 0, radius = 3.6) {
      super(x, y, z, radius);
      this.maxSpeed = 78;
      this.maxForce = 32;
      this.vision = 72;
      this.attackRange = 1.5;
      this.attackCooldown = 0.8;
      this.predatorDamage = 22;
      this.hitsToKill = 4;
    }

    nearestBoid(boids) {
      let best = null;
      let bestD = Infinity;
      for (const b of boids) {
        if (!b.alive || b === this) continue;
        const d = this.pos.dist(b.pos);
        if (d < bestD) { bestD = d; best = b; }
      }
      return { victim: best, distance: bestD };
    }

    tick(dt, world) {
      if (!this.alive) return;
      const boids = (world && world.boids) || [];
      this.tickCooldown(dt);

      const { victim, distance } = this.nearestBoid(boids);
      if (victim && distance < this.vision * 3) {
        this.applyForce(this.seek(victim.pos));
        const range = this.attackRange + this.r + victim.r;
        if (distance < range && this.readyToAttack()) {
          victim.hits += 1;
          victim.health -= this.predatorDamage;
          this.cooldown = this.attackCooldown;
          if (victim.health <= 0 || victim.hits >= this.hitsToKill) victim.alive = false;
        }
      } else {
        this.applyForce(this.wander());
      }

      this.applyForce(this.boundaries().mult(this.boundariesWeight));
      this.update();
      this.edges();
      this.health = 100; // predators are not subject to starvation
      this.age += dt;
    }
  }

  globalThis.AL.Predator3D = Predator3D;
})();
