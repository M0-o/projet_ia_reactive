// src/legacy/vehicle3d.js
// The repository's vehicle.js is an empty placeholder, so the 3D base
// "vehicle" for non-boid entities is introduced here. It simply specialises
// Boid3D's steering (already 3D) and adds an explicit cooldown timer used by
// the predator. Predator3D extends this.

(function () {
  if (typeof AL.Boid3D === "undefined") {
    throw new Error("Vehicle3D requires boid3d.js");
  }

  class Vehicle3D extends AL.Boid3D {
    constructor(x = 0, y = 0, z = 0, radius = 3.6) {
      super(x, y, z, radius);
      this.cooldown = 0;
      this.vision = 72;
    }

    // Advances the cooldown timer; returns true when an attack is allowed.
    readyToAttack() {
      return this.cooldown <= 0;
    }

    tickCooldown(dt) {
      if (this.cooldown > 0) this.cooldown = Math.max(0, this.cooldown - dt);
    }
  }

  globalThis.AL.Vehicle3D = Vehicle3D;
})();
