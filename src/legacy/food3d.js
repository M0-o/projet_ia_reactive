// src/legacy/food3d.js
// Food cluster. Mirrors the `Point` struct's food layout in shaders/common.wgsl
// (posR.xyz + radius, extra.x = active, extra.y = respawn timer).

(function () {
  class Food3D {
    constructor(pos, r = 3.2, id = 0) {
      this.pos = pos && pos.copy ? pos : createVector(pos[0] || 0, pos[1] || 0, pos[2] || 0);
      this.r = r;
      this.id = id;
      this.active = true;
      this.timer = 0;
      this.energy = 100;
    }

    // Eaten by a boid: goes dormant, then regrows after `delay` seconds.
    consume(delay = 8) {
      this.active = false;
      this.timer = delay;
    }

    tick(dt) {
      if (!this.active) {
        this.timer -= dt;
        if (this.timer <= 0) { this.active = true; this.timer = 0; }
      }
    }

    toArray(out = new Float32Array(8)) {
      out[0] = this.pos.x; out[1] = this.pos.y; out[2] = this.pos.z; out[3] = this.r;
      out[4] = this.active ? 1 : 0; out[5] = this.timer; out[6] = this.id; out[7] = this.energy;
      return out;
    }
  }

  globalThis.AL.Food3D = Food3D;
})();
