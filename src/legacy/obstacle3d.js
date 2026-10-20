// src/legacy/obstacle3d.js
// Static spherical obstacle the boids must avoid. Mirrors the `Point`
// struct's obstacle layout in shaders/common.wgsl.

(function () {
  class Obstacle3D {
    constructor(pos, r = 12, id = 0) {
      this.pos = pos && pos.copy ? pos : createVector(pos[0] || 0, pos[1] || 0, pos[2] || 0);
      this.r = r;
      this.id = id;
      this.active = true;
    }

    toArray(out = new Float32Array(8)) {
      out[0] = this.pos.x; out[1] = this.pos.y; out[2] = this.pos.z; out[3] = this.r;
      out[4] = this.active ? 1 : 0; out[5] = 0; out[6] = this.id; out[7] = 1;
      return out;
    }
  }

  globalThis.AL.Obstacle3D = Obstacle3D;
})();
