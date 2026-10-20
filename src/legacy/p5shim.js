// src/legacy/p5shim.js
// Minimal p5.js global-mode shim. boids.js is an untouched 2D sketch written
// against p5 globals (createVector, dist, random, cos, width, ...). This
// provides just enough of that environment for the file to load and for the
// Boid3D subclasses to run their math as a CPU reference/oracle.
//
// This is NOT a full p5 implementation and does no rendering; the simulation
// itself runs in WebGPU compute shaders.

(function () {
  if (typeof globalThis.AL === "undefined") globalThis.AL = {};
  const TAU = Math.PI * 2;

  class Vec3 {
    constructor(x = 0, y = 0, z = 0) {
      this.x = x; this.y = y; this.z = z;
    }
    copy() { return new Vec3(this.x, this.y, this.z); }
    add(v) { this.x += v.x; this.y += v.y; this.z += v.z; return this; }
    sub(v) { this.x -= v.x; this.y -= v.y; this.z -= v.z; return this; }
    mult(n) { this.x *= n; this.y *= n; this.z *= n; return this; }
    div(n) { if (n !== 0) { this.x /= n; this.y /= n; this.z /= n; } return this; }
    mag() { return Math.hypot(this.x, this.y, this.z); }
    magSq() { return this.x * this.x + this.y * this.y + this.z * this.z; }
    setMag(m) { const l = this.mag(); if (l > 1e-6) this.mult(m / l); return this; }
    limit(m) { if (this.magSq() > m * m) this.setMag(m); return this; }
    dist(v) { return Math.hypot(this.x - v.x, this.y - v.y, this.z - v.z); }
    dot(v) { return this.x * v.x + this.y * v.y + this.z * v.z; }
    cross(v) {
      return new Vec3(this.y * v.z - this.z * v.y, this.z * v.x - this.x * v.z, this.x * v.y - this.y * v.x);
    }
    heading() { return Math.atan2(this.y, this.x); }
    normalize() { return this.setMag(1); }
    static add(a, b) { return new Vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
    static sub(a, b) { return new Vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
    static mult(a, n) { return new Vec3(a.x * n, a.y * n, a.z * n); }
    static random2D() {
      const a = Math.random() * TAU;
      return new Vec3(Math.cos(a), Math.sin(a), 0);
    }
    static random3D() {
      const u = Math.random() * 2 - 1;
      const t = Math.random() * TAU;
      const r = Math.sqrt(Math.max(0, 1 - u * u));
      return new Vec3(r * Math.cos(t), r * Math.sin(t), u);
    }
  }

  const G = globalThis;
  const noop = () => {};
  G.createVector = (x = 0, y = 0, z = 0) => new Vec3(x, y, z);
  G.p5 = Object.assign(G.p5 || {}, { Vector: Vec3 });
  G.random = (a, b) => (b === undefined ? Math.random() * (a === undefined ? 1 : a) : a + Math.random() * (b - a));
  G.dist = (...args) => {
    if (args.length >= 6) return Math.hypot(args[0] - args[3], args[1] - args[4], args[2] - args[5]);
    return Math.hypot(args[0] - args[2], args[1] - args[3]);
  };
  G.constrain = (v, a, b) => Math.min(Math.max(v, a), b);
  G.width = 1200;
  G.height = 800;
  G.cos = Math.cos; G.sin = Math.sin; G.atan2 = Math.atan2; G.abs = Math.abs;

  // No-op drawing API so debug branches in the original file can never throw.
  for (const fn of ["push", "pop", "stroke", "noFill", "fill", "circle", "line", "rect",
    "strokeWeight", "point", "image", "imageMode", "translate", "rotate", "triangle",
    "noStroke", "background", "ellipse", "strokeJoin", "text", "textSize", "beginShape",
    "vertex", "endShape", "strokeCap", "colorMode", "createCanvas"]) {
    if (typeof G[fn] === "undefined") G[fn] = noop;
  }

  AL.Vec3 = Vec3;
})();
