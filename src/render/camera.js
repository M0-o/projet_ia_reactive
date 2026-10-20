// camera.js
// Orbit camera: yaw/pitch around a target point, with mouse + wheel + touch
// controls. Produces the Camera uniform expected by render_bindings.wgsl:
//
//   struct Camera {
//     viewProj, view, invViewProj : mat4x4<f32>,
//     camPos, viewport            : vec4<f32>,
//   };
//
// so 3 mat4x4 + 2 vec4 = 224 bytes. The buffer is padded to 256.

import { lookAt, perspective, multiply, invert } from "../math/mat4.js";

export const CAMERA_BYTES = 256;

export class OrbitCamera {
  constructor(canvas, worldSize) {
    this.canvas = canvas;
    this.target = [0, 0, 0];
    this.yaw = 0.6;
    this.pitch = 0.35;
    this.distance = Math.max(worldSize[0], worldSize[2]) * 1.35 || 300;
    this.fovY = (50 * Math.PI) / 180;
    this.near = 0.5;
    this.far = 4000;
    this.minDistance = 12;
    this.maxDistance = 3000;

    // scratch
    this._eye = [0, 0, 0];
    this._data = new Float32Array(CAMERA_BYTES / 4);
    this._view = null;
    this._proj = null;
    this._vp = null;
    this._aspect = 1;

    this._attach();
  }

  _attach() {
    const el = this.canvas;
    let dragging = false;
    let lastX = 0, lastY = 0;
    let panning = false;

    const onDown = (e) => {
      dragging = true;
      panning = e.button === 1 || e.shiftKey;
      lastX = e.clientX;
      lastY = e.clientY;
      el.setPointerCapture?.(e.pointerId);
    };
    const onMove = (e) => {
      if (!dragging) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      if (panning) {
        this._pan(dx, dy);
      } else {
        this.yaw -= dx * 0.005;
        this.pitch = clamp(this.pitch + dy * 0.005, -1.5, 1.5);
      }
    };
    const onUp = (e) => {
      dragging = false;
      el.releasePointerCapture?.(e.pointerId);
    };
    el.addEventListener("pointerdown", onDown);
    el.addEventListener("pointermove", onMove);
    el.addEventListener("pointerup", onUp);
    el.addEventListener("pointercancel", onUp);
    el.addEventListener("wheel", (e) => {
      e.preventDefault();
      const f = Math.exp(clamp(e.deltaY, -120, 120) * 0.0015);
      this.distance = clamp(this.distance * f, this.minDistance, this.maxDistance);
    }, { passive: false });
    // Middle-drag pan also works via right button context suppression.
    el.addEventListener("contextmenu", (e) => e.preventDefault());
  }

  _pan(dx, dy) {
    // Move the target in the camera's right/up plane.
    const right = [Math.cos(this.yaw), 0, -Math.sin(this.yaw)];
    const up = [0, 1, 0];
    const scale = this.distance * 0.0016;
    for (let i = 0; i < 3; i++) {
      this.target[i] += (-right[i] * dx + up[i] * dy) * scale;
    }
  }

  // Frame the whole world box nicely.
  frame(worldSize) {
    const r = 0.5 * Math.max(worldSize[0], worldSize[1], worldSize[2]);
    this.distance = clamp(r * 2.4, this.minDistance, this.maxDistance);
  }

  eye() {
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    this._eye[0] = this.target[0] + this.distance * cp * sy;
    this._eye[1] = this.target[1] + this.distance * sp;
    this._eye[2] = this.target[2] + this.distance * cp * cy;
    return this._eye;
  }

  // Recompute matrices for the given viewport size (device pixels).
  update(widthPx, heightPx) {
    this._aspect = widthPx / Math.max(1, heightPx);
    const eye = this.eye();
    this._view = lookAt(eye, this.target, [0, 1, 0]);
    this._proj = perspective(this.fovY, this._aspect, this.near, this.far);
    this._vp = multiply(this._proj, this._view);
    const inv = invert(this._vp) || new Float32Array(16);

    const d = this._data;
    d.set(this._vp, 0);
    d.set(this._view, 16);
    d.set(inv, 32);
    d[48] = eye[0]; d[49] = eye[1]; d[50] = eye[2]; d[51] = 1;
    d[52] = widthPx; d[53] = heightPx; d[54] = this.near; d[55] = this.far;
    return d;
  }

  get data() { return this._data; }
  get viewProj() { return this._vp; }
}

function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
