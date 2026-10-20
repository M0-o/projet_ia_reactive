// device.js
// WebGPU bootstrap with a friendly, actionable failure path. If the browser
// has no WebGPU (old browser, disabled flag, no secure context, or a missing
// adapter) we surface a clear message instead of a blank canvas.

export const WEBGPU_HINT =
  "WebGPU is required for this simulation. Try a recent Chrome, Edge, or " +
  "another browser with WebGPU enabled. If you are on Chrome, make sure " +
  "chrome://flags/#enable-unsafe-webgpu is set to Enabled and that the page " +
  "is served over http://localhost or https:// (WebGPU needs a secure " +
  "context). Then reload.";

// Returns { device, adapter, canvas, context, format } or throws an Error
// whose .userMessage is safe to show in the UI.
export async function initWebGPU(canvas) {
  if (typeof navigator === "undefined" || !navigator.gpu) {
    throw noWebGPU("This browser does not expose navigator.gpu.");
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) {
    throw noWebGPU("No WebGPU adapter was available on this machine.");
  }
  const requiredFeatures = [];
  // Optional features we can use if present; not required for correctness.
  const supported = adapter.features;
  if (supported.has("timestamp-query")) requiredFeatures.push("timestamp-query");

  const device = await adapter.requestDevice({
    label: "boids-device",
    requiredFeatures,
    requiredLimits: {},
  });

  if (!device) throw noWebGPU("The WebGPU device could not be created.");

  // Surface validation/OOM errors instead of letting the frame silently fail.
  device.onuncapturederror = (ev) => {
    const msg = ev.error && ev.error.message ? ev.error.message : String(ev.error);
    console.error("[WebGPU]", msg);
    if (typeof window !== "undefined" && typeof window.__reportGpuError === "function") {
      window.__reportGpuError(msg);
    }
  };

  const context = canvas.getContext("webgpu");
  if (!context) throw noWebGPU("The canvas could not be configured for WebGPU.");
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "premultiplied" });

  return { device, adapter, canvas, context, format };
}

function noWebGPU(detail) {
  const err = new Error(detail + " " + WEBGPU_HINT);
  err.userMessage = detail + " " + WEBGPU_HINT;
  err.isWebGPUMissing = true;
  return err;
}

// Wire a device-lost handler that reports to the UI rather than dying silently.
export function onDeviceLost(device, onLost) {
  device.lost.then((info) => {
    onLost(info.message || "The WebGPU device was lost.");
  });
}
