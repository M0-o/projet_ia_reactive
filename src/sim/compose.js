// compose.js
// Shader files are concatenated into a single module per pipeline. This keeps
// WGSL free of an include mechanism while sharing structs/helpers.
//
//   compute module = common + compute_bindings + sim
//   render  module = common + render_bindings + render_common + <module>

export const COMMON = "shaders/common.wgsl";
export const COMPUTE_BINDINGS = "shaders/compute_bindings.wgsl";
export const RENDER_BINDINGS = "shaders/render_bindings.wgsl";
export const RENDER_COMMON = "shaders/render_common.wgsl";
export const SIM = "shaders/sim.wgsl";

export const RENDER_MODULES = {
  sky: "shaders/render/sky.wgsl",
  models: "shaders/render/models.wgsl",
  food: "shaders/render/food.wgsl",
  box: "shaders/render/box.wgsl",
  debug: "shaders/render/debug_lines.wgsl",
  hashcells: "shaders/render/hashcells.wgsl",
};

// Render modules that need the camera's inverse view-projection (sky) and the
// full set of storage bindings are all composed the same way here.
export function allShaderFiles() {
  return [COMMON, COMPUTE_BINDINGS, RENDER_BINDINGS, RENDER_COMMON, SIM, ...Object.values(RENDER_MODULES)];
}

export function composeCompute(parts) {
  return [parts[COMMON], parts[COMPUTE_BINDINGS], parts[SIM]].join("\n");
}

export function composeRender(parts, name) {
  return [parts[COMMON], parts[RENDER_BINDINGS], parts[RENDER_COMMON], parts[RENDER_MODULES[name]]].join("\n");
}

// `read(path) -> Promise<string>` (fetch in the browser, fs in Node).
export async function loadShaderParts(read) {
  const files = allShaderFiles();
  const texts = await Promise.all(files.map((f) => read(f)));
  const parts = {};
  files.forEach((f, i) => { parts[f] = texts[i]; });
  return parts;
}
