// compute_bindings.wgsl
// Bindings shared by every compute pipeline.
// Binding indices must match src/sim/bindings.js (computeLayout).
//
// Only 5 storage buffers are used (WebGPU's default limit is 8 per stage).
// All spatial-hash scratch space lives inside the single `grid` buffer; its
// section offsets are the O_* constants in common.wgsl.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> boids: array<Agent>;
@group(0) @binding(2) var<storage, read_write> predators: array<Agent>;
@group(0) @binding(3) var<storage, read_write> food: array<Point>;
@group(0) @binding(4) var<storage, read>      obstacles: array<Point>;
@group(0) @binding(5) var<storage, read_write> grid: array<atomic<u32>>;
