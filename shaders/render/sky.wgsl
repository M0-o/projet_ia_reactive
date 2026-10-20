// sky.wgsl
// Full-screen night-sky pass. Draws a single triangle that covers the viewport
// (no vertex buffer) and shades each pixel from its camera ray: a vertical
// gradient plus a twinkling star field. Drawn first, with the depth test
// disabled, as the scene background.

struct SkyOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) ndc : vec2<f32>,
};

@vertex
fn vsSky(@builtin(vertex_index) vi: u32) -> SkyOut {
  // Oversized triangle covering clip space.
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0),
  );
  let p = pos[vi];
  var o: SkyOut;
  o.clip = vec4<f32>(p, 0.0, 1.0);
  o.ndc = p;
  return o;
}

@fragment
fn fsSky(i: SkyOut) -> @location(0) vec4<f32> {
  let dir = rayDir(i.ndc);
  var c = skyColor(dir) + starField(dir, P.timing.y);
  // A subtle, very dim galactic band for depth.
  let band = exp(-pow(abs(dir.y - 0.28 + dir.x * 0.15) * 3.2, 2.0));
  c = c + vec3<f32>(0.020, 0.022, 0.040) * band;
  return vec4<f32>(c, 1.0);
}
