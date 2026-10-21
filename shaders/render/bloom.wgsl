// bloom.wgsl
// Post-processing for the night scene. The scene is drawn into an HDR
// (rgba16float) target so the phoenix's emissive can exceed 1.0; bloom then
// extracts the bright excess, blurs it, and adds it back when compositing to
// the swapchain.
//
// This module is deliberately standalone: it declares its own bind group (0)
// with the textures it samples plus one uniform, so it shares no storage
// buffers with the scene shaders.
//
// Passes:
//   fsBright    scene  -> bloom (threshold + downsample to half resolution)
//   fsBlur      bloom  -> bloom (separable Gaussian; run horizontally then
//                              vertically, ping-ponging between two textures)
//   fsComposite scene + bloom -> swapchain

struct Post {
  // x = threshold (luminance above which a pixel blooms)
  // y = strength   (how much bloom is added back)
  // z = radius     (blur sample spacing in texels)
  // w = texel step (1/width, or 1/height, depending on the blur direction)
  params : vec4<f32>,
};

@group(0) @binding(0) var<uniform> post : Post;
@group(0) @binding(1) var srcTex : texture_2d<f32>;
@group(0) @binding(2) var bloomTex : texture_2d<f32>;
@group(0) @binding(3) var linSmp : sampler;

struct VOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) uv : vec2<f32>,
};

// Oversized full-screen triangle. UV is 0..1 across the viewport.
@vertex
fn vsFullscreen(@builtin(vertex_index) vi: u32) -> VOut {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0),
  );
  let p = pos[vi];
  var o: VOut;
  o.clip = vec4<f32>(p, 0.0, 1.0);
  // Flip V: clip space is +Y up, texture space is +Y down.
  o.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  return o;
}

// Rec. 709 luminance — what a bloom threshold usually keys off.
fn luminance(c: vec3<f32>) -> f32 {
  return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Bright pass: keep only what is brighter than the threshold, with a soft knee
// so the brightest pixels dominate instead of everything clipping equally.
@fragment
fn fsBright(i: VOut) -> @location(0) vec4<f32> {
  let c = textureSample(srcTex, linSmp, i.uv).rgb;
  let l = luminance(c);
  // Soft knee: smoothly ramp in over a band just above the threshold.
  let knee = max(post.params.x * 0.5, 1e-4);
  let w = clamp((l - post.params.x) / knee, 0.0, 1.0);
  return vec4<f32>(c * w, 1.0);
}

// Separable Gaussian blur. `post.params.w` carries the per-axis texel step and
// `post.params.z` the radius, so the same entry point runs horizontally and
// vertically. Sample offsets grow with the radius so the slider controls the
// halo's width: at radius 1 the kernel spans ~2 texels each way, at radius 4 it
// spans ~10, which on the half-res bloom target is a wide, soft glow.
const BLUR_TAPS: u32 = 6u;
const BLUR_SIGMA: f32 = 2.2;

@fragment
fn fsBlur(i: VOut) -> @location(0) vec4<f32> {
  let step = post.params.w;
  let r = max(post.params.z, 0.0);
  // Centre tap.
  var acc = textureSample(srcTex, linSmp, i.uv).rgb;
  var wsum = 1.0;
  for (var k = 1u; k <= BLUR_TAPS; k = k + 1u) {
    let x = f32(k);
    let w = exp(-(x * x) / (2.0 * BLUR_SIGMA * BLUR_SIGMA));
    // Offsets scale with the radius so the slider widens the kernel; the taps
    // also spread out (x^1.35) to cover a broad area without banding.
    let off = step * (r * pow(x, 1.35));
    acc = acc + (textureSample(srcTex, linSmp, i.uv + off).rgb
               + textureSample(srcTex, linSmp, i.uv - off).rgb) * w;
    wsum = wsum + 2.0 * w;
  }
  return vec4<f32>(acc / wsum, 1.0);
}

// Composite: the HDR scene plus the blurred bloom, written to the swapchain.
// The scene is NOT tonemapped here — that would change how every pixel looks
// versus the pre-bloom build. We simply add the glow back and clamp, so bloom
// is purely additive on top of the existing night look.
@fragment
fn fsComposite(i: VOut) -> @location(0) vec4<f32> {
  let scene = textureSample(srcTex, linSmp, i.uv).rgb;
  let bloom = textureSample(bloomTex, linSmp, i.uv).rgb;
  let c = scene + bloom * post.params.y;
  return vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
