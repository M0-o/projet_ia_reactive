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
//   fsBlur      bloom  -> bloom (separable two-scale blur; run horizontally then
//                              vertically, ping-ponging between two textures)
//   fsComposite scene + bloom -> swapchain

struct Post {
  // x = threshold (luminance above which a pixel blooms)
  // y = strength   (how much bloom is added back)
  // z = radius     (width of the bloom's soft veil)
  // w = texel step (1/width, or 1/height, depending on the pass/direction)
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
//
// The four extra taps form a small cross centred on the source. This is not
// just antialiasing: the bloom target is half-resolution, so a bright pixel
// typically lands in a single texel. A blur whose first off-centre tap is a
// whole texel away would sample *past* such a texel and get nothing back, and
// the glow would degenerate into a dimmed copy of the source (a hard little
// streak). Dilating the source by one texel in each direction here gives the
// blur something to grab, which is what turns it into a soft haze.
@fragment
fn fsBright(i: VOut) -> @location(0) vec4<f32> {
  let s = post.params.w;
  var acc = textureSample(srcTex, linSmp, i.uv).rgb;
  acc = acc + textureSample(srcTex, linSmp, i.uv + vec2<f32>(s, 0.0)).rgb;
  acc = acc + textureSample(srcTex, linSmp, i.uv - vec2<f32>(s, 0.0)).rgb;
  acc = acc + textureSample(srcTex, linSmp, i.uv + vec2<f32>(0.0, s)).rgb;
  acc = acc + textureSample(srcTex, linSmp, i.uv - vec2<f32>(0.0, s)).rgb;
  let c = acc * 0.2;
  let l = luminance(c);
  // Soft knee: smoothly ramp in over a band just above the threshold.
  let knee = max(post.params.x * 0.5, 1e-4);
  let w = clamp((l - post.params.x) / knee, 0.0, 1.0);
  return vec4<f32>(c * w, 1.0);
}

// Separable blur, run twice (horizontally then vertically) over the same
// samples. `post.params.w` carries the per-axis texel step and `post.params.z`
// the radius.
//
// It is deliberately a **two-scale** kernel:
//
//   * a *core* — a tight Gaussian a texel or so wide, which keeps the glow
//     reading as a glow and gives the source a bright centre;
//   * a *veil* — a much wider Gaussian whose width follows the radius slider.
//
// A single-scale blur can only be tight (a hot little blob, which is what
// looks like a "streak") or broad (a washed-out smear). Summing the two gives
// a bright core wrapped in a wide, soft, low-contrast haze — the "misty" look.
//
// The taps are *contiguous*: tap k sits at (k - 0.5) texel steps, so there is
// never a gap between them and the kernel samples every position between the
// centre and the far edge. A kernel with gaps (the classic `k^1.35` spacing)
// skips over the one-or-two-texel-wide source the bright pass produces and
// leaves the glow as a dimmed copy of the source.
//
// The tap count is fixed, so the nearest tap always stays half a texel from
// the centre and always picks the source up. The radius slider widens the
// *veil* instead — that is the width the eye reads as "haze" — and the core
// stays tight so the source keeps a bright centre.
const BLUR_TAPS: u32 = 32u;
const CORE_SIGMA: f32 = 1.2;               // texels — the tight glow at the source
const VEIL_SIGMA_BASE: f32 = 3.0;          // texels
const VEIL_SIGMA_PER_RADIUS: f32 = 6.0;    // texels per unit of the slider
const CORE_WEIGHT: f32 = 0.40;
const VEIL_WEIGHT: f32 = 0.60;

@fragment
fn fsBlur(i: VOut) -> @location(0) vec4<f32> {
  let step = post.params.w;
  let coreSigma = CORE_SIGMA;
  let veilSigma = VEIL_SIGMA_BASE + max(post.params.z, 0.25) * VEIL_SIGMA_PER_RADIUS;
  // Centre tap.
  var accCore = textureSample(srcTex, linSmp, i.uv).rgb;
  var accVeil = accCore;
  var wCore = 1.0;
  var wVeil = 1.0;
  for (var k = 1u; k <= BLUR_TAPS; k = k + 1u) {
    let d = f32(k) - 0.5;
    let off = step * d;
    let s = textureSample(srcTex, linSmp, i.uv + off).rgb
          + textureSample(srcTex, linSmp, i.uv - off).rgb;
    let wc = exp(-(d * d) / (2.0 * coreSigma * coreSigma));
    let wv = exp(-(d * d) / (2.0 * veilSigma * veilSigma));
    accCore = accCore + s * wc;
    accVeil = accVeil + s * wv;
    wCore = wCore + 2.0 * wc;
    wVeil = wVeil + 2.0 * wv;
  }
  let core = accCore / wCore;
  let veil = accVeil / wVeil;
  return vec4<f32>(core * CORE_WEIGHT + veil * VEIL_WEIGHT, 1.0);
}

// Composite: the HDR scene plus the blurred bloom, written to the swapchain.
// The scene is NOT tonemapped here — that would change how every pixel looks
// versus the pre-bloom build. We simply add the glow back and clamp, so bloom
// is purely additive on top of the existing night look.
@fragment
fn fsComposite(i: VOut) -> @location(0) vec4<f32> {
  let scene = textureSample(srcTex, linSmp, i.uv).rgb;
  let bloom = textureSample(bloomTex, linSmp, i.uv).rgb;
  // Desaturate the glow toward its own luminance so it reads as suspended mist
  // rather than a coloured streak. Mixing toward *white* instead would add a
  // constant 0.14 to every pixel of the frame — a full-screen wash, not bloom.
  let haze = mix(bloom, vec3<f32>(luminance(bloom)), 0.35);
  let c = scene + haze * post.params.y;
  return vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
