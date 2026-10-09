struct TAAU {
  resolution : vec2<f32>,
  blend      : f32,
  reset      : f32,
};
@group(0) @binding(0) var currentTex : texture_2d<f32>;
@group(0) @binding(1) var historyTex : texture_2d<f32>;
@group(0) @binding(2) var velTex     : texture_2d<f32>;
@group(0) @binding(3) var samp       : sampler;
@group(0) @binding(4) var<uniform> u : TAAU;

struct VSOut { @builtin(position) pos : vec4<f32> };
@vertex
fn vs(@builtin(vertex_index) vi : u32) -> VSOut {
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0)
  );
  var o : VSOut;
  o.pos = vec4<f32>(p[vi], 0.0, 1.0);
  return o;
}

@fragment
fn fs(frag : VSOut) -> @location(0) vec4<f32> {
  let uv   = frag.pos.xy / u.resolution;
  let curr = textureSampleLevel(currentTex, samp, uv, 0.0).rgb;
  if (u.reset > 0.5) {
    return vec4<f32>(curr, 1.0);
  }
  let vel    = textureSampleLevel(velTex, samp, uv, 0.0).rg;
  let histUV = uv - vel;
  var result = curr;
  if (histUV.x >= 0.0 && histUV.x <= 1.0 && histUV.y >= 0.0 && histUV.y <= 1.0) {
    let hist = textureSampleLevel(historyTex, samp, histUV, 0.0).rgb;
    let texel = 1.0 / u.resolution;
    var mn = curr;
    var mx = curr;
    for (var y = -1; y <= 1; y = y + 1) {
      for (var x = -1; x <= 1; x = x + 1) {
        let c = textureSampleLevel(currentTex, samp,
          uv + vec2<f32>(f32(x), f32(y)) * texel, 0.0).rgb;
        mn = min(mn, c);
        mx = max(mx, c);
      }
    }
    let clamped = clamp(hist, mn, mx);
    result = mix(clamped, curr, u.blend);
  }
  return vec4<f32>(result, 1.0);
}