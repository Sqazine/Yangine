struct BloomU {
  texel    : vec2<f32>,
  dstRes   : vec2<f32>,
  threshold: f32,
  knee     : f32,
};
@group(0) @binding(0) var srcTex : texture_2d<f32>;
@group(0) @binding(1) var samp   : sampler;
@group(0) @binding(2) var<uniform> u : BloomU;

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
fn fs_pre(frag : VSOut) -> @location(0) vec4<f32> {
  let uv = frag.pos.xy / u.dstRes;
  let c  = textureSampleLevel(srcTex, samp, uv, 0.0).rgb;
  let knee = max(u.threshold * u.knee, 1e-4);
  var q = max(c - vec3<f32>(u.threshold) + vec3<f32>(knee), vec3<f32>(0.0));
  q = min(q, vec3<f32>(2.0 * knee));
  q = q * q / (4.0 * knee);
  let result = q + max(c - vec3<f32>(u.threshold), vec3<f32>(0.0));
  return vec4<f32>(result, 1.0);
}

@fragment
fn fs_down(frag : VSOut) -> @location(0) vec4<f32> {
  let uv = frag.pos.xy / u.dstRes;
  let t  = u.texel;
  let a = textureSampleLevel(srcTex, samp, uv + t * vec2<f32>(-2.0, -2.0), 0.0).rgb;
  let b = textureSampleLevel(srcTex, samp, uv + t * vec2<f32>( 0.0, -2.0), 0.0).rgb;
  let c = textureSampleLevel(srcTex, samp, uv + t * vec2<f32>( 2.0, -2.0), 0.0).rgb;
  let d = textureSampleLevel(srcTex, samp, uv + t * vec2<f32>(-2.0,  0.0), 0.0).rgb;
  let e = textureSampleLevel(srcTex, samp, uv + t * vec2<f32>( 0.0,  0.0), 0.0).rgb;
  let f = textureSampleLevel(srcTex, samp, uv + t * vec2<f32>( 2.0,  0.0), 0.0).rgb;
  let g = textureSampleLevel(srcTex, samp, uv + t * vec2<f32>(-2.0,  2.0), 0.0).rgb;
  let h = textureSampleLevel(srcTex, samp, uv + t * vec2<f32>( 0.0,  2.0), 0.0).rgb;
  let i = textureSampleLevel(srcTex, samp, uv + t * vec2<f32>( 2.0,  2.0), 0.0).rgb;
  let j = textureSampleLevel(srcTex, samp, uv + t * vec2<f32>(-1.0, -1.0), 0.0).rgb;
  let k = textureSampleLevel(srcTex, samp, uv + t * vec2<f32>( 1.0, -1.0), 0.0).rgb;
  let l = textureSampleLevel(srcTex, samp, uv + t * vec2<f32>(-1.0,  1.0), 0.0).rgb;
  let m = textureSampleLevel(srcTex, samp, uv + t * vec2<f32>( 1.0,  1.0), 0.0).rgb;
  var result = (a + c + g + i) * 0.03125;
  result += (b + d + f + h) * 0.0625;
  result += (j + k + l + m) * 0.125;
  result += e * 0.25;
  result *= 1.0 / 1.125;
  return vec4<f32>(result, 1.0);
}

@fragment
fn fs_up(frag : VSOut) -> @location(0) vec4<f32> {
  let uv = frag.pos.xy / u.dstRes;
  let t  = u.texel;
  var s = textureSampleLevel(srcTex, samp, uv, 0.0).rgb * 4.0;
  s += textureSampleLevel(srcTex, samp, uv + t * vec2<f32>(-1.0,  0.0), 0.0).rgb * 2.0;
  s += textureSampleLevel(srcTex, samp, uv + t * vec2<f32>( 1.0,  0.0), 0.0).rgb * 2.0;
  s += textureSampleLevel(srcTex, samp, uv + t * vec2<f32>( 0.0, -1.0), 0.0).rgb * 2.0;
  s += textureSampleLevel(srcTex, samp, uv + t * vec2<f32>( 0.0,  1.0), 0.0).rgb * 2.0;
  s += textureSampleLevel(srcTex, samp, uv + t * vec2<f32>(-1.0, -1.0), 0.0).rgb;
  s += textureSampleLevel(srcTex, samp, uv + t * vec2<f32>( 1.0, -1.0), 0.0).rgb;
  s += textureSampleLevel(srcTex, samp, uv + t * vec2<f32>(-1.0,  1.0), 0.0).rgb;
  s += textureSampleLevel(srcTex, samp, uv + t * vec2<f32>( 1.0,  1.0), 0.0).rgb;
  return vec4<f32>(s / 16.0, 1.0);
}