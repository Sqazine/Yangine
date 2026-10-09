struct CompU {
  resolution     : vec2<f32>,
  exposure       : f32,
  bloomIntensity : f32,
};
@group(0) @binding(0) var hdrTex   : texture_2d<f32>;
@group(0) @binding(1) var bloomTex : texture_2d<f32>;
@group(0) @binding(2) var samp     : sampler;
@group(0) @binding(3) var<uniform> u : CompU;

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

fn aces(x : vec3<f32>) -> vec3<f32> {
  let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs(frag : VSOut) -> @location(0) vec4<f32> {
  let uv = frag.pos.xy / u.resolution;
  let hdr   = textureSampleLevel(hdrTex, samp, uv, 0.0).rgb;
  let bloom = textureSampleLevel(bloomTex, samp, uv, 0.0).rgb;
  var color = hdr + bloom * u.bloomIntensity;
  color *= u.exposure;
  color = aces(color);
  color = pow(color, vec3<f32>(1.0 / 2.2));
  return vec4<f32>(color, 1.0);
}