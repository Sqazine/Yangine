struct Frame {
  viewProj      : mat4x4<f32>,
  prevViewProj  : mat4x4<f32>,
  lightViewProj : mat4x4<f32>,
  invViewProj   : mat4x4<f32>,
  camPos        : vec4<f32>,
  lightDir      : vec4<f32>,
  lightColor    : vec4<f32>,
  params        : vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame : Frame;
@group(1) @binding(0) var albedoMetalTex : texture_2d<f32>;
@group(1) @binding(1) var normalRoughTex : texture_2d<f32>;
@group(1) @binding(2) var posEmTex       : texture_2d<f32>;
@group(1) @binding(3) var shadowTex      : texture_depth_2d;
@group(1) @binding(4) var shadowSamp     : sampler_comparison;
@group(1) @binding(5) var depthTex       : texture_depth_2d;
@group(1) @binding(6) var samp           : sampler;

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

const PI = 3.14159265359;

fn distributionGGX(NdotH : f32, rough : f32) -> f32 {
  let a  = rough * rough;
  let a2 = a * a;
  let d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
  return a2 / max(PI * d * d, 1e-7);
}
fn geometrySchlick(NdotV : f32, rough : f32) -> f32 {
  let r = rough + 1.0;
  let k = (r * r) / 8.0;
  return NdotV / (NdotV * (1.0 - k) + k);
}
fn geometrySmith(NdotV : f32, NdotL : f32, rough : f32) -> f32 {
  return geometrySchlick(NdotV, rough) * geometrySchlick(NdotL, rough);
}
fn fresnelSchlick(cosT : f32, F0 : vec3<f32>) -> vec3<f32> {
  return F0 + (vec3<f32>(1.0) - F0) * pow(clamp(1.0 - cosT, 0.0, 1.0), 5.0);
}

fn sampleShadow(worldPos : vec3<f32>) -> f32 {
  let lc  = frame.lightViewProj * vec4<f32>(worldPos, 1.0);
  let ndc = lc.xyz / lc.w;
  let uv  = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - 0.5 * ndc.y);
  if (ndc.z > 1.0 || uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
    return 1.0;
  }
  let bias = 0.0025;
  return textureSampleCompareLevel(shadowTex, shadowSamp, uv, ndc.z - bias);
}

@fragment
fn fs(frag : VSOut) -> @location(0) vec4<f32> {
  let coord = vec2<i32>(frag.pos.xy);
  let depth = textureLoad(depthTex, coord, 0);
  if (depth >= 1.0) {
    let uv  = frag.pos.xy / vec2<f32>(textureDimensions(albedoMetalTex));
    let t   = clamp(uv.y, 0.0, 1.0);
    let sky = mix(vec3<f32>(0.015, 0.02, 0.04), vec3<f32>(0.09, 0.13, 0.22), t);
    return vec4<f32>(sky, 1.0);
  }
  let uv = frag.pos.xy / vec2<f32>(textureDimensions(albedoMetalTex));
  let am = textureSampleLevel(albedoMetalTex, samp, uv, 0.0);
  let nr = textureSampleLevel(normalRoughTex, samp, uv, 0.0);
  let pe = textureSampleLevel(posEmTex,       samp, uv, 0.0);

  let N      = normalize(nr.xyz);
  let albedo = am.rgb;
  let metallic = am.a;
  let rough  = clamp(nr.a, 0.045, 1.0);
  let worldPos = pe.xyz;

  let V = normalize(frame.camPos.xyz - worldPos);
  let L = normalize(-frame.lightDir.xyz);
  let H = normalize(V + L);
  let NdotL = max(dot(N, L), 0.0);
  let NdotV = max(dot(N, V), 1e-4);
  let NdotH = max(dot(N, H), 0.0);
  let VdotH = max(dot(V, H), 0.0);

  let shadow   = sampleShadow(worldPos);
  let radiance = frame.lightColor.rgb * frame.lightColor.a * shadow;

  let F0 = mix(vec3<f32>(0.04), albedo, metallic);
  let D  = distributionGGX(NdotH, rough);
  let G  = geometrySmith(NdotV, NdotL, rough);
  let F  = fresnelSchlick(VdotH, F0);
  let spec = (D * G * F) / (4.0 * NdotV * NdotL + 1e-4);

  let kd = (vec3<f32>(1.0) - F) * (1.0 - metallic);
  let diffuse = kd * albedo / PI;
  var color = (diffuse + spec) * radiance * NdotL;
  color += albedo * 0.06;
  color += albedo * pe.a * 3.0;
  return vec4<f32>(color, 1.0);
}