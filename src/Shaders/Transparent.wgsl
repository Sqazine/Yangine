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
struct ObjectU {
  model        : mat4x4<f32>,
  prevModel    : mat4x4<f32>,
  normalMatrix : mat4x4<f32>,
  baseColor    : vec4<f32>,
  mats         : vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame : Frame;
@group(1) @binding(0) var<uniform> obj   : ObjectU;

struct VSOut {
  @builtin(position) pos      : vec4<f32>,
  @location(0)      worldPos  : vec3<f32>,
  @location(1)      normal    : vec3<f32>,
  @location(2)      uv        : vec2<f32>,
};

@vertex
fn vs(@location(0) position : vec3<f32>,
      @location(1) normal   : vec3<f32>,
      @location(2) uv       : vec2<f32>) -> VSOut {
  var o : VSOut;
  let world = obj.model * vec4<f32>(position, 1.0);
  o.worldPos = world.xyz;
  o.normal   = (obj.normalMatrix * vec4<f32>(normal, 0.0)).xyz;
  o.uv       = uv;
  o.pos = frame.viewProj * world;
  return o;
}

@fragment
fn fs(i : VSOut) -> @location(0) vec4<f32> {
  let N = normalize(i.normal);
  let V = normalize(frame.camPos.xyz - i.worldPos);
  let L = normalize(-frame.lightDir.xyz);
  let H = normalize(V + L);
  let NdotL = max(dot(N, L), 0.0);
  let NdotV = max(dot(N, V), 1e-4);
  let base  = obj.baseColor.rgb;
  let opacity = obj.mats.w;
  let lit = base * (0.22 + 0.78 * NdotL);
  let fres = pow(1.0 - NdotV, 3.0);
  let spec = pow(max(dot(N, H), 0.0), 64.0) * 0.6;
  let color = lit + fres * 0.5 + vec3<f32>(spec);
  return vec4<f32>(color, opacity);
}