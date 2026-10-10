// Forward+ 的前置通道：只写 albedo / normal / velocity + 深度。
//
// 与延迟渲染的 G-buffer 的区别是**不写世界坐标缓冲**——
// Forward+ 在真正的着色通道里由顶点插值直接拿到 worldPos，
// 因此省掉一张 rgba16float 的写入与带宽，这也是 Forward+ 相对 Deferred 的主要收益之一。

struct Frame {
  viewProj      : mat4x4<f32>,
  prevViewProj  : mat4x4<f32>,
  invViewProj   : mat4x4<f32>,
  view          : mat4x4<f32>,
  proj          : mat4x4<f32>,
  invProj       : mat4x4<f32>,
  camPos        : vec4<f32>,
  camFwd        : vec4<f32>,
  sunDir        : vec4<f32>,
  sunColor      : vec4<f32>,
  params        : vec4<f32>,
  csmSplits     : vec4<f32>,
  screen        : vec4<f32>,
  aoParams      : vec4<f32>,
  misc          : vec4<f32>,
  gridParams    : vec4<f32>,
};

struct ObjectU {
  model        : mat4x4<f32>,
  prevModel    : mat4x4<f32>,
  normalMatrix : mat4x4<f32>,
  baseColor    : vec4<f32>,
  mats         : vec4<f32>,  // x metallic y roughness z emissive w opacity
};

@group(0) @binding(0) var<uniform> frame : Frame;
@group(1) @binding(0) var<uniform> obj   : ObjectU;

struct VSOut {
  @builtin(position) pos      : vec4<f32>,
  @location(0)      normal    : vec3<f32>,
  @location(1)      velCurr   : vec2<f32>,
  @location(2)      velPrev   : vec2<f32>,
};

@vertex
fn vs(@location(0) position : vec3<f32>,
      @location(1) normal   : vec3<f32>,
      @location(2) uv       : vec2<f32>) -> VSOut {
  var o : VSOut;
  let world = obj.model * vec4<f32>(position, 1.0);
  o.normal = (obj.normalMatrix * vec4<f32>(normal, 0.0)).xyz;
  let clip = frame.viewProj * world;
  let prevClip = frame.prevViewProj * (obj.prevModel * vec4<f32>(position, 1.0));
  o.velCurr = clip.xy / clip.w;
  o.velPrev = prevClip.xy / prevClip.w;
  o.pos = clip;
  return o;
}

struct PrepassOut {
  @location(0) albedoMetal : vec4<f32>,
  @location(1) normalRough : vec4<f32>,
  @location(2) velocity    : vec2<f32>,
};

@fragment
fn fs(i : VSOut) -> PrepassOut {
  var o : PrepassOut;
  o.albedoMetal = vec4<f32>(obj.baseColor.rgb, obj.mats.x);
  o.normalRough = vec4<f32>(normalize(i.normal), obj.mats.y);
  let curUV  = vec2<f32>(0.5 * i.velCurr.x + 0.5, 0.5 - 0.5 * i.velCurr.y);
  let prevUV = vec2<f32>(0.5 * i.velPrev.x + 0.5, 0.5 - 0.5 * i.velPrev.y);
  o.velocity = curUV - prevUV;
  return o;
}
