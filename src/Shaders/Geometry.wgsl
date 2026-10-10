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
  params        : vec4<f32>,  // x=lightCount y=camNear z=camFar w=time
  csmSplits     : vec4<f32>,  // 各级联远界（世界距离）
  screen        : vec4<f32>,  // w h 1/w 1/h
  aoParams      : vec4<f32>,  // x=radius y=intensity z=power w=noiseSeed
  misc          : vec4<f32>,  // x=renderPath y=aoMethod z=tileSizeX w=tileSizeY
  gridParams    : vec4<f32>,  // x=tilesX y=tilesY z=lightsPerTile w=pad
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
  @location(3)      velCurr   : vec2<f32>,
  @location(4)      velPrev   : vec2<f32>,
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
  let clip     = frame.viewProj * world;
  let prevClip = frame.prevViewProj * (obj.prevModel * vec4<f32>(position, 1.0));
  o.velCurr = clip.xy / clip.w;
  o.velPrev = prevClip.xy / prevClip.w;
  o.pos = clip;
  return o;
}

struct GBufferOut {
  @location(0) albedoMetal : vec4<f32>,
  @location(1) normalRough : vec4<f32>,
  @location(2) positionEm  : vec4<f32>,
  @location(3) velocity    : vec2<f32>,
};

@fragment
fn fs(i : VSOut) -> GBufferOut {
  var o : GBufferOut;
  o.albedoMetal = vec4<f32>(obj.baseColor.rgb, obj.mats.x);
  o.normalRough = vec4<f32>(normalize(i.normal), obj.mats.y);
  o.positionEm  = vec4<f32>(i.worldPos, obj.mats.z);
  let curUV  = vec2<f32>(0.5 * i.velCurr.x + 0.5, 0.5 - 0.5 * i.velCurr.y);
  let prevUV = vec2<f32>(0.5 * i.velPrev.x + 0.5, 0.5 - 0.5 * i.velPrev.y);
  o.velocity = curUV - prevUV;
  return o;
}