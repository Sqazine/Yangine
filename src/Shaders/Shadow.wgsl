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

@vertex
fn vs(@location(0) position : vec3<f32>) -> @builtin(position) vec4<f32> {
  let world = obj.model * vec4<f32>(position, 1.0);
  return frame.lightViewProj * world;
}