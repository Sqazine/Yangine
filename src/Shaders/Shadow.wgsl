// 阴影通道：每条槽位（CSM 级联 / 立方体面 / 聚光）各自渲染一次。
// 投影矩阵通过 group(2) 的槽位 uniform 传入，因此同一套管线可复用于任意光源类型。

struct ObjectU {
  model        : mat4x4<f32>,
  prevModel    : mat4x4<f32>,
  normalMatrix : mat4x4<f32>,
  baseColor    : vec4<f32>,
  mats         : vec4<f32>,
};

@group(1) @binding(0) var<uniform> obj     : ObjectU;
@group(2) @binding(0) var<uniform> slotVP : mat4x4<f32>;

@vertex
fn vs(@location(0) position : vec3<f32>) -> @builtin(position) vec4<f32> {
  let world = obj.model * vec4<f32>(position, 1.0);
  return slotVP * world;
}