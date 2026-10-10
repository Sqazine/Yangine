// 天空背景：全屏竖直渐变，写进 HDR 场景色。
//
// 为什么前向路径需要独立的一趟：
//   延迟路径的全屏光照着色器本来就覆盖每一个像素，顺手在 depth >= 1 处返回天空即可；
//   而前向路径只在几何体覆盖到的像素上执行片元着色器，**背景像素永远不会被执行到**，
//   所以天空必须单独铺一趟，否则背景就是清屏色（黑）——四条路径会一眼看出不一样。
//
// 渐变色必须与 Lighting.wgsl 的 sky 分支逐字一致，否则同一场景在延迟与前向路径下
// 背景不同。verify_layout.py 会比对两处的常量。

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

@group(0) @binding(0) var<uniform> frame : Frame;

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
  let uv = frag.pos.xy * frame.screen.zw;
  let t = clamp(uv.y, 0.0, 1.0);
  return vec4<f32>(mix(vec3<f32>(0.015, 0.02, 0.04), vec3<f32>(0.09, 0.13, 0.22), t), 1.0);
}
