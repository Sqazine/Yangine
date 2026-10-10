// 透明前向渲染。
//
// **半透物体不接受阴影**：玻璃体只做光照与菲涅尔，不采样任何阴影贴图，
// 因此不需要绑定 CSM / Cube / Spot 阴影资源，少一个绑定组、少一次纹理采样。
// 这是有意为之的取舍——半影在半透表面上几乎不可见，而每个片元跑 36 次 PCSS 采样
// 对透明叠层来说性价比极低；更重要的是透明体「接收阴影」会让重叠处出现双重暗化。

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

struct Light {
  position  : vec4<f32>,
  direction : vec4<f32>,
  color     : vec4<f32>,
  params    : vec4<f32>,
  shadowP   : vec4<f32>,
  cone      : vec4<f32>,
  csmSizes  : vec4<f32>,
  misc      : vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame : Frame;
@group(1) @binding(0) var<uniform> obj   : ObjectU;

@group(2) @binding(0) var<storage, read> lights     : array<Light>;
@group(2) @binding(1) var<storage, read> shadowMats : array<mat4x4<f32>>;

struct VSOut {
  @builtin(position) pos      : vec4<f32>,
  @location(0)      worldPos  : vec3<f32>,
  @location(1)      normal    : vec3<f32>,
};

@vertex
fn vs(@location(0) position : vec3<f32>,
      @location(1) normal   : vec3<f32>,
      @location(2) uv       : vec2<f32>) -> VSOut {
  var o : VSOut;
  let world = obj.model * vec4<f32>(position, 1.0);
  o.worldPos = world.xyz;
  o.normal   = (obj.normalMatrix * vec4<f32>(normal, 0.0)).xyz;
  o.pos = frame.viewProj * world;
  return o;
}

const TYPE_DIR = 0.0;
const TYPE_SPOT = 2.0;

@fragment
fn fs(i : VSOut) -> @location(0) vec4<f32> {
  let N = normalize(i.normal);
  let V = normalize(frame.camPos.xyz - i.worldPos);
  let NdotV = max(dot(N, V), 1e-4);
  let base = obj.baseColor.rgb;
  let opacity = obj.mats.w;

  var color = vec3<f32>(0.0);
  var highlight = vec3<f32>(0.0);
  let lightCount = u32(frame.params.x);
  for (var li = 0u; li < lightCount; li = li + 1u) {
    let lt = lights[li];
    let ltype = lt.params.x;

    var toLight : vec3<f32>;
    var atten : f32;
    if (ltype == TYPE_DIR) {
      toLight = -normalize(lt.direction.xyz);
      atten = 1.0;
    } else {
      let delta = lt.position.xyz - i.worldPos;
      let dist = length(delta);
      if (dist < 1e-4) { continue; }
      toLight = delta / dist;
      let d2 = dist * dist;
      let win = clamp(1.0 - pow(dist / max(lt.position.w, 1e-3), 4.0), 0.0, 1.0);
      atten = win * win / d2;
    }

    let NdotL = dot(N, toLight);
    if (NdotL <= 0.0) { continue; }

    if (ltype == TYPE_SPOT) {
      let cosA = dot(-toLight, normalize(lt.direction.xyz));
      let cone = smoothstep(lt.direction.w, lt.cone.x, cosA);
      if (cone <= 0.0) { continue; }
      atten *= cone;
    }
    if (atten <= 1e-6) { continue; }

    // 不受阴影：阴影项恒为 1
    let radiance = lt.color.rgb * lt.color.a * atten;
    color += base * radiance * NdotL;

    // 逐光源累加高光，形状随光源方向变化
    let H = normalize(V + toLight);
    let spec = pow(max(dot(N, H), 0.0), 96.0) * 0.9;
    highlight += lt.color.rgb * lt.color.a * atten * spec;
  }

  // 菲涅尔边缘 + 累积高光 + 环境项
  let fres = pow(1.0 - NdotV, 3.0);
  color += base * 0.10 + fres * 0.32 + highlight;
  color += base * obj.mats.z;

  return vec4<f32>(color, opacity);
}
