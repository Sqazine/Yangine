// 前向着色：逐物体渲染，片元着色器内联完成光照。两个入口点：
//   fs          → Forward+：只遍历 LightCull 为该屏幕分块挑选出的光源
//   fs_forward  → 传统 Forward：单趟绘制，遍历**全部**光源（无 G-buffer、无剔除），
//                 并额外产出 velocity（没有前置通道，TAA 的运动矢量只能由它自己写）
// 二者的 BRDF、阴影（PCSS + CSM + Cube + Spot）完全共用 shadePixel()/shadeLight()，
// 与延迟路径的 shadeLight() 也逐字对应，因此四条路径的画质应当一致，
// 差异只体现在「光照在前向还是延迟」「光源集合是否分块裁剪」这两件事上。

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
  misc          : vec4<f32>,  // z/w = tileSize
  gridParams    : vec4<f32>,  // x=tilesX y=tilesY z=lightsPerTile
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

@group(2) @binding(0) var shadowCsmTex   : texture_depth_2d_array;
@group(2) @binding(1) var shadowLocalTex : texture_depth_2d_array;
@group(2) @binding(2) var shadowSamp     : sampler_comparison;
@group(2) @binding(3) var aoTex          : texture_2d<f32>;
@group(2) @binding(4) var samp           : sampler;

@group(3) @binding(0) var<storage, read> lights       : array<Light>;
@group(3) @binding(1) var<storage, read> shadowMats   : array<mat4x4<f32>>;
@group(3) @binding(2) var<storage, read> lightGrid    : array<u32>;
@group(3) @binding(3) var<storage, read> lightIndices : array<u32>;

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

// 传统 Forward 需要自己写 velocity（它没有前置通道，而 TAA 需要运动矢量），
// 所以单独一套 varying。不并进上面的 VSOut：那样会让 Forward+ 白白多插值 4 个分量，
// 而省带宽正是 Forward+ 存在的理由。
struct VSOutV {
  @builtin(position) pos      : vec4<f32>,
  @location(0)      worldPos  : vec3<f32>,
  @location(1)      normal    : vec3<f32>,
  @location(2)      velCurr   : vec2<f32>,
  @location(3)      velPrev   : vec2<f32>,
};

// 传统 Forward 的顶点着色器：与 vs 相同，额外算当前/上一帧的投影坐标供 velocity 使用
@vertex
fn vs_forward(@location(0) position : vec3<f32>,
              @location(1) normal   : vec3<f32>,
              @location(2) uv       : vec2<f32>) -> VSOutV {
  var o : VSOutV;
  let world = obj.model * vec4<f32>(position, 1.0);
  o.worldPos = world.xyz;
  o.normal   = (obj.normalMatrix * vec4<f32>(normal, 0.0)).xyz;
  let clip = frame.viewProj * world;
  let prevClip = frame.prevViewProj * (obj.prevModel * vec4<f32>(position, 1.0));
  o.velCurr = clip.xy / clip.w;
  o.velPrev = prevClip.xy / prevClip.w;
  o.pos = clip;
  return o;
}

const PI = 3.14159265359;
const BLOCKER_SAMPLES = 12u;
const PCF_SAMPLES = 24u;
const TYPE_DIR = 0.0;
const TYPE_POINT = 1.0;
const TYPE_SPOT = 2.0;

// ---------- BRDF ----------
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

// ---------- PCSS ----------
fn vogelDisk(i : u32, n : u32, rot : f32) -> vec2<f32> {
  let r = sqrt((f32(i) + 0.5) / f32(n));
  let theta = f32(i) * 2.39996323 + rot;
  return vec2<f32>(r * cos(theta), r * sin(theta));
}
fn hash12(p : vec2<f32>) -> f32 {
  var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
fn linearizeDepth(d : f32, zNear : f32, zFar : f32, isOrtho : f32) -> f32 {
  if (isOrtho > 0.5) { return zNear + d * (zFar - zNear); }
  return (zFar * zNear) / max(zFar - d * (zFar - zNear), 1e-5);
}
fn depthAt(tex : texture_depth_2d_array, layer : i32, uv : vec2<f32>) -> f32 {
  let dim = vec2<i32>(textureDimensions(tex));
  let c = clamp(vec2<i32>(uv * vec2<f32>(dim)), vec2<i32>(0), dim - vec2<i32>(1));
  return textureLoad(tex, c, layer, 0);
}

fn pcss2D(
  tex : texture_depth_2d_array,
  layer : i32,
  uv : vec2<f32>,
  compareDepth : f32,
  zNear : f32,
  zFar : f32,
  isOrtho : f32,
  lightSizeUV : f32,
  penumbraScale : f32,
  rot : f32
) -> f32 {
  let dim = vec2<f32>(textureDimensions(tex));
  let oneTexel = 1.0 / dim;

  let searchUV = clamp(lightSizeUV * 2.5, oneTexel.x * 2.0, 0.08);
  var blockerSum = 0.0;
  var blockerCount = 0.0;
  for (var i = 0u; i < BLOCKER_SAMPLES; i = i + 1u) {
    let o = vogelDisk(i, BLOCKER_SAMPLES, rot) * searchUV;
    let sd = depthAt(tex, layer, uv + o);
    if (sd < compareDepth) {
      blockerSum += sd;
      blockerCount += 1.0;
    }
  }
  if (blockerCount < 0.5) { return 1.0; }
  let blockerDepth = blockerSum / blockerCount;

  let dRecv = linearizeDepth(compareDepth, zNear, zFar, isOrtho);
  let dBlk  = linearizeDepth(blockerDepth, zNear, zFar, isOrtho);
  var penumbraUV : f32;
  if (isOrtho > 0.5) {
    penumbraUV = max(dRecv - dBlk, 0.0) * lightSizeUV;
  } else {
    penumbraUV = (max(dRecv - dBlk, 0.0) / max(dBlk, 1e-3)) * lightSizeUV;
  }
  penumbraUV = clamp(penumbraUV * penumbraScale, oneTexel.x, 0.12);

  var sum = 0.0;
  for (var i = 0u; i < PCF_SAMPLES; i = i + 1u) {
    let o = vogelDisk(i, PCF_SAMPLES, rot + 1.7) * penumbraUV;
    sum += textureSampleCompareLevel(tex, shadowSamp, uv + o, layer, compareDepth);
  }
  return sum / f32(PCF_SAMPLES);
}

// ---------- 阴影解算 ----------
fn dirToCubeUV(dir : vec3<f32>) -> vec3<f32> {
  let a = abs(dir);
  var face : f32;
  var sc : f32;
  var tc : f32;
  var ma : f32;
  if (a.x >= a.y && a.x >= a.z) {
    ma = a.x; sc = -dir.z; tc = -dir.y;
    face = select(1.0, 0.0, dir.x > 0.0);
  } else if (a.y >= a.z) {
    ma = a.y; sc = dir.x;  tc = dir.z;
    face = select(3.0, 2.0, dir.y > 0.0);
  } else {
    ma = a.z; sc = dir.x;  tc = -dir.y;
    face = select(5.0, 4.0, dir.z > 0.0);
  }
  return vec3<f32>((sc / ma + 1.0) * 0.5, (tc / ma + 1.0) * 0.5, face);
}

fn projectToUV(vp : mat4x4<f32>, worldPos : vec3<f32>) -> vec3<f32> {
  let clip = vp * vec4<f32>(worldPos, 1.0);
  let ndc = clip.xyz / clip.w;
  return vec3<f32>(ndc.x * 0.5 + 0.5, 0.5 - 0.5 * ndc.y, ndc.z);
}

// csmSplits[c] 是级联 c 的远界（矩阵按 [splits[c-1], splits[c]] 构建），
// 故选「第一个 viewDepth ≤ splits[c] 的 c」。写成 `cascade = c` 会差一位，
// 使远处选中的正交框不含该点 → projectToUV 出界 → return 1.0，阴影静默消失。
fn shadowCascade(li : u32, worldPos : vec3<f32>, viewDepth : f32, rot : f32) -> f32 {
  let count = i32(lights[li].params.z);
  if (count <= 0) { return 1.0; }

  var cascade = 0;
  for (var c = 0; c < count; c = c + 1) {
    if (viewDepth > frame.csmSplits[c]) { cascade = c + 1; }
  }
  cascade = min(cascade, count - 1);

  let matBase = i32(lights[li].params.y);
  let bias = lights[li].params.w + f32(cascade) * 0.0009;
  let cmp1 = projectToUV(shadowMats[matBase + cascade], worldPos);
  if (cmp1.z <= 0.0 || any(cmp1.xy < vec2<f32>(0.0)) || any(cmp1.xy > vec2<f32>(1.0))) {
    return 1.0;
  }
  let s1 = pcss2D(shadowCsmTex, cascade, cmp1.xy, cmp1.z - bias,
                  lights[li].shadowP.z, lights[li].shadowP.w, 1.0,
                  lights[li].csmSizes[cascade], lights[li].misc.x, rot);
  if (cascade + 1 >= count) { return s1; }

  // 混合带必须落在本级联区间 [splits[cascade-1], splits[cascade]] 内，否则融合永不触发
  let bandEnd = frame.csmSplits[cascade];
  var bandPrev = 0.0;
  if (cascade > 0) { bandPrev = frame.csmSplits[cascade - 1]; }
  let bandStart = mix(bandPrev, bandEnd, 0.75);
  if (viewDepth < bandStart) { return s1; }

  let cmp2 = projectToUV(shadowMats[matBase + cascade + 1], worldPos);
  if (cmp2.z <= 0.0 || any(cmp2.xy < vec2<f32>(0.0)) || any(cmp2.xy > vec2<f32>(1.0))) {
    return s1;
  }
  let s2 = pcss2D(shadowCsmTex, cascade + 1, cmp2.xy, cmp2.z - (bias + 0.0009),
                  lights[li].shadowP.z, lights[li].shadowP.w, 1.0,
                  lights[li].csmSizes[cascade + 1], lights[li].misc.x, rot + 2.3);
  return mix(s1, s2, smoothstep(bandStart, bandEnd, viewDepth));
}

fn shadowCube(li : u32, worldPos : vec3<f32>, rot : f32) -> f32 {
  let matBase = i32(lights[li].params.y);
  let dir = normalize(lights[li].position.xyz - worldPos);
  let face = dirToCubeUV(dir);
  let cmp = projectToUV(shadowMats[matBase + i32(face.z)], worldPos);
  if (cmp.z <= 0.0) { return 1.0; }
  let layer = matBase + i32(face.z);
  let bias = lights[li].params.w + face.z * 0.0012;
  return pcss2D(shadowLocalTex, layer, face.xy, cmp.z - bias,
                lights[li].shadowP.z, lights[li].shadowP.w, 0.0,
                lights[li].shadowP.x, lights[li].misc.x, rot);
}

fn shadowSpot(li : u32, worldPos : vec3<f32>, rot : f32) -> f32 {
  let matBase = i32(lights[li].params.y);
  let cmp = projectToUV(shadowMats[matBase], worldPos);
  if (cmp.z <= 0.0 || any(cmp.xy < vec2<f32>(0.0)) || any(cmp.xy > vec2<f32>(1.0))) {
    return 1.0;
  }
  return pcss2D(shadowLocalTex, matBase, cmp.xy, cmp.z - lights[li].params.w,
                lights[li].shadowP.z, lights[li].shadowP.w, 0.0,
                lights[li].shadowP.x, lights[li].misc.x, rot);
}

/** 屏幕像素 → 分块索引。 */
fn tileIndex(px : vec2<f32>) -> u32 {
  let ts = vec2<f32>(max(frame.misc.z, 1.0), max(frame.misc.w, 1.0));
  let t = vec2<u32>(px / ts);
  let tilesX = u32(frame.gridParams.x);
  return min(t.y, u32(frame.gridParams.y) - 1u) * tilesX + min(t.x, tilesX - 1u);
}

/**
 * 单盏光源对当前着色点的贡献。被剔除（背向 / 超出锥 / 衰减为 0）时返回 0。
 *
 * 抽成函数是为了让 Forward+（遍历分块列表）与传统 Forward（遍历全部光源）
 * **共用同一套** BRDF 与阴影代码 —— 两条路径的差别只在「遍历哪些光源」。
 * WGSL 的 continue 不能跨函数，所以用 return vec3(0.0) 代替。
 */
fn shadeLight(
  li : u32,
  px : vec2<f32>,
  worldPos : vec3<f32>,
  N : vec3<f32>,
  V : vec3<f32>,
  NdotV : f32,
  viewDepth : f32,
  albedo : vec3<f32>,
  metallic : f32,
  rough : f32,
  F0 : vec3<f32>
) -> vec3<f32> {
  let lt = lights[li];
  let ltype = lt.params.x;

  var toLight : vec3<f32>;
  var atten : f32;
  if (ltype == TYPE_DIR) {
    toLight = -normalize(lt.direction.xyz);
    atten = 1.0;
  } else {
    let delta = lt.position.xyz - worldPos;
    let dist = length(delta);
    if (dist < 1e-4) { return vec3<f32>(0.0); }
    toLight = delta / dist;
    let d2 = dist * dist;
    let win = clamp(1.0 - pow(dist / max(lt.position.w, 1e-3), 4.0), 0.0, 1.0);
    atten = win * win / d2;
  }

  let NdotL = dot(N, toLight);
  if (NdotL <= 0.0) { return vec3<f32>(0.0); }

  if (ltype == TYPE_SPOT) {
    let cosA = dot(-toLight, normalize(lt.direction.xyz));
    let cone = smoothstep(lt.direction.w, lt.cone.x, cosA);
    if (cone <= 0.0) { return vec3<f32>(0.0); }
    atten *= cone;
  }
  if (atten <= 1e-6) { return vec3<f32>(0.0); }

  let rot = hash12(px + vec2<f32>(frame.params.w * 1.7));
  var shadow = 1.0;
  let strength = lt.shadowP.y;
  if (strength > 0.0 && lt.params.z > 0.0) {
    if (ltype == TYPE_DIR) {
      shadow = shadowCascade(li, worldPos, viewDepth, rot);
    } else if (ltype == TYPE_POINT) {
      shadow = shadowCube(li, worldPos, rot);
    } else {
      shadow = shadowSpot(li, worldPos, rot);
    }
    shadow = mix(1.0, shadow, strength);
  }

  let radiance = lt.color.rgb * lt.color.a * atten * shadow;
  let H = normalize(V + toLight);
  let NdotH = max(dot(N, H), 0.0);
  let VdotH = max(dot(V, H), 0.0);
  let D = distributionGGX(NdotH, rough);
  let G = geometrySmith(NdotV, NdotL, rough);
  let F = fresnelSchlick(VdotH, F0);
  let spec = (D * G * F) / (4.0 * NdotV * NdotL + 1e-4);
  let kd = (vec3<f32>(1.0) - F) * (1.0 - metallic);
  return (kd * albedo / PI + spec) * radiance * NdotL;
}

/**
 * 前向着色的公共主体。tiled = true 时只遍历本分块可见的光源（Forward+），
 * = false 时遍历全部光源（传统 Forward）。这是两条路径的唯一差别。
 */
fn shadePixel(
  px : vec2<f32>,
  worldPos : vec3<f32>,
  worldNormal : vec3<f32>,
  tiled : bool
) -> vec3<f32> {
  let N = normalize(worldNormal);
  let albedo = obj.baseColor.rgb;
  let metallic = obj.mats.x;
  let rough = clamp(obj.mats.y, 0.045, 1.0);

  let V = normalize(frame.camPos.xyz - worldPos);
  let NdotV = max(dot(N, V), 1e-4);
  let viewDepth = dot(worldPos - frame.camPos.xyz, frame.camFwd.xyz);

  let ao = textureLoad(aoTex, vec2<i32>(px), 0).r;
  let F0 = mix(vec3<f32>(0.04), albedo, metallic);
  var color = vec3<f32>(0.0);

  if (tiled) {
    // 只遍历该分块可见的光源
    let tile = tileIndex(px);
    let stride = u32(frame.gridParams.z);
    let n = min(lightGrid[tile], stride);
    for (var k = 0u; k < n; k = k + 1u) {
      color += shadeLight(lightIndices[tile * stride + k], px, worldPos,
                          N, V, NdotV, viewDepth, albedo, metallic, rough, F0);
    }
  } else {
    // 传统 Forward：逐物体、逐片元地过一遍**全部**光源，没有任何剔除
    let lightCount = u32(frame.params.x);
    for (var i = 0u; i < lightCount; i = i + 1u) {
      color += shadeLight(i, px, worldPos,
                          N, V, NdotV, viewDepth, albedo, metallic, rough, F0);
    }
  }

  // 环境项（乘 AO）+ 自发光
  let hemi = mix(vec3<f32>(0.020, 0.024, 0.032), vec3<f32>(0.075, 0.098, 0.145), N.y * 0.5 + 0.5);
  color += albedo * hemi * ao;
  color += albedo * obj.mats.z * 3.0;
  return color;
}

/// Forward+：着色阶段只遍历本分块的光源列表（深度由前置通道写好，这里 less-equal）。
@fragment
fn fs(frag : VSOut) -> @location(0) vec4<f32> {
  let color = shadePixel(frag.pos.xy, frag.worldPos, frag.normal, true);
  return vec4<f32>(color, 1.0);
}

struct ForwardOut {
  @location(0) color    : vec4<f32>,
  // 传统 Forward 没有前置通道，velocity 必须由这次绘制自己产出，否则 TAA 会拖影
  @location(1) velocity : vec2<f32>,
};

/// 传统 Forward：单趟绘制，光照在片元着色器里内联完成，无 G-buffer、无光源剔除。
@fragment
fn fs_forward(frag : VSOutV) -> ForwardOut {
  var o : ForwardOut;
  let color = shadePixel(frag.pos.xy, frag.worldPos, frag.normal, false);
  o.color = vec4<f32>(color, 1.0);
  let curUV  = vec2<f32>(0.5 * frag.velCurr.x + 0.5, 0.5 - 0.5 * frag.velCurr.y);
  let prevUV = vec2<f32>(0.5 * frag.velPrev.x + 0.5, 0.5 - 0.5 * frag.velPrev.y);
  o.velocity = curUV - prevUV;
  return o;
}
