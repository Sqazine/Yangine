// 延迟光照：读 G-buffer，逐光源解算阴影后做 PBR 累加。两个入口点：
//   fs        → Deferred：遍历**全部**光源（无裁剪），每条光线都要过一遍所有灯
//   fs_tiled  → Tiled Deferred：只遍历**本分块**可见的光源列表（LightCull.wgsl 产出）
// 二者的光照模型、阴影解算、环境项完全共用 shadePixel() / shadeLight()，
// 差别只在「遍历哪些光源」——这样改一边不会漏改另一边。
//
// 阴影技术：
//   - 平行光 → Cascade Shadow Map，按视空间深度选级联，边界做双级联混合
//   - 点光源 → Cube Shadow Map，着色器内做面选择，PCSS 走面内 UV
//   - 聚光灯 → 单张 2D 阴影图
// 三者共用同一套 PCSS：blocker search → 半影估计 → Vogel 盘 PCF。

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

struct Light {
  position  : vec4<f32>,  // xyz 位置, w 衰减半径
  direction : vec4<f32>,  // xyz 方向, w cos(outer)
  color     : vec4<f32>,  // rgb 颜色, a 强度
  params    : vec4<f32>,  // x type, y matBase, z matCount, w baseBias
  shadowP   : vec4<f32>,  // x lightSizeUV, y shadowStrength, z near, w far
  cone      : vec4<f32>,  // x cosInner, y isOrtho, z rangeFalloff, w texelWorld
  csmSizes  : vec4<f32>,  // 各级联的 lightSizeUV
  misc      : vec4<f32>,  // x=penumbraScale, y=attenuationRadius, zw=spare
};

@group(0) @binding(0) var<uniform> frame : Frame;

@group(1) @binding(0) var albedoMetalTex : texture_2d<f32>;
@group(1) @binding(1) var normalRoughTex : texture_2d<f32>;
@group(1) @binding(2) var posEmTex       : texture_2d<f32>;
@group(1) @binding(3) var depthTex       : texture_depth_2d;
@group(1) @binding(4) var samp           : sampler;
@group(1) @binding(5) var aoTex          : texture_2d<f32>;  // r16float，存可见度

@group(2) @binding(0) var shadowCsmTex   : texture_depth_2d_array;   // 2048², CSM
@group(2) @binding(1) var shadowLocalTex : texture_depth_2d_array;   // 1024², cube + spot
@group(2) @binding(2) var shadowSamp     : sampler_comparison;

@group(3) @binding(0) var<storage, read> lights     : array<Light>;
@group(3) @binding(1) var<storage, read> shadowMats : array<mat4x4<f32>>;
// Tiled Deferred 专用：分块光源列表（LightCull.wgsl 产出）。
// fs 这条管线的布局里没有这两个 binding —— 模块级声明但入口点不使用是合法的。
@group(3) @binding(2) var<storage, read> lightGrid    : array<u32>;
@group(3) @binding(3) var<storage, read> lightIndices : array<u32>;

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

// ---------- PCSS 工具 ----------

// 黄金角 Vogel 盘：均匀覆盖圆盘且无明显栅格感。
fn vogelDisk(i : u32, n : u32, rot : f32) -> vec2<f32> {
  let r = sqrt((f32(i) + 0.5) / f32(n));
  let theta = f32(i) * 2.39996323 + rot;
  return vec2<f32>(r * cos(theta), r * sin(theta));
}

// 逐像素随机旋转，让采样噪声被 TAA 摊平而不是形成固定图案。
fn hash12(p : vec2<f32>) -> f32 {
  var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

// 把阴影图深度线性化。正交为线性，透视需反解 WebGPU 的 z_ndc 映射。
fn linearizeDepth(d : f32, zNear : f32, zFar : f32, isOrtho : f32) -> f32 {
  if (isOrtho > 0.5) {
    return zNear + d * (zFar - zNear);
  }
  return (zFar * zNear) / max(zFar - d * (zFar - zNear), 1e-5);
}

fn depthAt(tex : texture_depth_2d_array, layer : i32, uv : vec2<f32>) -> f32 {
  let dim = vec2<i32>(textureDimensions(tex));
  let c = clamp(vec2<i32>(uv * vec2<f32>(dim)), vec2<i32>(0), dim - vec2<i32>(1));
  return textureLoad(tex, c, layer, 0);
}

/**
 * 2D 阴影图的 PCSS。三个阶段：
 *   1. blocker search：以 searchUV 半径采样，取比受光点更近的平均深度
 *   2. 半影估计：由遮挡体距离算出透视/正交下的半影宽度（UV 单位）
 *   3. PCF：以该宽度为半径做 Vogel 盘过滤
 * 搜索半径与过滤半径解耦，因此半影是物理正确的（而非固定大小的糊边）。
 */
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

  // ---- 1. blocker search ----
  // 搜索半径取光源尺寸的固定倍数：足够找到近处遮挡体，又不会漏掉大半影
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
  // 无遮挡体 → 全亮
  if (blockerCount < 0.5) { return 1.0; }
  let blockerDepth = blockerSum / blockerCount;

  // ---- 2. 半影估计 ----
  let dRecv = linearizeDepth(compareDepth, zNear, zFar, isOrtho);
  let dBlk  = linearizeDepth(blockerDepth, zNear, zFar, isOrtho);
  var penumbraUV : f32;
  if (isOrtho > 0.5) {
    // 正交：半影世界宽度 = (d_recv - d_blk) * lightSizeUV（lightSizeUV 已含 tanθ/世界宽度）
    penumbraUV = max(dRecv - dBlk, 0.0) * lightSizeUV;
  } else {
    // 透视：半影与遮挡体到光源的距离成反比
    penumbraUV = (max(dRecv - dBlk, 0.0) / max(dBlk, 1e-3)) * lightSizeUV;
  }
  penumbraUV = clamp(penumbraUV * penumbraScale, oneTexel.x, 0.12);

  // ---- 3. PCF ----
  var sum = 0.0;
  for (var i = 0u; i < PCF_SAMPLES; i = i + 1u) {
    let o = vogelDisk(i, PCF_SAMPLES, rot + 1.7) * penumbraUV;
    sum += textureSampleCompareLevel(tex, shadowSamp, uv + o, layer, compareDepth);
  }
  return sum / f32(PCF_SAMPLES);
}

// ---------- 立方体面选择 ----------
// 返回 (u, v, face)。面序与 CUBE_FACE_NORMALS 一致。
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
  let uv = vec2<f32>((sc / ma + 1.0) * 0.5, (tc / ma + 1.0) * 0.5);
  return vec3<f32>(uv, face);
}

fn projectToUV(vp : mat4x4<f32>, worldPos : vec3<f32>) -> vec3<f32> {
  let clip = vp * vec4<f32>(worldPos, 1.0);
  let ndc = clip.xyz / clip.w;
  return vec3<f32>(ndc.x * 0.5 + 0.5, 0.5 - 0.5 * ndc.y, ndc.z);
}

// ---------- 各类光源的阴影解算 ----------

// 级联：按视空间深度定位级联，落在混合带内则与下一级联做 smoothstep 融合。
//
// csmSplits[c] 是级联 c 的**远界**，级联矩阵也按 [splits[c-1], splits[c]] 构建，
// 所以这里必须选「第一个满足 viewDepth ≤ splits[c] 的 c」。
// 曾经这里写成 `if (viewDepth > splits[c]) cascade = c;`（把 splits[c] 当近界），
// 差一位：viewDepth 一旦超出 splits[0]，选中的级联就小一级，其正交框不含该点 →
// projectToUV 出界 → 直接 return 1.0，阴影整片静默消失，且最后一级联几乎永不采样。
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
  let layer = cascade;
  let lsz = lights[li].csmSizes[cascade];
  let s1 = pcss2D(shadowCsmTex, layer, cmp1.xy, cmp1.z - bias,
                  lights[li].shadowP.z, lights[li].shadowP.w, 1.0,
                  lsz, lights[li].misc.x, rot);
  if (cascade + 1 >= count) { return s1; }

  // 混合带：本级联的**最后 25%**与下一级联融合。
  // 带子必须落在本级联自己的区间 [splits[cascade-1], splits[cascade]] 内；
  // 写成 mix(csmSplits[cascade], csmSplits[cascade+1], 0.75) 会让 bandStart 落在下一级联区间里，
  // 于是 `viewDepth < bandStart` 恒成立、融合永不触发（级联边界露出硬接缝）。
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
  let t = smoothstep(bandStart, bandEnd, viewDepth);
  return mix(s1, s2, t);
}

fn shadowCube(li : u32, worldPos : vec3<f32>, L : vec3<f32>, rot : f32) -> f32 {
  let matBase = i32(lights[li].params.y);
  let dir = normalize(lights[li].position.xyz - worldPos);
  let face = dirToCubeUV(dir);
  let vp = shadowMats[matBase + i32(face.z)];
  let cmp = projectToUV(vp, worldPos);
  if (cmp.z <= 0.0) { return 1.0; }
  let layer = matBase + i32(face.z);
  // 每个面的 depthBias 递增，避免相邻面在物体边界处互相泄漏
  let bias = lights[li].params.w + face.z * 0.0012;
  return pcss2D(shadowLocalTex, layer, face.xy, cmp.z - bias,
                lights[li].shadowP.z, lights[li].shadowP.w, 0.0,
                lights[li].shadowP.x, lights[li].misc.x, rot);
}

fn shadowSpot(li : u32, worldPos : vec3<f32>, rot : f32) -> f32 {
  let matBase = i32(lights[li].params.y);
  let cmp = projectToUV(shadowMats[matBase], worldPos);
  // 投影图只覆盖外锥，锥外直接视为无遮挡
  if (cmp.z <= 0.0 || any(cmp.xy < vec2<f32>(0.0)) || any(cmp.xy > vec2<f32>(1.0))) {
    return 1.0;
  }
  let bias = lights[li].params.w;
  return pcss2D(shadowLocalTex, matBase, cmp.xy, cmp.z - bias,
                lights[li].shadowP.z, lights[li].shadowP.w, 0.0,
                lights[li].shadowP.x, lights[li].misc.x, rot);
}

/** 屏幕像素 → 分块索引（与 LightCull.wgsl 的 TILE、ForwardPlus.wgsl 的同名函数一致）。 */
fn tileIndex(px : vec2<f32>) -> u32 {
  let ts = vec2<f32>(max(frame.misc.z, 1.0), max(frame.misc.w, 1.0));
  let t = vec2<u32>(px / ts);
  let tilesX = u32(frame.gridParams.x);
  return min(t.y, u32(frame.gridParams.y) - 1u) * tilesX + min(t.x, tilesX - 1u);
}

/**
 * 单盏光源对当前着色点的贡献。被剔除（背向 / 超出锥 / 衰减为 0）时返回 0。
 *
 * 抽成函数是为了让 fs（遍历全部光源）与 fs_tiled（遍历分块列表）**共用同一套**
 * BRDF 与阴影代码 —— 两条路径的差别只在「遍历哪些光源」，其余必须逐字一致，
 * 否则迟早会因为改了一边忘了另一边而出现两条路径画质不一致。
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

  // ---- 几何关系 ----
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
    // 平方反比 + 窗口函数：在 range 处平滑归零，避免硬截断的亮度突变
    let d2 = dist * dist;
    let win = clamp(1.0 - pow(dist / max(lt.position.w, 1e-3), 4.0), 0.0, 1.0);
    atten = win * win / d2;
  }

  let NdotL = dot(N, toLight);
  if (NdotL <= 0.0) { return vec3<f32>(0.0); }

  // ---- 聚光锥衰减 ----
  if (ltype == TYPE_SPOT) {
    let cosA = dot(-toLight, normalize(lt.direction.xyz));
    let cone = smoothstep(lt.direction.w, lt.cone.x, cosA);
    if (cone <= 0.0) { return vec3<f32>(0.0); }
    atten *= cone;
  }
  if (atten <= 1e-6) { return vec3<f32>(0.0); }

  // ---- 阴影 ----
  // 用片元坐标 + 帧序号做种子：同一像素在帧间取不同旋转角，
  // 噪声被 TAA 摊平，而不是固定成固定图案
  let rot = hash12(px + vec2<f32>(frame.params.w * 1.7));
  var shadow = 1.0;
  let strength = lt.shadowP.y;
  if (strength > 0.0 && lt.params.z > 0.0) {
    if (ltype == TYPE_DIR) {
      shadow = shadowCascade(li, worldPos, viewDepth, rot);
    } else if (ltype == TYPE_POINT) {
      shadow = shadowCube(li, worldPos, toLight, rot);
    } else {
      shadow = shadowSpot(li, worldPos, rot);
    }
    // strength < 1 时把阴影往全亮方向插值，等价于「阴影不纯黑」
    shadow = mix(1.0, shadow, strength);
  }

  let radiance = lt.color.rgb * lt.color.a * atten * shadow;

  // ---- Cook-Torrance ----
  let H = normalize(V + toLight);
  let NdotH = max(dot(N, H), 0.0);
  let VdotH = max(dot(V, H), 0.0);
  let D = distributionGGX(NdotH, rough);
  let G = geometrySmith(NdotV, NdotL, rough);
  let F = fresnelSchlick(VdotH, F0);
  let spec = (D * G * F) / (4.0 * NdotV * NdotL + 1e-4);
  let kd = (vec3<f32>(1.0) - F) * (1.0 - metallic);
  let diffuse = kd * albedo / PI;
  return (diffuse + spec) * radiance * NdotL;
}

/**
 * 全屏延迟着色的公共主体。tiled = true 时只遍历本分块可见的光源，
 * = false 时遍历全部光源 —— 这是 Deferred 与 Tiled Deferred 的唯一差别。
 */
fn shadePixel(frag : VSOut, tiled : bool) -> vec4<f32> {
  let uv = frag.pos.xy * frame.screen.zw;
  let depth = textureLoad(depthTex, vec2<i32>(frag.pos.xy), 0);
  if (depth >= 1.0) {
    let t = clamp(uv.y, 0.0, 1.0);
    let sky = mix(vec3<f32>(0.015, 0.02, 0.04), vec3<f32>(0.09, 0.13, 0.22), t);
    return vec4<f32>(sky, 1.0);
  }

  let am = textureSampleLevel(albedoMetalTex, samp, uv, 0.0);
  let nr = textureSampleLevel(normalRoughTex, samp, uv, 0.0);
  let pe = textureSampleLevel(posEmTex,       samp, uv, 0.0);

  let N      = normalize(nr.xyz);
  let albedo = am.rgb;
  let metallic = am.a;
  let rough  = clamp(nr.a, 0.045, 1.0);
  let worldPos = pe.xyz;

  let V = normalize(frame.camPos.xyz - worldPos);
  let NdotV = max(dot(N, V), 1e-4);
  let viewDepth = dot(worldPos - frame.camPos.xyz, frame.camFwd.xyz);

  let F0 = mix(vec3<f32>(0.04), albedo, metallic);
  var color = vec3<f32>(0.0);

  if (tiled) {
    let tile = tileIndex(frag.pos.xy);
    let stride = u32(frame.gridParams.z);
    let n = min(lightGrid[tile], stride);
    for (var k = 0u; k < n; k = k + 1u) {
      color += shadeLight(lightIndices[tile * stride + k], frag.pos.xy, worldPos,
                          N, V, NdotV, viewDepth, albedo, metallic, rough, F0);
    }
  } else {
    let lightCount = u32(frame.params.x);
    for (var i = 0u; i < lightCount; i = i + 1u) {
      color += shadeLight(i, frag.pos.xy, worldPos,
                          N, V, NdotV, viewDepth, albedo, metallic, rough, F0);
    }
  }

  // 环境项：粗糙的半球环境光近似（乘 AO）+ 自发光
  let ao = textureLoad(aoTex, vec2<i32>(frag.pos.xy), 0).r;
  let hemi = mix(vec3<f32>(0.020, 0.024, 0.032), vec3<f32>(0.075, 0.098, 0.145), N.y * 0.5 + 0.5);
  color += albedo * hemi * ao;
  color += albedo * pe.a * 3.0;

  return vec4<f32>(color, 1.0);
}

/// Deferred：全屏光照遍历**全部**光源（无分块剔除）。
@fragment
fn fs(frag : VSOut) -> @location(0) vec4<f32> {
  return shadePixel(frag, false);
}

/// Tiled Deferred：全屏光照只遍历**本分块**的光源列表。
@fragment
fn fs_tiled(frag : VSOut) -> @location(0) vec4<f32> {
  return shadePixel(frag, true);
}