// 屏幕空间环境光遮蔽（AO）。
//
// 四种实现各自独立成入口点，运行时通过选择管线切换：
//   fs_ssao     — 经典 Crytek 半球采样：法线半球内随机取点，比较深度求遮挡比例
//   fs_hbao     — Horizon-Based：沿水平方位角行进求地平线仰角，对方位角积分
//   fs_hbaoplus — HBAO 的双尺度改进版：近景细节 + 远景大尺度，配合平方衰减
//   fs_gtao     — Ground Truth：逐切片求地平线角，余弦加权圆弧积分解析求可见度
//
// 输出单通道 r16float，存的是「可见度」（1 = 完全无遮挡），
// 光照阶段直接乘在环境项上。
//
// ---------------------------------------------------------------------------
// 三个容易写错、且错了以后「正视看不出、掠射一片黑」的点
// ---------------------------------------------------------------------------
//
// 1) 反投影必须用**像素中心 uv**，不能用精确的行进 uv。
//    深度值属于「像素中心那条视线」；拿别的 uv 去反投影，得到的点根本不在几何体
//    表面上，误差 = 像素偏移 × 深度梯度。掠射角下深度梯度极大，平坦地面会被重建出
//    巨大高度差，于是「自己挡住自己」，整片地面发黑；正视时又完全正常。
//    这个错误还会让结果强烈依赖采样步数——步数越多，撞上像素边界的机会越多。
//    正确做法：先把 uv 吸附到像素中心（snapUV），再用吸附后的 uv 反投影。
//    这样得到的点严格落在该像素所见的表面上，任何视角、任何步数都不会自遮挡。
//
// 2) 衰减距离要用**三维距离 |S−P|**，不能用 length(S.xy − P.xy)。
//    后者把仰角方向的抬升也算成了「水平距离」：紧贴墙壁的高墙明明是最近的遮挡体，
//    却因为抬升大而被判成「离得远」衰减掉。而且它随相机朝向变化——同一场景转个相机
//    遮挡量就变了，这是明显的缺陷。三维距离旋转不变，与「半径球」的语义一致。
//
// 3) 地平线积分要**按弧段增量累加**，不能只取「sin^w(elev)×atten 最大的单个样本」。
//    单个样本只用到某一个距离处的衰减，会把「近处低仰角 + 远处高仰角」的连续遮挡
//    压成一个值，系统性偏低。正解是把行进过程中「新露出来的那段弧」按它自己那处的
//    距离衰减累加：
//        HBAO: a = Σ (sin h_i − sin h_prev) × atten_i        （∫cos α dα = sin α）
//        GTAO: a = Σ (sin²h_i − sin²h_prev) × atten_i       （∫sin α cos α dα = ½sin²α）
//    两者在 h→π/2 时都归一化到 1，即该方位角被完全遮挡。
//
// 关于仰角的基准：HBAO 与 GTAO 都以**切平面**为基准量仰角
// （elev = asin(dot(S−P, N)/|S−P|)），而不是相对视平面。
// 相对视平面的量法（atan2(S.z−P.z, 水平距离)）在掠射角下同样会失效：地面近乎平行于
// 视线，沿视平面行进时重建出的地面点 z 变化极大，会被误判成「朝相机隆起的遮挡体」；
// 同时还会漏掉位于受光点后方的遮挡体。以切平面为基准，平面上的样本恒为 0。

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
  screen        : vec4<f32>,  // w h 1/w 1/h
  aoParams      : vec4<f32>,  // x=radius y=intensity z=power w=noiseSeed
  misc          : vec4<f32>,  // x=renderPath y=aoMethod z=tileSizeX w=tileSizeY
  gridParams    : vec4<f32>,  // x=tilesX y=tilesY z=lightsPerTile w=pad
};

@group(0) @binding(0) var<uniform> frame : Frame;
@group(1) @binding(0) var depthTex  : texture_depth_2d;
@group(1) @binding(1) var normalTex : texture_2d<f32>;
@group(1) @binding(2) var albedoTex : texture_2d<f32>;

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

const SSAO_SAMPLES = 16u;
const HBAO_DIRS = 8u;
const HBAO_STEPS = 6u;
const GTAO_SLICES = 4u;
const GTAO_STEPS = 8u;

fn hash12(p : vec2<f32>) -> f32 {
  var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

// ---------- 视空间重建 ----------
// 视空间约定：相机在原点，朝 -Z 看，因此「线性深度」= -z（正值，越大越远）。

/** 把 uv 吸附到像素中心。见文件头第 (1) 条。 */
fn snapUV(uv : vec2<f32>) -> vec2<f32> {
  let dim = vec2<i32>(textureDimensions(depthTex));
  let c = clamp(vec2<i32>(uv * vec2<f32>(dim)), vec2<i32>(0), dim - vec2<i32>(1));
  return (vec2<f32>(c) + vec2<f32>(0.5)) / vec2<f32>(dim);
}

fn viewPosFromDepth(uv : vec2<f32>, depth : f32) -> vec3<f32> {
  let ndc = vec3<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth);
  let clip = frame.invProj * vec4<f32>(ndc, 1.0);
  return clip.xyz / clip.w;
}

fn uvFromViewPos(p : vec3<f32>) -> vec2<f32> {
  let clip = frame.proj * vec4<f32>(p, 1.0);
  let ndc = clip.xy / clip.w;
  return vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

fn viewNormal(nw : vec3<f32>) -> vec3<f32> {
  return normalize((frame.view * vec4<f32>(nw, 0.0)).xyz);
}

/** occ ∈ [0,1] 为遮挡量，转存为可见度并套用强度/幂次。 */
fn finalize(occ : f32) -> vec4<f32> {
  let intensity = frame.aoParams.y;
  let power = frame.aoParams.z;
  let vis = clamp(1.0 - clamp(occ, 0.0, 1.0) * intensity, 0.0, 1.0);
  return vec4<f32>(pow(vis, max(power, 0.01)), 0.0, 0.0, 1.0);
}

/**
 * 采样该 uv 处几何体的视空间位置。
 * 先吸附到像素中心，再用**吸附后的** uv 反投影——这样得到的点严格落在
 * 该像素所见的表面上。w = 0 表示天空（该像素无几何体）。
 */
fn sampleSurface(uv : vec2<f32>) -> vec4<f32> {
  let dim = vec2<i32>(textureDimensions(depthTex));
  let c = clamp(vec2<i32>(uv * vec2<f32>(dim)), vec2<i32>(0), dim - vec2<i32>(1));
  let d = textureLoad(depthTex, c, 0);
  if (d >= 1.0) { return vec4<f32>(0.0); }
  let puv = (vec2<f32>(c) + vec2<f32>(0.5)) / vec2<f32>(dim);
  return vec4<f32>(viewPosFromDepth(puv, d), 1.0);
}

/** 读取当前片元的视空间位置 / 视空间法线，天空返回 w=0。 */
fn surfaceAt(coord : vec2<i32>, uv : vec2<f32>) -> vec4<f32> {
  let d0 = textureLoad(depthTex, coord, 0);
  if (d0 >= 1.0) { return vec4<f32>(0.0); }
  return vec4<f32>(viewPosFromDepth(uv, d0), 1.0);
}

fn surfaceNormalAt(coord : vec2<i32>) -> vec3<f32> {
  return viewNormal(normalize(textureLoad(normalTex, coord, 0).xyz));
}

// ================= SSAO =================

/** 以法线为轴的余弦加权半球随机方向。 */
fn hemisphereDir(n : vec3<f32>, seed : vec2<f32>, i : u32) -> vec3<f32> {
  let u0 = hash12(seed + vec2<f32>(f32(i) * 1.37, 0.71));
  let u1 = hash12(seed + vec2<f32>(0.29, f32(i) * 2.11));
  let r = sqrt(u0);
  let phi = 2.0 * PI * u1;
  let t = vec3<f32>(r * cos(phi), r * sin(phi), sqrt(max(1.0 - u0, 0.0)));
  let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.95);
  let xt = normalize(cross(up, n));
  let yt = cross(n, xt);
  return normalize(xt * t.x + yt * t.y + n * t.z);
}

@fragment
fn fs_ssao(frag : VSOut) -> @location(0) vec4<f32> {
  let coord = vec2<i32>(frag.pos.xy);
  let uv = frag.pos.xy * frame.screen.zw;
  let s0 = surfaceAt(coord, uv);
  if (s0.w < 0.5) { return vec4<f32>(1.0, 0.0, 0.0, 1.0); }

  let P = s0.xyz;
  let Nv = surfaceNormalAt(coord);
  let radius = frame.aoParams.x;
  let seed = frag.pos.xy + vec2<f32>(frame.aoParams.w * 13.7);

  var occ = 0.0;
  for (var i = 0u; i < SSAO_SAMPLES; i = i + 1u) {
    // 半径按 sqrt 分布，让采样点在近处更密（抵消平方反比的稀疏感）
    let t = (f32(i) + 0.5) / f32(SSAO_SAMPLES);
    let dir = hemisphereDir(Nv, seed, i);
    let sp = P + dir * (radius * mix(0.15, 1.0, sqrt(t)));

    let suv = uvFromViewPos(sp);
    if (any(suv < vec2<f32>(0.0)) || any(suv > vec2<f32>(1.0))) { continue; }
    let sw = sampleSurface(suv);
    if (sw.w < 0.5) { continue; }
    let S = sw.xyz;

    // S.z > sp.z 表示场景几何比采样点更靠近相机 → 采样点被埋在几何体内
    let buried = S.z - sp.z;
    let diff = P.z - S.z;
    if (buried > 0.02) {
      // 距离衰减用「半径/深度差」的软形式而非硬截断：
      // 硬截断会让刚超出半径的遮挡体被整块丢弃，在半径边界上留下可见接缝
      occ += smoothstep(0.0, 1.0, radius / max(abs(diff), 1e-4));
    }
  }
  return finalize(occ / f32(SSAO_SAMPLES));
}

// ================= HBAO / HBAO+ / GTAO 共用的地平线搜索 =================

/**
 * 沿某个水平方位角行进，返回该方位角上的遮挡量 ∈ [0,1]。
 *
 * 仰角以**切平面**为基准：elev = asin(dot(S−P, N)/|S−P|)。
 * 取的是弧段累加而不是「仰角最大的样本」——见文件头第 (3) 条。
 * weightPow = 1 → HBAO 的 ΣΔsin；weightPow = 2 → GTAO 的 ΣΔsin²。
 */
fn horizonOcclusion(
  P : vec3<f32>,
  Nv : vec3<f32>,
  dir2 : vec2<f32>,
  radius : f32,
  steps : u32,
  jitter : f32,
  falloffPow : f32,
  weightPow : f32
) -> f32 {
  var occ = 0.0;
  var hPrev = 0.0;   // 迄今见过的最高仰角的 sin 值
  for (var s = 1u; s <= steps; s = s + 1u) {
    let t = (f32(s) + jitter) / f32(steps);
    let sp = P + vec3<f32>(dir2.x, dir2.y, 0.0) * (radius * t);
    let suv = uvFromViewPos(sp);
    if (any(suv < vec2<f32>(0.0)) || any(suv > vec2<f32>(1.0))) { continue; }
    let sw = sampleSurface(suv);
    if (sw.w < 0.5) { continue; }
    let delta = sw.xyz - P;
    // 只认位于该方位角正方向上的遮挡体
    if (dot(delta, vec3<f32>(dir2.x, dir2.y, 0.0)) <= 0.0) { continue; }
    let len = length(delta);
    if (len < 1e-4) { continue; }

    let sinE = max(sin(asin(clamp(dot(delta, Nv) / len, -1.0, 1.0))), 0.0);
    // 衰减按三维距离算（见文件头第 (2) 条），并让每段弧带上自己那处的衰减
    let q = clamp(len / radius, 0.0, 1.0);
    let atten = pow(1.0 - q * q, falloffPow);

    if (sinE > hPrev) {
      occ += (pow(sinE, weightPow) - pow(hPrev, weightPow)) * atten;
      hPrev = sinE;
    }
  }
  return clamp(occ, 0.0, 1.0);
}

// ================= HBAO / HBAO+ =================

/**
 * 对 HBAO_DIRS 个方位角积分。每个方位角的贡献取经典的 sin(h) 权重，
 * 即「切平面以上 [0,h] 这段弧被挡住」，h = π/2 时该方位角完全被挡。
 */
fn hbaoCore(
  P : vec3<f32>,
  Nv : vec3<f32>,
  radius : f32,
  seed : f32,
  falloffPow : f32
) -> f32 {
  var occ = 0.0;
  for (var d = 0u; d < HBAO_DIRS; d = d + 1u) {
    let jitter = hash12(vec2<f32>(seed, f32(d) * 1.7));
    let theta = (f32(d) + jitter) / f32(HBAO_DIRS) * 2.0 * PI;
    let dir2 = vec2<f32>(cos(theta), sin(theta));
    occ += horizonOcclusion(P, Nv, dir2, radius, HBAO_STEPS, jitter - 0.5, falloffPow, 1.0);
  }
  return occ / f32(HBAO_DIRS);
}

@fragment
fn fs_hbao(frag : VSOut) -> @location(0) vec4<f32> {
  let coord = vec2<i32>(frag.pos.xy);
  let uv = frag.pos.xy * frame.screen.zw;
  let s0 = surfaceAt(coord, uv);
  if (s0.w < 0.5) { return vec4<f32>(1.0, 0.0, 0.0, 1.0); }

  let P = s0.xyz;
  let Nv = surfaceNormalAt(coord);
  let seed = hash12(frag.pos.xy + vec2<f32>(frame.aoParams.w * 7.3)) * 64.0;

  let occ = hbaoCore(P, Nv, frame.aoParams.x, seed, 1.0);
  return finalize(occ);
}

@fragment
fn fs_hbaoplus(frag : VSOut) -> @location(0) vec4<f32> {
  let coord = vec2<i32>(frag.pos.xy);
  let uv = frag.pos.xy * frame.screen.zw;
  let s0 = surfaceAt(coord, uv);
  if (s0.w < 0.5) { return vec4<f32>(1.0, 0.0, 0.0, 1.0); }

  let P = s0.xyz;
  let Nv = surfaceNormalAt(coord);
  let seed = hash12(frag.pos.xy + vec2<f32>(frame.aoParams.w * 7.3)) * 64.0;
  let r = frame.aoParams.x;

  // 双尺度：近景半径抓细节，3 倍半径抓大尺度遮蔽，平方衰减让过渡更干净
  let near = hbaoCore(P, Nv, r, seed, 2.0);
  let far = hbaoCore(P, Nv, r * 3.0, seed + 17.0, 2.0);
  return finalize(near * 0.55 + far * 0.45);
}

// ================= GTAO =================

@fragment
fn fs_gtao(frag : VSOut) -> @location(0) vec4<f32> {
  let coord = vec2<i32>(frag.pos.xy);
  let uv = frag.pos.xy * frame.screen.zw;
  let s0 = surfaceAt(coord, uv);
  if (s0.w < 0.5) { return vec4<f32>(1.0, 0.0, 0.0, 1.0); }

  let P = s0.xyz;
  let Nv = surfaceNormalAt(coord);
  let radius = frame.aoParams.x;
  let seed = frag.pos.xy + vec2<f32>(frame.aoParams.w * 5.1);
  let jitter = hash12(seed) - 0.5;

  // 切片只需扫 [0, π)：θ 与 θ+π 是同一条直线，一个切片同时覆盖两侧。
  // 与 HBAO 的差别在权重：HBAO 用经典的 sin(h)，GTAO 用切平面坐标系下
  // 真正的余弦加权弧积分——∫sin α cos α dα = ½sin²α，故弧段增量为 Δsin²。
  var occ = 0.0;
  for (var s = 0u; s < GTAO_SLICES; s = s + 1u) {
    let theta = (f32(s) + hash12(seed + vec2<f32>(f32(s) * 3.1, 0.0))) / f32(GTAO_SLICES) * PI;
    let dir2 = vec2<f32>(cos(theta), sin(theta));
    let a = horizonOcclusion(P, Nv,  dir2, radius, GTAO_STEPS, jitter, 1.0, 2.0);
    let b = horizonOcclusion(P, Nv, -dir2, radius, GTAO_STEPS, jitter, 1.0, 2.0);
    occ += 0.5 * (a + b);
  }
  occ = occ / f32(GTAO_SLICES);

  // 多重散射近似（Jimenez 2016）。注意多项式作用在**可见度**上，不是遮挡量：
  // 亮的表面靠多次反弹拿到更多补光 → 可见度升高；暗的表面把反弹光吸收掉 → 更低。
  // 套反了会让白色物体在凹处变得更黑，正好与物理相反。
  var vis = clamp(1.0 - occ, 0.0, 1.0);
  let alb = textureLoad(albedoTex, coord, 0).rgb;
  let lum = clamp(dot(alb, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
  let ka = 2.0404 * lum - 0.3324;
  let kb = -4.7951 * lum + 0.6417;
  let kc = 2.7552 * lum + 0.6903;
  vis = clamp(((vis * ka + kb) * vis + kc) * vis, 0.0, 1.0);

  return finalize(1.0 - vis);
}

// 关闭 AO：直接写全可见
@fragment
fn fs_off(frag : VSOut) -> @location(0) vec4<f32> {
  return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
