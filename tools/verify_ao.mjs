// 数值校验：四种 AO 算法是否真的「无遮挡 → 可见度 1、有遮挡 → 明显变暗」。
//
// 着色器能编译、管线能创建，都不代表 AO 的数值是对的。最难发现的一类错误是
// **掠射角自遮挡**：地面近乎平行于视线时，一点点重建误差就会被放大成「朝相机隆起的
// 遮挡体」，整片地面发黑；正视时又完全正常，肉眼极难定位。
//
// 这里用合成深度缓冲（地面 + 一个正对相机的箱子）在 CPU 上重算 AO，做定点数值断言。
// 三处曾经的坑，都已固化成断言：
//   (1) 反投影必须用像素中心 uv，否则结果随采样步数漂移 → 第 5b 节
//   (2) 仰角必须以切平面为基准，否则掠射地面自遮挡     → 第 5 节
//   (3) 衰减距离必须用三维距离，否则随相机朝向变化
//
// 用法： 需经 esbuild 打包（依赖 Math.ts），见 package.json 的 verify:ao

import * as M from "../src/Math.ts";

let failures = 0;
let checks = 0;

function ok(cond, label, detail = "") {
  checks++;
  if (cond) {
    console.log(`  [ok]   ${label}${detail ? "  " + detail : ""}`);
  } else {
    console.log(`  [FAIL] ${label} ${detail}`);
    failures++;
  }
}
function near(a, b, eps = 1e-3) { return Math.abs(a - b) <= eps; }
function section(t, fn) { console.log(`\n=== ${t} ===`); fn(); }

function mul4(m, v) {
  return [
    m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12] * v[3],
    m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13] * v[3],
    m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14] * v[3],
    m[3] * v[0] + m[7] * v[1] + m[11] * v[2] + m[15] * v[3],
  ];
}
function project(m, p) {
  const c = mul4(m, p);
  return [c[0] / c[3], c[1] / c[3], c[2] / c[3]];
}
function norm3(v) {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

// ---------- 相机：位于原点、朝 -Z，故视空间即世界空间（view = I） ----------
const W = 256, H = 256;
const NEAR = 0.1, FAR = 100;
const FOV = (60 * Math.PI) / 180;
const proj = M.mat4Perspective(FOV, 1, NEAR, FAR);
const invProj = M.mat4Invert(proj);
const RADIUS = 0.8;
const ALBEDO_LUM = 0.5; // 测试场景用中灰，供 GTAO 多重散射使用

function rayDir(ndcX, ndcY) {
  const c = mul4(invProj, [ndcX, ndcY, 1, 1]);
  return norm3([c[0] / c[3], c[1] / c[3], c[2] / c[3]]);
}

// ---------- 合成场景：地面 y = -2，地上一个正对相机的箱子 ----------
// 箱子的 +Z 面朝向相机，屏幕上是一块明确的竖直面，根部形成强遮挡。
// （若改用平行于视线的墙，几乎看不出遮挡，测不出东西。）
const BOX_MIN = [1, -2, -14];
const BOX_MAX = [4, 1, -8];

function hitBox(o, d) {
  let tmin = 0, tmax = Infinity, axis = -1, sign = 0;
  for (let a = 0; a < 3; a++) {
    if (Math.abs(d[a]) < 1e-9) {
      if (o[a] < BOX_MIN[a] || o[a] > BOX_MAX[a]) return null;
      continue;
    }
    const inv = 1 / d[a];
    let tn = (BOX_MIN[a] - o[a]) * inv;
    let tf = (BOX_MAX[a] - o[a]) * inv;
    let s = -1; // 从 min 面进入 → 外法线 -1
    if (tn > tf) { const t = tn; tn = tf; tf = t; s = 1; }
    if (tn > tmin) { tmin = tn; axis = a; sign = s; }
    if (tf < tmax) tmax = tf;
    if (tmin > tmax) return null;
  }
  if (axis < 0) return null;
  const n = [0, 0, 0];
  n[axis] = sign;
  return { t: tmin, p: [o[0] + d[0] * tmin, o[1] + d[1] * tmin, o[2] + d[2] * tmin], n };
}

function hitScene(dir) {
  let best = null;
  if (dir[1] < -1e-6) {
    const t = -2 / dir[1];
    const p = [dir[0] * t, dir[1] * t, dir[2] * t];
    if (t > 0 && p[2] < 0) best = { t, p, n: [0, 1, 0] };
  }
  const b = hitBox([0, 0, 0], dir);
  if (b && (!best || b.t < best.t)) best = b;
  return best;
}

const depth = new Float64Array(W * H).fill(1);
const nrm = new Float64Array(W * H * 3);
for (let j = 0; j < H; j++) {
  for (let i = 0; i < W; i++) {
    const uv = [(i + 0.5) / W, (j + 0.5) / H];
    const dir = rayDir(uv[0] * 2 - 1, 1 - uv[1] * 2);
    const h = hitScene(dir);
    const k = j * W + i;
    if (!h) continue;
    depth[k] = project(proj, [h.p[0], h.p[1], h.p[2], 1])[2];
    nrm[k * 3 + 0] = h.n[0];
    nrm[k * 3 + 1] = h.n[1];
    nrm[k * 3 + 2] = h.n[2];
  }
}

// ---------- 着色器函数的 CPU 移植 ----------
// 采样噪声全部换成确定的低差异序列，把积分结构从随机数里隔离出来。
function viewPosFromDepth(uv, d) {
  const c = mul4(invProj, [uv[0] * 2 - 1, 1 - uv[1] * 2, d, 1]);
  return [c[0] / c[3], c[1] / c[3], c[2] / c[3]];
}
function uvFromViewPos(p) {
  const c = mul4(proj, [p[0], p[1], p[2], 1]);
  return [(c[0] / c[3]) * 0.5 + 0.5, 0.5 - 0.5 * (c[1] / c[3])];
}
// 与 WGSL 的 vec2<i32>(uv * dim) 对齐：截断而非四舍五入
function pixelOf(uv) {
  const i = Math.min(W - 1, Math.max(0, Math.floor(uv[0] * W)));
  const j = Math.min(H - 1, Math.max(0, Math.floor(uv[1] * H)));
  return [i, j];
}
function depthAtUV(uv) {
  const [i, j] = pixelOf(uv);
  return depth[j * W + i];
}
function normalAtUV(uv) {
  const [i, j] = pixelOf(uv);
  const k = j * W + i;
  return [nrm[k * 3], nrm[k * 3 + 1], nrm[k * 3 + 2]];
}
/**
 * 采样该 uv 处几何体的视空间位置，天空返回 null。
 *
 * 关键：先吸附到像素中心，再用**吸附后的** uv 反投影。
 * 深度值属于「像素中心那条视线」，若用精确的行进 uv 去反投影，得到的点不在表面上，
 * 误差 = 像素偏移 × 深度梯度。掠射角下深度梯度极大 → 假遮挡 → 地面发黑，
 * 且结果会随采样步数漂移。对应 Ao.wgsl 的 sampleSurface()。
 */
function sampleSurface(uv) {
  const [i, j] = pixelOf(uv);
  const d = depth[j * W + i];
  if (d >= 1) return null;
  return viewPosFromDepth([(i + 0.5) / W, (j + 0.5) / H], d);
}
function smoothstep(x) {
  const t = Math.min(1, Math.max(0, x));
  return t * t * (3 - 2 * t);
}

const SSAO_SAMPLES = 16, HBAO_DIRS = 8, HBAO_STEPS = 6, GTAO_SLICES = 4, GTAO_STEPS = 8;

function ssao(P, N, radius = RADIUS) {
  let occ = 0;
  for (let i = 0; i < SSAO_SAMPLES; i++) {
    const u0 = (i + 0.5) / SSAO_SAMPLES;
    const u1 = ((i * 7) % SSAO_SAMPLES + 0.5) / SSAO_SAMPLES;
    const r = Math.sqrt(u0), phi = 2 * Math.PI * u1;
    const t = [r * Math.cos(phi), r * Math.sin(phi), Math.sqrt(Math.max(1 - u0, 0))];
    const up = Math.abs(N[1]) > 0.95 ? [1, 0, 0] : [0, 1, 0];
    const xt = norm3([
      up[1] * N[2] - up[2] * N[1], up[2] * N[0] - up[0] * N[2], up[0] * N[1] - up[1] * N[0],
    ]);
    const yt = [
      N[1] * xt[2] - N[2] * xt[1], N[2] * xt[0] - N[0] * xt[2], N[0] * xt[1] - N[1] * xt[0],
    ];
    const dir = norm3([
      xt[0] * t[0] + yt[0] * t[1] + N[0] * t[2],
      xt[1] * t[0] + yt[1] * t[1] + N[1] * t[2],
      xt[2] * t[0] + yt[2] * t[1] + N[2] * t[2],
    ]);
    const s = radius * (0.15 + 0.85 * Math.sqrt((i + 0.5) / SSAO_SAMPLES));
    const sp = [P[0] + dir[0] * s, P[1] + dir[1] * s, P[2] + dir[2] * s];
    const suv = uvFromViewPos(sp);
    if (suv[0] < 0 || suv[0] > 1 || suv[1] < 0 || suv[1] > 1) continue;
    const S = sampleSurface(suv);
    if (!S) continue;
    const buried = S[2] - sp[2];
    const diff = P[2] - S[2];
    if (buried > 0.02) occ += smoothstep(radius / Math.max(Math.abs(diff), 1e-4));
  }
  return 1 - occ / SSAO_SAMPLES;
}

/**
 * HBAO / GTAO 共用的地平线搜索：返回该方位角上的遮挡量 ∈ [0,1]。
 *
 * 弧段增量累加，而不是取「sin^w(elev)×atten 最大的样本」：
 * 单样本只用到一个距离处的衰减，会把「近处低仰角 + 远处高仰角」压成一个值，
 * 系统性偏低。weightPow = 1 → ΣΔsin（HBAO），= 2 → ΣΔsin²（GTAO）。
 * 衰减按三维距离 |S−P| 算，旋转不变。
 */
function horizonOcclusion(P, N, dir2, radius, steps, falloffPow, weightPow) {
  let occ = 0;
  let hPrev = 0;
  for (let s = 1; s <= steps; s++) {
    const t = s / steps;
    const sp = [P[0] + dir2[0] * radius * t, P[1] + dir2[1] * radius * t, P[2]];
    const suv = uvFromViewPos(sp);
    if (suv[0] < 0 || suv[0] > 1 || suv[1] < 0 || suv[1] > 1) continue;
    const S = sampleSurface(suv);
    if (!S) continue;
    const d = [S[0] - P[0], S[1] - P[1], S[2] - P[2]];
    if (d[0] * dir2[0] + d[1] * dir2[1] <= 0) continue;
    const len = Math.hypot(d[0], d[1], d[2]);
    if (len < 1e-4) continue;
    const sinE = Math.max(
      Math.sin(Math.asin(Math.min(1, Math.max(-1, (d[0] * N[0] + d[1] * N[1] + d[2] * N[2]) / len)))),
      0
    );
    const q = Math.min(1, Math.max(0, len / radius));
    const atten = Math.pow(1 - q * q, falloffPow);
    if (sinE > hPrev) {
      occ += (Math.pow(sinE, weightPow) - Math.pow(hPrev, weightPow)) * atten;
      hPrev = sinE;
    }
  }
  return Math.min(1, Math.max(0, occ));
}

function hbaoCore(P, N, radius, falloffPow, steps = HBAO_STEPS) {
  let occ = 0;
  for (let d = 0; d < HBAO_DIRS; d++) {
    const theta = ((d + 0.5) / HBAO_DIRS) * 2 * Math.PI;
    const dir2 = [Math.cos(theta), Math.sin(theta)];
    occ += horizonOcclusion(P, N, dir2, radius, steps, falloffPow, 1.0);
  }
  return 1 - occ / HBAO_DIRS;
}

/** GTAO：逐切片两侧求地平线，弧段权重 Δsin²，最后套多重散射。 */
function gtaoSingle(P, N, radius = RADIUS, steps = GTAO_STEPS, slices = GTAO_SLICES) {
  let occ = 0;
  for (let s = 0; s < slices; s++) {
    const theta = ((s + 0.5) / slices) * Math.PI;
    const dir2 = [Math.cos(theta), Math.sin(theta)];
    const a = horizonOcclusion(P, N, dir2, radius, steps, 1.0, 2.0);
    const b = horizonOcclusion(P, N, [-dir2[0], -dir2[1]], radius, steps, 1.0, 2.0);
    occ += 0.5 * (a + b);
  }
  return 1 - occ / slices;
}
/** 多重散射：多项式作用在「可见度」上 */
function multiBounce(vis, lum = ALBEDO_LUM) {
  const a = 2.0404 * lum - 0.3324;
  const b = -4.7951 * lum + 0.6417;
  const c = 2.7552 * lum + 0.6903;
  return Math.min(1, Math.max(0, ((vis * a + b) * vis + c) * vis));
}
function gtao(P, N, radius = RADIUS) {
  return multiBounce(gtaoSingle(P, N, radius));
}

const METHODS = {
  ssao: (P, N) => ssao(P, N),
  hbao: (P, N) => hbaoCore(P, N, RADIUS, 1.0),
  hbaoplus: (P, N) => {
    const near = hbaoCore(P, N, RADIUS, 2.0);
    const far = hbaoCore(P, N, RADIUS * 3, 2.0);
    return 1 - ((1 - near) * 0.55 + (1 - far) * 0.45);
  },
  gtao: (P, N) => gtao(P, N),
};

function probe(target) {
  const uv = uvFromViewPos(target);
  const [i, j] = pixelOf(uv);
  const puv = [(i + 0.5) / W, (j + 0.5) / H];
  const d = depth[j * W + i];
  return { uv: puv, P: viewPosFromDepth(puv, d), N: normalAtUV(puv), d };
}

// =====================================================================
section("1. 深度 ↔ 视空间往返自洽", () => {
  for (const t of [[0, -2, -8], [-3, -2, -12], [2, -2, -20], [0, 3, -15]]) {
    const uv = uvFromViewPos(t);
    const d = project(proj, [t[0], t[1], t[2], 1])[2];
    const back = viewPosFromDepth(uv, d);
    const err = Math.hypot(back[0] - t[0], back[1] - t[1], back[2] - t[2]);
    ok(err < 1e-3, `往返误差 < 1e-3 @ [${t}]`, `err=${err.toExponential(2)}`);
  }
});

section("2. GTAO 权重：解析弧积分 = 数值积分", () => {
  // 切平面坐标系下，遮挡弧 [0,h] 的余弦加权积分应为 ½sin²h
  // （∫sinα cosα dα = ½sin²α；整个四分之一圆为 ½）
  for (const deg of [10, 30, 45, 60, 85]) {
    const h = (deg * Math.PI) / 180;
    const N = 20000;
    let num = 0;
    for (let i = 0; i < N; i++) {
      const a = ((i + 0.5) / N) * h;
      num += Math.sin(a) * Math.cos(a) * (h / N);
    }
    const analytic = 0.5 * Math.sin(h) * Math.sin(h);
    ok(near(num, analytic, 1e-5), `h=${deg}°: 数值积分 = ½sin²h`, `${num.toFixed(6)} vs ${analytic.toFixed(6)}`);
  }
  // 完全被挡（h = 90°）应等于整个四分之一圆的积分 ½ → 归一化后该侧 occ = 1
  {
    const N = 20000, h = Math.PI / 2;
    let num = 0;
    for (let i = 0; i < N; i++) {
      const a = ((i + 0.5) / N) * h;
      num += Math.sin(a) * Math.cos(a) * (h / N);
    }
    ok(near(num, 0.5, 1e-5), "h=90°: 积分 = ½（即该侧完全被挡）", `got ${num.toFixed(6)}`);
  }
  // 弧段增量形式必须与整段解析式一致：ΣΔsin² 在无衰减时应等于 sin²h
  {
    const h = (67.4 * Math.PI) / 180, steps = 64;
    let acc = 0, prev = 0;
    for (let s = 1; s <= steps; s++) {
      const e = h * (s / steps);
      acc += Math.sin(e) ** 2 - prev ** 2;
      prev = Math.sin(e);
    }
    ok(near(acc, Math.sin(h) ** 2, 1e-6), "ΣΔsin² 累加 = sin²h", `${acc.toFixed(6)} vs ${Math.sin(h) ** 2}`);
  }
});

section("3. GTAO 多重散射：多项式必须作用在可见度上", () => {
  const f = (v, lum) => {
    const a = 2.0404 * lum - 0.3324;
    const b = -4.7951 * lum + 0.6417;
    const c = 2.7552 * lum + 0.6903;
    return ((v * a + b) * v + c) * v;
  };
  for (const lum of [0, 0.25, 0.5, 0.75, 1]) {
    ok(near(f(0, lum), 0, 1e-9), `lum=${lum}: f(0)=0`);
    ok(near(f(1, lum), 1, 5e-3), `lum=${lum}: f(1)≈1`, `got ${f(1, lum).toFixed(5)}`);
  }
  // 物理方向：亮表面靠多次反弹拿到补光 → 可见度**升高**；暗表面 → 降低
  ok(f(0.5, 1.0) > 0.55, "白表面：多次散射提高可见度", `f(0.5)=${f(0.5, 1).toFixed(4)}`);
  ok(f(0.5, 0.0) < 0.49, "黑表面：多次散射降低可见度", `f(0.5)=${f(0.5, 0).toFixed(4)}`);
});

section("4. 开阔地面（无任何遮挡）不得自遮挡", () => {
  const open = probe([-4, -2, -12]);
  console.log(`       探测点 P=[${open.P.map((v) => v.toFixed(2))}] N=[${open.N}]`);
  for (const [name, fn] of Object.entries(METHODS)) {
    const v = fn(open.P, open.N);
    ok(v >= 0.9, `${name}: 开阔地面可见度 ≥ 0.90`, `got ${v.toFixed(4)}`);
  }
});

section("5. 掠射角地面（最容易出现假遮挡的情形）", () => {
  for (const z of [-25, -40]) {
    const g = probe([-4, -2, z]);
    const V = norm3([-g.P[0], -g.P[1], -g.P[2]]);
    const cos = g.N[0] * V[0] + g.N[1] * V[1] + g.N[2] * V[2];
    console.log(`       z=${z}: N·V = ${cos.toFixed(4)}（越小越掠射）`);
    for (const [name, fn] of Object.entries(METHODS)) {
      const v = fn(g.P, g.N);
      ok(v >= 0.85, `${name} @ z=${z}: 掠射地面可见度 ≥ 0.85`, `got ${v.toFixed(4)}`);
    }
  }
});

section("5b. 结果必须与采样步数无关（防「像素吸附」回归）", () => {
  // 若反投影用了精确行进 uv 而非像素中心 uv，采样点越多越容易撞上像素边界，
  // 结果会随步数明显漂移。这是「掠射角发黑」那个 bug 最好用的照妖镜。
  const g = probe([-4, -2, -40]);
  const open = probe([-4, -2, -12]);
  for (const [name, steps] of [["hbao", HBAO_STEPS], ["gtao", GTAO_STEPS]]) {
    const fn = (P, N, s) =>
      name === "hbao"
        ? hbaoCore(P, N, RADIUS, 1.0, s)
        : gtaoSingle(P, N, RADIUS, s);
    for (const [tag, p] of [["掠射 z=-40", g], ["开阔 z=-12", open]]) {
      const v2 = fn(p.P, p.N, steps * 2);
      const v4 = fn(p.P, p.N, steps * 4);
      const drift = Math.max(Math.abs(v2 - v4), 0);
      ok(drift < 0.02, `${name} @ ${tag}: 步数 ${steps * 2} 与 ${steps * 4} 结果一致`, `漂移 ${drift.toFixed(4)}`);
    }
  }
});

section("6. 箱子根部必须明显变暗", () => {
  // 箱子 +Z 面在 z=-8，探测点在其前方 0.15，箱子高出地面 3
  const corner = probe([2.5, -2, -7.85]);
  const open = probe([-4, -2, -12]);
  console.log(`       箱根 P=[${corner.P.map((v) => v.toFixed(2))}]`);
  for (const [name, fn] of Object.entries(METHODS)) {
    const vc = fn(corner.P, corner.N);
    const vo = fn(open.P, open.N);
    ok(vc <= 0.9, `${name}: 箱根可见度 ≤ 0.90`, `got ${vc.toFixed(4)}`);
    ok(vc < vo - 0.1, `${name}: 箱根比开阔处暗 ≥ 0.1`, `${vc.toFixed(4)} vs ${vo.toFixed(4)}`);
  }
  // GTAO 在箱根处明显比其余三者亮，这是两个**有意为之**的设计叠加的结果：
  //   ① 余弦加权（弧段权重 sin² 而非 HBAO 的 sin）——低仰角遮挡体贡献本就该小得多；
  //   ② 多重散射补偿——亮表面的反弹补光会抬高可见度。
  // 把这两条显式断言下来，避免日后被误判成「GTAO 偏亮 = bug」。
  const gSingle = gtaoSingle(corner.P, corner.N);
  const gComp = gtao(corner.P, corner.N);
  ok(gSingle <= 0.9, "gtao: 单次散射箱根 ≤ 0.90（与其余三者同口径可比）", `got ${gSingle.toFixed(4)}`);
  ok(gComp >= gSingle, "gtao: 多重散射只能抬高可见度", `${gSingle.toFixed(4)} → ${gComp.toFixed(4)}`);
});

section("7. 探测点本身有效", () => {
  const targets = [[-4, -2, -12], [2.5, -2, -7.85], [-4, -2, -25], [-4, -2, -40]];
  for (const t of targets) {
    const p = probe(t);
    // 掠射角下一个像素要覆盖很大的深度范围，误差按距离的百分比衡量
    const err = Math.hypot(p.P[0] - t[0], p.P[1] - t[1], p.P[2] - t[2]);
    ok(p.d < 1 && err < 0.06 * Math.abs(t[2]), `探测 [${t}] 命中几何体`, `err=${err.toFixed(3)}`);
  }
});

console.log(`\n=== 结论: ${checks - failures}/${checks} 通过, ${failures} 失败 ===`);
process.exit(failures > 0 ? 1 : 0);
