// CSM 结构与 CPU/GPU 一致性校验。
//
// 为什么需要它：级联选择一旦 off-by-one，后果是**静默的**——着色器发现点落在
// 本级联正交框外时直接 `return 1.0`（判为受光），于是那片阴影不报错、不闪烁，
// 就是「没了」。这种 bug 肉眼只会看到「某些物体没影子」，很难联想到是选择逻辑差一位。
//
// 本脚本用**不变量**把它钉住：
//   ① splits[c] 是级联 c 的**远界**，bounds[c] = [splits[c-1], splits[c]]
//   ② 视深度 v 必须选中「v ∈ bounds[c]」的那个 c，且每级联都有非空的可用区间
//   ③ 级联矩阵必须把自己的视锥切片 8 个角点全部映射进单位立方体
//   ④ WGSL 的选级联循环必须与 Math.selectCascade 对拍一致（CPU/GPU 同语义）
//   ⑤ 混合带必须落在本级联自己的区间内
//   ⑥ 半影比例补偿：用共享 zFar 线性化再乘 k，必须等于用本级联 zFar 线性化
//
// 用法： 需经 esbuild 打包（依赖 Math.ts），见 package.json 的 verify:csm

import * as M from "../src/Math.ts";
import { readFileSync } from "node:fs";
import { join } from "node:path";

// 注意：esbuild 打包后 import.meta.url 指向 node_modules/.cache，
// 所以源码路径一律基于 npm 的工作目录（包根）解析。
const ROOT = process.cwd();

let failures = 0;
let checks = 0;
function ok(cond, label, detail = "") {
  checks++;
  console.log(`  ${cond ? "[ok]  " : "[FAIL]"} ${label}${detail ? "  " + detail : ""}`);
  if (!cond) failures++;
}
function section(t, fn) { console.log(`\n=== ${t} ===`); fn(); }
const f = (v, n = 3) => v.toFixed(n);

// ---------- 与 Renderer 一致的场景参数（demo 相机 + 画布比例） ----------
const CAM_NEAR = 0.1, CAM_FAR = 100, CASCADES = 4, LAMBDA = 0.7;
const FOV = (50 * Math.PI) / 180;
const ASPECTS = [1.0, 16 / 9, 867 / 415, 21 / 9];

/** 与 Renderer 的 sliceCenter 一致：视锥切片的包围球。 */
function sliceCenter(eye, fwd, aspect, n, f) {
  const tanHalf = Math.tan(FOV * 0.5);
  const mid = (n + f) * 0.5;
  const halfH = mid * tanHalf;
  const halfW = halfH * aspect;
  const r = Math.hypot(halfW, halfH, (f - n) * 0.5);
  return {
    c: [eye[0] + fwd[0] * mid, eye[1] + fwd[1] * mid, eye[2] + fwd[2] * mid],
    r, mid, halfW, halfH,
  };
}

const mul4 = (m, v) => [
  m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12] * v[3],
  m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13] * v[3],
  m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14] * v[3],
  m[3] * v[0] + m[7] * v[1] + m[11] * v[2] + m[15] * v[3],
];
const norm = (v) => { const l = Math.hypot(...v) || 1; return [v[0] / l, v[1] / l, v[2] / l]; };
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

const SUN_DIR = norm([0.55, -1.0, 0.4]);   // Main.ts 里平行光的 direction

section("1. splits / bounds 约定自洽", () => {
  for (const aspect of ASPECTS) {
    const { bounds } = M.cascadeBounds(CAM_NEAR * 4, CAM_FAR, CASCADES, LAMBDA);
    const splits = bounds.map((b) => b.f);
    let mono = true;
    for (let c = 1; c < splits.length; c++) if (splits[c] <= splits[c - 1]) mono = false;
    ok(mono, `aspect=${f(aspect, 2)}: 远界严格递增`, splits.map((v) => f(v, 2)).join(" < "));
    ok(Math.abs(splits[splits.length - 1] - CAM_FAR) < 1e-6, `末级联远界 == 相机远平面`);
    let okBounds = true;
    for (let c = 0; c < CASCADES; c++) {
      if (Math.abs(bounds[c].f - splits[c]) > 1e-9) okBounds = false;
      const expectNear = c === 0 ? CAM_NEAR * 4 : splits[c - 1];
      if (Math.abs(bounds[c].n - expectNear) > 1e-9) okBounds = false;
    }
    ok(okBounds, `bounds[c] == [splits[c-1], splits[c]]`);
  }
});

section("2. selectCascade：每级联覆盖自己的区间，且无死级联", () => {
  const { bounds } = M.cascadeBounds(CAM_NEAR * 4, CAM_FAR, CASCADES, LAMBDA);
  const splits = bounds.map((b) => b.f);

  for (let c = 0; c < CASCADES; c++) {
    const mid = (bounds[c].n + bounds[c].f) * 0.5;
    const got = M.selectCascade(mid, splits, CASCADES);
    ok(got === c, `级联 ${c} 中点 v=${f(mid, 2)} → 选中 ${c}`, `got ${got}`);
  }
  // 端点：区间的两个端点都必须落在本级联（左开右闭）
  for (let c = 0; c < CASCADES; c++) {
    const hi = M.selectCascade(bounds[c].f, splits, CASCADES);
    ok(hi === c, `级联 ${c} 右端点 v=${f(bounds[c].f, 2)} → 选中 ${c}`, `got ${hi}`);
  }
  // 无死级联：每级联都有一个非空、且被真实选中的区间
  for (let c = 0; c < CASCADES; c++) {
    const lo = c === 0 ? 0 : splits[c - 1];
    const w = splits[c] - lo;
    let hit = 0;
    for (let v = lo + w * 1e-4; v <= splits[c]; v += w / 512) if (M.selectCascade(v, splits, CASCADES) === c) hit++;
    ok(w > 1e-6 && hit > 0, `级联 ${c} 有非空且被选中的区间`, `宽 ${f(w, 2)}，采样命中 ${hit}`);
  }
  // 一致性：级联覆盖范围内的视深度，其选中级联的 [near, far] 必须包含它
  // （范围外：近平面到 near*4 之间、以及远平面之外，不属于任何级联，不参与判定）
  let viol = 0;
  for (let v = bounds[0].n; v <= CAM_FAR; v += 0.05) {
    const c = M.selectCascade(v, splits, CASCADES);
    const b = bounds[Math.min(c, CASCADES - 1)];
    if (v < b.n - 1e-6 || v > b.f + 1e-6) viol++;
  }
  ok(viol === 0, `全部 v ∈ [${f(bounds[0].n, 2)}, ${CAM_FAR}] 都落在所选级联的 [near, far] 内`, `越界 ${viol}`);
});

section("3. 级联矩阵必须覆盖自己的视锥切片（8 角点落入单位立方体）", () => {
  for (const aspect of ASPECTS) {
    const eye = [6.657, 4.77, 7.903];
    const fwd = norm([-6.657, -3.77, -7.903]);
    const right = norm(cross(fwd, [0, 1, 0]));
    const up = cross(right, fwd);
    const { bounds } = M.cascadeBounds(CAM_NEAR * 4, CAM_FAR, CASCADES, LAMBDA);

    let worstX = 0, worstY = 0, zMin = 1e9, zMax = -1e9;
    for (let c = 0; c < CASCADES; c++) {
      const s = sliceCenter(eye, fwd, aspect, bounds[c].n, bounds[c].f);
      const rc = s.r;
      const res = M.cascadeMatrix(SUN_DIR, s.c, rc, (rc * 2) / 2048);
      const halfD = (bounds[c].f - bounds[c].n) * 0.5;
      for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) {
        const p = [
          s.c[0] + right[0] * sx * s.halfW + up[0] * sy * s.halfH + fwd[0] * sz * halfD,
          s.c[1] + right[1] * sx * s.halfW + up[1] * sy * s.halfH + fwd[1] * sz * halfD,
          s.c[2] + right[2] * sx * s.halfW + up[2] * sy * s.halfH + fwd[2] * sz * halfD,
          1,
        ];
        const clip = mul4(res.vp, p);
        const ndc = [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3]];
        worstX = Math.max(worstX, Math.abs(ndc[0]));
        worstY = Math.max(worstY, Math.abs(ndc[1]));
        zMin = Math.min(zMin, ndc[2]);
        zMax = Math.max(zMax, ndc[2]);
      }
    }
    // 容差 0.02：texel 对齐最多把中心推开半个 texel，换算成 NDC 约 1e-3 量级
    ok(worstX <= 1.02 && worstY <= 1.02, `aspect=${f(aspect, 2)}: 8 角点横向都在框内`, `max|ndc.xy| = ${f(Math.max(worstX, worstY), 4)}`);
    ok(zMin >= -1e-3 && zMax <= 1.0 + 1e-3, `aspect=${f(aspect, 2)}: 8 角点深度都在 [0,1]`, `z ∈ [${f(zMin, 4)}, ${f(zMax, 4)}]`);
  }
});

section("4. 混合带必须落在本级联自己的区间内", () => {
  const { bounds } = M.cascadeBounds(CAM_NEAR * 4, CAM_FAR, CASCADES, LAMBDA);
  for (let c = 0; c + 1 < CASCADES; c++) {
    const bandEnd = bounds[c].f;
    const bandPrev = c > 0 ? bounds[c - 1].f : 0.0;
    const bandStart = bandPrev + (bandEnd - bandPrev) * 0.75;
    const inside = bandStart >= bounds[c].n - 1e-9 && bandEnd <= bounds[c].f + 1e-9 && bandStart < bandEnd;
    // 且必须真的会被触发：存在 v ∈ [bandStart, bandEnd] 被选为级联 c
    let reach = false;
    for (let v = bandStart; v <= bandEnd + 1e-9; v += (bandEnd - bandStart) / 64 + 1e-9) {
      if (M.selectCascade(v, bounds.map((b) => b.f), CASCADES) === c) { reach = true; break; }
    }
    ok(inside && reach, `级联 ${c} 混合带 [${f(bandStart, 2)}, ${f(bandEnd, 2)}] 在本级联区间内且可达`);
  }
});

section("5. CPU/GPU 选择逻辑对拍（字面复刻 WGSL 循环）", () => {
  const { bounds } = M.cascadeBounds(CAM_NEAR * 4, CAM_FAR, CASCADES, LAMBDA);
  const splits = bounds.map((b) => b.f);

  // 与 Lighting.wgsl / ForwardPlus.wgsl 中 shadowCascade() 的选择循环逐字对应
  const wgslMirror = (viewDepth, count) => {
    let cascade = 0;
    for (let c = 0; c < count; c++) if (viewDepth > splits[c]) cascade = c + 1;
    return Math.min(cascade, count - 1);
  };
  const oldBuggy = (viewDepth, count) => {
    let cascade = 0;
    for (let c = 0; c < count; c++) if (viewDepth > splits[c]) cascade = c;
    return cascade;
  };

  let diff = 0, firstBad = null;
  for (let v = 0; v <= 130; v += 0.01) {
    if (wgslMirror(v, CASCADES) !== M.selectCascade(v, splits, CASCADES)) {
      diff++;
      if (firstBad === null) firstBad = v;
    }
  }
  ok(diff === 0, "WGSL 复刻与 selectCascade 全域一致", diff ? `首个分歧 v=${f(firstBad, 2)}，共 ${diff} 处` : "13001 个采样点");

  // 反证：旧写法必然违反第 2 节的不变量 —— 这是本测试存在的理由
  let oldViol = 0;
  for (let v = 0; v <= 130; v += 0.01) {
    const c = oldBuggy(v, CASCADES);
    const b = bounds[c];
    if (v < b.n - 1e-6 || v > b.f + 1e-6) oldViol++;
  }
  console.log(`       反证：旧的 \`cascade = c\` 写法有 ${oldViol} / 13001 个视深度落在所选级联框外（会 return 1.0 → 无阴影）`);
});

section("6. 半影比例补偿：共享 zFar 线性化 × k == 本级联 zFar 线性化", () => {
  const { bounds } = M.cascadeBounds(CAM_NEAR * 4, CAM_FAR, CASCADES, LAMBDA);
  const eye = [6.657, 4.77, 7.903];
  const fwd = norm([-6.657, -3.77, -7.903]);
  const aspect = 867 / 415;
  const zNear = 0.1;
  const base = sliceCenter(eye, fwd, aspect, bounds[0].n, bounds[CASCADES - 1].f);
  const zFarBase = base.r * 6 + 10;
  const lin = (d, zn, zf) => zn + d * (zf - zn);

  let worst = 0;
  for (let c = 0; c < CASCADES; c++) {
    const s = sliceCenter(eye, fwd, aspect, bounds[c].n, bounds[c].f);
    const zFarC = s.r * 6 + 10;
    const k = (zFarC - zNear) / (zFarBase - zNear);
    for (const [dr, db] of [[0.2, 0.1], [0.5, 0.3], [0.9, 0.8], [0.99, 0.5]]) {
      const shared = lin(dr, zNear, zFarBase) - lin(db, zNear, zFarBase);
      const own = lin(dr, zNear, zFarC) - lin(db, zNear, zFarC);
      worst = Math.max(worst, Math.abs(shared * k - own));
    }
  }
  ok(worst < 1e-3, "k 补偿后与本级联线性化等价", `最大误差 ${worst.toExponential(2)}`);

  // 不补偿时的放大倍数（越大说明近处半影被糊得越狠）
  const s0 = sliceCenter(eye, fwd, aspect, bounds[0].n, bounds[0].f);
  const blowup = (zFarBase - zNear) / (s0.r * 6 + 10 - zNear);
  console.log(`       不补偿时 级联 0 半影被放大 ${f(blowup, 2)} 倍（会顶到 penumbraUV 上限，看起来像阴影没了）`);
  ok(blowup > 2, "确实存在显著的半影放大（证明补偿有必要）", `${f(blowup, 2)}×`);
});

section("7. 源码守卫：两个 WGSL 必须用修正后的写法", () => {
  for (const name of ["Lighting.wgsl", "ForwardPlus.wgsl"]) {
    const src = readFileSync(join(ROOT, "src", "Shaders", name), "utf8");
    ok(/cascade\s*=\s*c\s*\+\s*1\s*;/.test(src), `${name}: 选级联用 cascade = c + 1`);
    ok(/cascade\s*=\s*min\(cascade,\s*count\s*-\s*1\)/.test(src), `${name}: 选级联后 clamp 到 count-1`);
    ok(/bandEnd/.test(src), `${name}: 混合带用 bandEnd（本级联远界）`);
    ok(!/if\s*\(viewDepth\s*>\s*frame\.csmSplits\[c\]\)\s*\{\s*cascade\s*=\s*c\s*;/.test(src), `${name}: 已不存在旧的 off-by-one 写法`);
    ok(!/mix\(splitEnd,\s*nextSplit/.test(src), `${name}: 已不存在旧混合带写法`);
  }
});

console.log(`\n=== 结论: ${checks - failures}/${checks} 通过, ${failures} 失败 ===`);
process.exit(failures > 0 ? 1 : 0);
