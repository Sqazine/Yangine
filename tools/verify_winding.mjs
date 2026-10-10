// 校验所有几何生成器的「绕序 ↔ 顶点法线」一致性。
//
// 为什么需要它：绕序决定背面剔除，顶点法线决定光照，两者必须指向同一侧。
// 一旦某个生成器的绕序反了，症状是**这个物体从外面直接消失（被整片剔除）**，
// 而不是变黑或变暗——所以极易被误诊成材质、深度或光照问题。
// 透明物体用 cullMode:"none" 会把症状藏起来，但那只是没暴露，不是没 bug。
//
// 判定方式不依赖任何 API 的正反面约定（WebGPU/Vulkan 的 ccw/cw 与 y 轴翻转耦合，
// 很容易记反）：对每个三角形算右手法则几何法线 cross(v1−v0, v2−v0)，
// 与三个顶点法线的均值比较。同向 = 该三角形自洽。
//
// 用法： 需经 esbuild 打包（依赖 Scene.ts），见 package.json 的 verify:winding

import { makeCube, makePlane, makeSphere } from "../src/Scene.ts";

let failures = 0;
let checks = 0;

function ok(cond, label, detail = "") {
  checks++;
  console.log(`  ${cond ? "[ok]  " : "[FAIL]"} ${label}${detail ? "  " + detail : ""}`);
  if (!cond) failures++;
}
function section(t, fn) { console.log(`\n=== ${t} ===`); fn(); }

function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function len(a) { return Math.hypot(a[0], a[1], a[2]); }
const norm = (a) => { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const fmt = (a) => `[${a.map((v) => v.toFixed(2)).join(",")}]`;

const pos = (m, i) => [m.vertices[i * 8], m.vertices[i * 8 + 1], m.vertices[i * 8 + 2]];
const nrm = (m, i) => [m.vertices[i * 8 + 3], m.vertices[i * 8 + 4], m.vertices[i * 8 + 5]];

/** 三角形 t 的右手法则几何法线（未归一化）。 */
function geoNormal(m, t) {
  const i0 = m.indices[t], i1 = m.indices[t + 1], i2 = m.indices[t + 2];
  return cross(sub(pos(m, i1), pos(m, i0)), sub(pos(m, i2), pos(m, i0)));
}

/** 三角形 t 三个顶点法线的均值（同一面内应一致）。 */
function vertexNormal(m, t) {
  const ids = [m.indices[t], m.indices[t + 1], m.indices[t + 2]];
  return [
    ids.reduce((s, i) => s + nrm(m, i)[0], 0),
    ids.reduce((s, i) => s + nrm(m, i)[1], 0),
    ids.reduce((s, i) => s + nrm(m, i)[2], 0),
  ];
}

/**
 * 报告绕序自洽率，并打印前几个反向三角形（几何法线 vs 顶点法线）便于定位。
 * 退化三角形（几何法线长度 ≈ 0，如球体两极）单独计数并排除。
 */
function consistency(m, name) {
  let agree = 0, disagree = 0, degenerate = 0;
  const bad = [];
  for (let t = 0; t < m.indexCount; t += 3) {
    const g = geoNormal(m, t);
    const n = vertexNormal(m, t);
    if (len(g) < 1e-9 || len(n) < 1e-9) { degenerate++; continue; }
    if (dot(g, n) > 0) agree++;
    else {
      disagree++;
      if (bad.length < 3) {
        bad.push(`tri#${t / 3} idx=[${m.indices[t]},${m.indices[t + 1]},${m.indices[t + 2]}] 几何n=${fmt(norm(g))} 顶点n=${fmt(norm(n))}`);
      }
    }
  }
  const total = agree + disagree;
  const ratio = total ? agree / total : 0;
  console.log(`       ${name}: 三角形 ${total}（退化 ${degenerate}） 自洽 ${agree} / 反向 ${disagree} → ${(ratio * 100).toFixed(1)}%`);
  for (const b of bad) console.log(`         · ${b}`);
  return { agree, disagree, total, ratio, degenerate };
}

const cube = makeCube(2);
const plane = makePlane(4);
const sphere = makeSphere(1, 16, 12);

section("1. 绕序必须与顶点法线同向（自洽率 100%）", () => {
  const r = {
    makeCube: consistency(cube, "makeCube"),
    makePlane: consistency(plane, "makePlane"),
    makeSphere: consistency(sphere, "makeSphere"),
  };
  for (const [name, v] of Object.entries(r)) {
    ok(v.ratio === 1, `${name}: 绕序与顶点法线全部自洽`, `${(v.ratio * 100).toFixed(1)}%（反向 ${v.disagree}）`);
  }
});

section("2. 方向语义", () => {
  // 地面：顶点法线朝上，绕序几何法线也必须朝上
  let allUp = true;
  for (let i = 0; i < plane.vertices.length / 8; i++) {
    if (Math.abs(nrm(plane, i)[1] - 1) > 1e-6) allUp = false;
  }
  ok(allUp, "makePlane: 顶点法线全部为 +Y（朝上）");

  const gPlane = norm(geoNormal(plane, 0));
  ok(gPlane[1] > 0.99, "makePlane: 几何法线朝 +Y（从上方不会被剔除）", `n=${fmt(gPlane)}`);

  // 封闭体：几何法线必须背向形心（立方体/球体形心都在原点）
  for (const [name, mesh] of [["makeCube", cube], ["makeSphere", sphere]]) {
    let outward = 0, inward = 0;
    for (let t = 0; t < mesh.indexCount; t += 3) {
      const g = geoNormal(mesh, t);
      if (len(g) < 1e-9) continue;
      if (dot(g, pos(mesh, mesh.indices[t])) > 0) outward++; else inward++;
    }
    ok(inward === 0, `${name}: 每个三角形几何法线都背向形心`, `朝外 ${outward} / 朝内 ${inward}`);
  }
});

section("3. 索引与退化三角形健全性", () => {
  for (const [name, mesh, segments] of [["makeCube", cube, 0], ["makePlane", plane, 0], ["makeSphere", sphere, 16]]) {
    let oob = 0, deg = 0;
    const vcount = mesh.vertices.length / 8;
    for (let t = 0; t < mesh.indexCount; t += 3) {
      for (let k = 0; k < 3; k++) {
        if (mesh.indices[t + k] >= vcount) oob++;
      }
      if (len(geoNormal(mesh, t)) < 1e-9) deg++;
    }
    ok(oob === 0, `${name}: 索引不越界（顶点数 ${vcount}）`, `越界 ${oob}`);
    // 退化三角形只应出现在球体两极（每极 segments 个）；立方体/平面应全为 0
    const expected = 2 * segments;
    ok(deg === expected, `${name}: 退化三角形数符合预期`, `${deg}（预期 ${expected}）`);
  }
});

console.log(`\n=== 结论: ${checks - failures}/${checks} 通过, ${failures} 失败 ===`);
process.exit(failures > 0 ? 1 : 0);
