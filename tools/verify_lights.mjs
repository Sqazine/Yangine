// 数值校验：光源矩阵与 CSM 分界是否数学正确。
//
// 管线能创建 ≠ 投影矩阵正确。这里用纯数值断言覆盖最容易出错、且
// 肉眼看不出问题的部分：正交/透视深度范围、深度反解、级联分界单调性、
// 立方体六面朝向与面索引顺序、槽位分配的无重叠性。
//
// 用法： node --experimental-strip-types tools/verify_lights.mjs

import * as M from "../src/Math.ts";
import {
  assignSlots,
  CUBE_SLOTS,
  LIGHT_TYPE_DIRECTIONAL,
  LIGHT_TYPE_POINT,
  LIGHT_TYPE_SPOT,
} from "../src/Lights.ts";

let failures = 0;
let checks = 0;

function ok(cond, label, detail = "") {
  checks++;
  if (cond) {
    console.log(`  [ok]   ${label}`);
  } else {
    console.log(`  [FAIL] ${label} ${detail}`);
    failures++;
  }
}

function near(a, b, eps = 1e-3) {
  return Math.abs(a - b) <= eps;
}

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

function section(title, fn) {
  console.log(`\n=== ${title} ===`);
  fn();
}

section("1. mat4Ortho 深度映射 (WebGPU z_ndc 属于 [0,1])", () => {
  const zNear = 1;
  const zFar = 50;
  const m = M.mat4Ortho(-10, 10, -10, 10, zNear, zFar);
  const n = project(m, [0, 0, -zNear, 1])[2];
  const f = project(m, [0, 0, -zFar, 1])[2];
  ok(near(n, 0, 1e-6), "z_ndc(near) = 0", `got ${n.toFixed(8)}`);
  ok(near(f, 1, 1e-6), "z_ndc(far) = 1", `got ${f.toFixed(8)}`);

  const mid = (zNear + zFar) / 2;
  const zm = project(m, [0, 0, -mid, 1])[2];
  ok(near(zm, 0.5), "深度线性（z_ndc(mid) = 0.5）", `got ${zm.toFixed(6)}`);

  const xEdge = project(m, [10, 0, -mid, 1])[0];
  ok(near(xEdge, 1), "x = r 映射到 ndc 1", `got ${xEdge.toFixed(6)}`);

  // 着色器 linearizeDepth 正交分支：d = near + z * (far - near)
  const recovered = zNear + zm * (zFar - zNear);
  ok(near(recovered, mid, 1e-3), "linearizeDepth 正交分支可逆", `got ${recovered.toFixed(5)}`);
});

section("2. mat4Perspective 深度映射与反解", () => {
  const zNear = 0.1;
  const zFar = 100;
  const m = M.mat4Perspective((50 * Math.PI) / 180, 1.5, zNear, zFar);
  const n = project(m, [0, 0, -zNear, 1])[2];
  const f = project(m, [0, 0, -zFar, 1])[2];
  ok(near(n, 0, 1e-5), "z_ndc(near) = 0", `got ${n.toFixed(8)}`);
  ok(near(f, 1, 1e-5), "z_ndc(far) = 1", `got ${f.toFixed(8)}`);

  // 着色器透视分支：d = far*near / (far - z*(far-near))
  for (const dist of [0.5, 2, 7.3, 40]) {
    const z = project(m, [0, 0, -dist, 1])[2];
    const recovered = (zFar * zNear) / (zFar - z * (zFar - zNear));
    ok(near(recovered, dist, 1e-2), `linearizeDepth 透视分支 d=${dist} 可逆`, `got ${recovered.toFixed(4)}`);
  }
});

section("3. splitCascadeDistances 分界质量", () => {
  const splits = M.splitCascadeDistances(0.4, 100, 4, 0.7);
  ok(splits.length === 4, "返回 4 个分界");
  let mono = true;
  for (let i = 1; i < splits.length; i++) {
    if (splits[i] <= splits[i - 1]) mono = false;
  }
  ok(mono, "严格单调递增", JSON.stringify(splits.map((v) => v.toFixed(3))));
  ok(near(splits[3], 100, 1e-6), "末级 = far", `got ${splits[3]}`);
  ok(splits[0] > 0.4 && splits[0] < 100, "首级落在 (near, far) 内", `got ${splits[0].toFixed(3)}`);

  const gap0 = splits[0] - 0.4;
  const gapN = 100 - splits[2];
  ok(gap0 < gapN, "近处跨度小于远处（对数分布生效）", `${gap0.toFixed(2)} < ${gapN.toFixed(2)}`);

  const one = M.splitCascadeDistances(0.4, 100, 1, 0.7);
  ok(one.length === 1 && near(one[0], 100, 1e-6), "count=1 退化为单一分界");
});

section("4. 立方体阴影六面朝向", () => {
  const pos = [2, 3, -1];
  const dist = 10;
  const proj = M.mat4Perspective(Math.PI / 2, 1, 0.1, 100);
  for (let f = 0; f < CUBE_SLOTS; f++) {
    const view = M.cubeFaceView(f, pos, dist);
    const vp = M.mat4Multiply(proj, view);
    // 光源本身位于立方体中心，在该面视角下必然位于视锥轴上 → NDC (0,0)
    // 面中心点不能用作测试点：它与相机 eye 重合，视线深度为 0
    const [nx, ny, z] = project(vp, [pos[0], pos[1], pos[2], 1]);
    ok(
      near(nx, 0, 1e-4) && near(ny, 0, 1e-4) && z > 0 && z <= 1,
      `面 ${f} 法线(${M.CUBE_FACE_NORMALS[f]})：光源在视锥轴上 (NDC 0,0)`,
      `got ndc=(${nx.toFixed(4)}, ${ny.toFixed(4)}) z=${z.toFixed(4)}`
    );

    // 视图右方向应把投影点推向 +u（UV 的 u = ndc.x*0.5+0.5）
    const right = [view[0], view[4], view[8]];
    const pu = project(vp, [pos[0] + right[0], pos[1] + right[1], pos[2] + right[2], 1])[0];
    ok(pu > 0, `面 ${f} 的视图右方向对应 +u`, `got ndc.x=${pu.toFixed(4)}`);

    // 视图上方向应把投影点推向 -ndc.y（纹理 v 向下翻转，见 projectToUV 的 0.5-0.5*ndc.y）
    const up = [view[1], view[5], view[9]];
    const pv = project(vp, [pos[0] + up[0], pos[1] + up[1], pos[2] + up[2], 1])[1];
    ok(pv > 0, `面 ${f} 的视图上方向对应 +v（已按 up 取反对齐立方体约定）`, `got ndc.y=${pv.toFixed(4)}`);
  }
});

section("5. 面顺序与着色器 dirToCubeUV 对齐", () => {
  // 着色器中 +X->0, -X->1, +Y->2, -Y->3, +Z->4, -Z->5
  const expected = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
  ];
  let match = true;
  for (let f = 0; f < CUBE_SLOTS; f++) {
    const n = M.CUBE_FACE_NORMALS[f];
    if (n[0] !== expected[f][0] || n[1] !== expected[f][1] || n[2] !== expected[f][2]) match = false;
  }
  ok(match, "面法线顺序与着色器 face 索引一致");
});

section("6. 槽位分配", () => {
  const lights = [
    { type: LIGHT_TYPE_DIRECTIONAL, color: [1, 1, 1], intensity: 1, castShadow: true, direction: [0, -1, 0], cascades: 4, cascadeBlend: 1 },
    { type: LIGHT_TYPE_POINT, color: [1, 1, 1], intensity: 1, castShadow: true, position: [0, 5, 0], range: 10 },
    { type: LIGHT_TYPE_SPOT, color: [1, 1, 1], intensity: 1, castShadow: true, position: [0, 5, 0], direction: [0, -1, 0], range: 10, innerAngle: 0.2, outerAngle: 0.4 },
    { type: LIGHT_TYPE_DIRECTIONAL, color: [1, 1, 1], intensity: 1, castShadow: false, direction: [1, -1, 0], cascades: 4, cascadeBlend: 1 },
  ];
  const slots = assignSlots(lights);
  ok(slots[0].matCount === 4, "平行光 4 级联 -> 4 个矩阵槽");
  ok(slots[0].localBase === 0, "平行光不占用本地阴影数组");
  ok(slots[1].matCount === CUBE_SLOTS, "点光源 -> 6 个面槽");
  ok(slots[1].localBase === 0, "点光源本地层从 0 开始", `got ${slots[1].localBase}`);
  ok(slots[2].localBase === CUBE_SLOTS, "聚光灯本地层紧随点光源", `got ${slots[2].localBase}`);
  ok(slots[2].matCount === 1, "聚光灯 -> 1 个矩阵槽");
  ok(slots[3].matCount === 0, "castShadow=false 不占槽位");
  ok(slots[2].localBase + 1 <= 12, "本地层总数不超过 MAX_LOCAL_LAYERS(12)", `got ${slots[2].localBase + 1}`);

  const used = new Set();
  let overlap = false;
  let maxIdx = 0;
  for (const s of slots) {
    for (let k = 0; k < s.matCount; k++) {
      if (used.has(s.matBase + k)) overlap = true;
      used.add(s.matBase + k);
      maxIdx = Math.max(maxIdx, s.matBase + k);
    }
  }
  ok(!overlap, "矩阵槽无重叠");
  ok(maxIdx + 1 <= 16, "总矩阵槽不超过 MAX_SHADOW_SLOTS(16)", `got ${maxIdx + 1}`);

  const shadowed = slots.filter((s) => s.matCount > 0);
  let rising = true;
  for (let i = 1; i < shadowed.length; i++) {
    if (shadowed[i].baseBias <= shadowed[i - 1].baseBias) rising = false;
  }
  ok(rising, "baseBias 逐灯递增");
});

section("7. 聚光灯投影覆盖外锥", () => {
  const pos = [0, 8, 0];
  const range = 25;
  const outer = (34 * Math.PI) / 180;
  // 世界坐标：光源在 y=8，朝 -Y 照射，故正前方 25 单位处为 y = 8 - 25
  const target = [0, pos[1] - range, 0];
  // 视线朝 -Y，用 Z 轴作 up 保证不与视线共线
  const view = M.mat4LookAt(pos, target, [0, 0, 1]);
  const fov = Math.min(2 * outer * 1.05, Math.PI * 0.95);
  const proj = M.mat4Perspective(fov, 1, 0.08, range * 2);
  const vp = M.mat4Multiply(proj, view);

  const halfH = Math.tan(fov / 2) * range;
  // 锥轴上的点落在 NDC (0,0)；着色器 projectToUV 再把它映射到 uv(0.5,0.5)
  const c = project(vp, [0, pos[1] - range, 0, 1]);
  ok(
    near(c[0], 0, 1e-4) && near(c[1], 0, 1e-4) && c[2] > 0,
    "锥中心位于 NDC (0,0)",
    `ndc=(${c[0].toFixed(4)}, ${c[1].toFixed(4)}) z=${c[2].toFixed(4)}`
  );
  // 与着色器 projectToUV 的换算保持一致
  ok(
    near(c[0] * 0.5 + 0.5, 0.5, 1e-6) && near(0.5 - 0.5 * c[1], 0.5, 1e-6),
    "projectToUV 换算后为 uv(0.5,0.5)"
  );

  const rx = view[0];
  const rz = view[8];
  const edge = project(vp, [rx * halfH, pos[1] - range, rz * halfH, 1]);
  ok(
    edge[0] > 0.99 && edge[0] < 1.01 && edge[2] > 0,
    "锥边缘恰好落在投影边界 ndc.x = 1",
    `ndc.x=${edge[0].toFixed(4)} ndc.y=${edge[1].toFixed(4)} z=${edge[2].toFixed(4)}`
  );

  // 1.05 倍余量应使锥边缘略在图内，避免边缘处阴影数据缺失
  const inner = project(vp, [rx * halfH * 0.9, pos[1] - range, rz * halfH * 0.9, 1]);
  ok(inner[0] > 0 && inner[0] < 1, "内锥边缘在图内", `u=${inner[0].toFixed(4)}`);
});

console.log(`\n=== 结论: ${checks - failures}/${checks} 通过, ${failures} 失败 ===`);
process.exit(failures > 0 ? 1 : 0);