// 最小化的 mat4 / vec3 数学库（列主序，匹配 WGSL mat4x4<f32>）。

export type Mat4 = Float32Array;
export type Vec3 = [number, number, number];

export function mat4Identity(): Mat4 {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

/** o = a * b （两者均为列主序）。 */
export function mat4Multiply(a: Mat4, b: Mat4): Mat4 {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    const b0 = b[c * 4],
      b1 = b[c * 4 + 1],
      b2 = b[c * 4 + 2],
      b3 = b[c * 4 + 3];
    o[c * 4 + 0] = a[0] * b0 + a[4] * b1 + a[8] * b2 + a[12] * b3;
    o[c * 4 + 1] = a[1] * b0 + a[5] * b1 + a[9] * b2 + a[13] * b3;
    o[c * 4 + 2] = a[2] * b0 + a[6] * b1 + a[10] * b2 + a[14] * b3;
    o[c * 4 + 3] = a[3] * b0 + a[7] * b1 + a[11] * b2 + a[15] * b3;
  }
  return o;
}

/**
 * 透视投影（WebGPU 约定：右手系、NDC z ∈ [0,1]）。
 * 返回的列主序矩阵第 [8]、[9] 元素分别对应 z 对 clip.x / clip.y 的系数，
 * 偏移抖动(jitter)时直接修改这两个元素即可在 NDC 上获得恒定偏移。
 */
export function mat4Perspective(
  fovY: number,
  aspect: number,
  near: number,
  far: number
): Mat4 {
  const f = 1 / Math.tan(fovY / 2);
  const nf = 1 / (near - far);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = far * nf;
  m[11] = -1;
  m[14] = far * near * nf;
  return m;
}

/**
 * 正交投影（WebGPU 约定：z_ndc ∈ [0,1]，z_view = -distance）。
 * near/far 传入的是沿 -Z 的距离，映射后 near→0、far→1。
 */
export function mat4Ortho(
  l: number,
  r: number,
  b: number,
  t: number,
  near: number,
  far: number
): Mat4 {
  const m = new Float32Array(16);
  m[0] = 2 / (r - l);
  m[5] = 2 / (t - b);
  m[10] = 1 / (near - far);
  m[12] = -(r + l) / (r - l);
  m[13] = -(t + b) / (t - b);
  m[14] = near / (near - far);
  m[15] = 1;
  return m;
}

/**
 * Practical split scheme（Lauritzen）：对数分布与均匀分布按 lambda 混合，
 * 兼顾近处精度与远处覆盖。返回 count 个级联的**结束**距离。
 */
export function splitCascadeDistances(
  near: number,
  far: number,
  count: number,
  lambda = 0.75
): number[] {
  const out: number[] = [];
  const ratio = far / near;
  for (let i = 1; i <= count; i++) {
    const p = i / count;
    const logD = near * Math.pow(ratio, p);
    const uniD = near + (far - near) * p;
    out.push(lambda * logD + (1 - lambda) * uniD);
  }
  out[out.length - 1] = far;
  return out;
}

/**
 * 把「级联结束距离」展开成每级联的 [near, far]，并给出写入帧 uniform 的远界数组。
 *
 * 三处约定必须**成对**出现（历史上这里出过一次 off-by-one，后果是远处整片阴影静默消失）：
 *   · splits[c] = 级联 c 的**远界**
 *   · bounds[c] = { n: splits[c-1], f: splits[c] }，splits[-1] 取 near —— 级联矩阵由它构建
 *   · 选取：viewDepth ≤ splits[c] 的最小 c（见 selectCascade）
 * 不能一边把 splits[c] 当远界（建矩阵时）、另一边当近界（选级联时）。
 */
export function cascadeBounds(
  near: number,
  far: number,
  count: number,
  lambda = 0.7
): { bounds: { n: number; f: number }[]; splits: number[] } {
  const splits = splitCascadeDistances(near, far, count, lambda);
  const bounds: { n: number; f: number }[] = [];
  let prev = near;
  for (let c = 0; c < count; c++) {
    bounds.push({ n: prev, f: splits[c] });
    prev = splits[c];
  }
  return { bounds, splits };
}

/**
 * 视深度 → 级联索引：第一个满足 viewDepth ≤ splits[c] 的 c；超出最远级联时兜底到 count-1。
 * 必须与 WGSL 侧 shadowCascade() 的选择循环语义一致 —— tools/verify_csm.mjs 会做 CPU/GPU 对拍。
 */
export function selectCascade(viewDepth: number, splits: ArrayLike<number>, count: number): number {
  for (let c = 0; c < count; c++) {
    if (viewDepth <= splits[c]) return c;
  }
  return Math.max(0, count - 1);
}

/**
 * CSM 级联的正交矩阵：以视锥切片的包围球为范围，按 texel 网格对齐以消除相机移动时的边缘爬行。
 *
 * 约定：包围球半径即正交框的横向半宽，所以切片 8 个角点必须全部落在单位立方体内 ——
 * tools/verify_csm.mjs 会逐角点验证这一点。一旦半径算错或对齐偏移过大，
 * 切片边缘会落到正交框外，该处阴影直接消失。
 */
export function cascadeMatrix(
  lightDir: Vec3,
  center: Vec3,
  radius: number,
  texelSize: number
): { vp: Mat4; center: Vec3; radius: number } {
  const back: Vec3 = [
    center[0] - lightDir[0] * radius * 3,
    center[1] - lightDir[1] * radius * 3,
    center[2] - lightDir[2] * radius * 3,
  ];
  // 接近垂直时改用 Z 轴作 up，避免 lookAt 叉积退化
  const up: Vec3 = Math.abs(lightDir[1]) > 0.98 ? [0, 0, 1] : [0, 1, 0];
  const view = mat4LookAt(back, center, up);

  // 把包围球中心投到光空间，按 texel 网格对齐
  const c = [center[0], center[1], center[2], 1];
  const vx = view[0] * c[0] + view[4] * c[1] + view[8] * c[2] + view[12];
  const vy = view[1] * c[0] + view[5] * c[1] + view[9] * c[2] + view[13];
  const snappedX = Math.round(vx / texelSize) * texelSize;
  const snappedY = Math.round(vy / texelSize) * texelSize;

  // 反算对齐后的世界空间中心
  const snappedCenter: Vec3 = [
    center[0] + view[0] * (snappedX - vx) + view[1] * (snappedY - vy),
    center[1] + view[4] * (snappedX - vx) + view[5] * (snappedY - vy),
    center[2] + view[8] * (snappedX - vx) + view[9] * (snappedY - vy),
  ];
  const snappedBack: Vec3 = [
    snappedCenter[0] - lightDir[0] * radius * 3,
    snappedCenter[1] - lightDir[1] * radius * 3,
    snappedCenter[2] - lightDir[2] * radius * 3,
  ];
  const snappedView = mat4LookAt(snappedBack, snappedCenter, up);
  const proj = mat4Ortho(-radius, radius, -radius, radius, 0.1, radius * 6 + 10);
  return { vp: mat4Multiply(proj, snappedView), center: snappedCenter, radius };
}

/** 立方体阴影的六个面法线，顺序与着色器 dirToCubeUV 的 face 索引一致。 */
export const CUBE_FACE_NORMALS: Vec3[] = [
  [1, 0, 0],
  [-1, 0, 0],
  [0, 1, 0],
  [0, -1, 0],
  [0, 0, 1],
  [0, 0, -1],
];

/** 立方体六面的参考 up（实际传入 mat4LookAt 时取负，见 cubeFaceView）。 */
export const CUBE_FACE_UPS: Vec3[] = [
  [0, -1, 0],
  [0, -1, 0],
  [0, 0, 1],
  [0, 0, -1],
  [0, -1, 0],
  [0, -1, 0],
];

/**
 * 立方体某一面的 view 矩阵。
 * up 需要取负：渲染到 2D array 后，纹理 v 轴（向下）与立方体映射约定反向，
 * 传入 -up 可让 NDC.y 与 (tc/ma+1)/2 直接对齐，省掉采样端的翻转。
 */
export function cubeFaceView(face: number, position: Vec3, distance: number): Mat4 {
  const n = CUBE_FACE_NORMALS[face];
  const up = CUBE_FACE_UPS[face];
  const eye: Vec3 = [
    position[0] + n[0] * distance,
    position[1] + n[1] * distance,
    position[2] + n[2] * distance,
  ];
  return mat4LookAt(eye, position, [-up[0], -up[1], -up[2]]);
}

export function mat4LookAt(eye: Vec3, center: Vec3, up: Vec3): Mat4 {
  const z = vec3Normalize(vec3Sub(eye, center)); // 指向相机后方
  const x = vec3Normalize(vec3Cross(up, z));
  const y = vec3Cross(z, x);
  const m = new Float32Array(16);
  m[0] = x[0];
  m[1] = y[0];
  m[2] = z[0];
  m[3] = 0;
  m[4] = x[1];
  m[5] = y[1];
  m[6] = z[1];
  m[7] = 0;
  m[8] = x[2];
  m[9] = y[2];
  m[10] = z[2];
  m[11] = 0;
  m[12] = -vec3Dot(x, eye);
  m[13] = -vec3Dot(y, eye);
  m[14] = -vec3Dot(z, eye);
  m[15] = 1;
  return m;
}

export function mat4Invert(a: Mat4): Mat4 {
  const a00 = a[0],
    a01 = a[1],
    a02 = a[2],
    a03 = a[3];
  const a10 = a[4],
    a11 = a[5],
    a12 = a[6],
    a13 = a[7];
  const a20 = a[8],
    a21 = a[9],
    a22 = a[10],
    a23 = a[11];
  const a30 = a[12],
    a31 = a[13],
    a32 = a[14],
    a33 = a[15];
  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;
  let det =
    b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  const o = new Float32Array(16);
  if (!det) return o;
  det = 1.0 / det;
  o[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  o[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  o[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  o[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  o[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  o[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  o[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  o[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  o[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  o[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  o[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  o[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  o[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  o[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  o[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  o[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return o;
}

/** 由 model 矩阵构造法线矩阵（左上 3x3 的逆的转置），以 mat4 形式返回。 */
export function mat4NormalFromModel(m: Mat4): Mat4 {
  const a00 = m[0],
    a01 = m[1],
    a02 = m[2];
  const a10 = m[4],
    a11 = m[5],
    a12 = m[6];
  const a20 = m[8],
    a21 = m[9],
    a22 = m[10];
  const det =
    a00 * (a11 * a22 - a12 * a21) -
    a01 * (a10 * a22 - a12 * a20) +
    a02 * (a10 * a21 - a11 * a20);
  const o = mat4Identity();
  if (!det) return o;
  const id = 1 / det;
  // 3x3 逆（列主序）
  const i00 = (a11 * a22 - a12 * a21) * id;
  const i01 = (a12 * a20 - a10 * a22) * id;
  const i02 = (a10 * a21 - a11 * a20) * id;
  const i10 = (a02 * a21 - a01 * a22) * id;
  const i11 = (a00 * a22 - a02 * a20) * id;
  const i12 = (a01 * a20 - a00 * a21) * id;
  const i20 = (a01 * a12 - a02 * a11) * id;
  const i21 = (a02 * a10 - a00 * a12) * id;
  const i22 = (a00 * a11 - a01 * a10) * id;
  // 转置后嵌入 mat4
  o[0] = i00;
  o[1] = i10;
  o[2] = i20;
  o[4] = i01;
  o[5] = i11;
  o[6] = i21;
  o[8] = i02;
  o[9] = i12;
  o[10] = i22;
  return o;
}

export function mat4Translation(t: Vec3): Mat4 {
  const m = mat4Identity();
  m[12] = t[0];
  m[13] = t[1];
  m[14] = t[2];
  return m;
}

export function mat4Scale(s: Vec3): Mat4 {
  const m = new Float32Array(16);
  m[0] = s[0];
  m[5] = s[1];
  m[10] = s[2];
  m[15] = 1;
  return m;
}

export function mat4RotationY(rad: number): Mat4 {
  const c = Math.cos(rad),
    s = Math.sin(rad);
  const m = mat4Identity();
  m[0] = c;
  m[2] = -s;
  m[8] = s;
  m[10] = c;
  return m;
}

export function mat4RotationX(rad: number): Mat4 {
  const c = Math.cos(rad),
    s = Math.sin(rad);
  const m = mat4Identity();
  m[5] = c;
  m[6] = s;
  m[9] = -s;
  m[10] = c;
  return m;
}

export function mat4TRS(t: Vec3, rotY: number, scale: Vec3): Mat4 {
  return mat4Multiply(
    mat4Multiply(mat4Translation(t), mat4RotationY(rotY)),
    mat4Scale(scale)
  );
}

// ---- vec3 ----
export function vec3Add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}
export function vec3Sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
export function vec3Scale(a: Vec3, s: number): Vec3 {
  return [a[0] * s, a[1] * s, a[2] * s];
}
export function vec3Dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
export function vec3Cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}
export function vec3Length(a: Vec3): number {
  return Math.sqrt(vec3Dot(a, a));
}
export function vec3Normalize(a: Vec3): Vec3 {
  const l = vec3Length(a);
  if (l < 1e-8) return [0, 0, 0];
  return [a[0] / l, a[1] / l, a[2] / l];
}

/** Halton 低差异序列，用于 TAA 子像素抖动。 */
export function halton(index: number, base: number): number {
  let result = 0;
  let f = 1;
  let i = index;
  while (i > 0) {
    f /= base;
    result += f * (i % base);
    i = Math.floor(i / base);
  }
  return result;
}
