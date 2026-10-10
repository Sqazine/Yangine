// 场景：几何体生成、轨道相机、可渲染对象（含材质与不透明/透明分类）。

import {
  Mat4,
  Vec3,
  mat4Identity,
  mat4LookAt,
  mat4Multiply,
  mat4Perspective,
  mat4TRS,
  vec3Add,
  vec3Normalize,
  vec3Sub,
} from "./Math";
import { Light } from "./Lights";

export interface Mesh {
  vertices: Float32Array; // 交错: [px,py,pz, nx,ny,nz, u,v]
  indices: Uint32Array;
  indexCount: number;
}

export interface Material {
  baseColor: [number, number, number, number]; // rgb + a(opacity)
  metallic: number;
  roughness: number;
  emissive: number; // 自发光强度
  opacity: number; // 透明度（<1 视为透明）
  transparent: boolean;
}

export interface Renderable {
  mesh: Mesh;
  position: Vec3;
  rotationY: number;
  scale: Vec3;
  material: Material;
  /** 可选动画回调：每帧根据 time 修改 transform。 */
  animate?: (r: Renderable, time: number) => void;
}

export class Scene {
  objects: Renderable[] = [];
  lights: Light[] = [];

  add(r: Renderable) {
    this.objects.push(r);
  }

  addLight(l: Light) {
    this.lights.push(l);
  }
}

// ---------- 几何体生成 ----------
//
// 【绕序约定】三个生成器必须统一：三角形按右手法则算出的几何法线
//   N_geo = normalize(cross(v1 − v0, v2 − v0))
// 必须与三个顶点的法线属性指向同一侧。两个方向各管一件事，缺一不可：
//   · 顶点法线 → 光照（着色器里 obj.normalMatrix 变换后直接用）
//   · 绕序     → 背面剔除（pGeometry / pPrepass / pForwardPlus 都是 cullMode:"back"）
// 两者不一致时的症状很反直觉：**物体直接从外面消失**（被剔除），而不是变黑或变暗。
// 因此它极易被误诊成材质、深度或光照问题，而根因只是几个顶点的排列顺序。
// 不透明物体务必走 tools/verify_winding.mjs 的检查；透明物体虽然用 cullMode:"none"
// 掩盖了剔除症状，但那只是把 bug 藏起来，不是修好。

function buildMesh(
  positions: number[],
  normals: number[],
  uvs: number[],
  indices: number[]
): Mesh {
  const n = positions.length / 3;
  const vertices = new Float32Array(n * 8);
  for (let i = 0; i < n; i++) {
    vertices[i * 8 + 0] = positions[i * 3 + 0];
    vertices[i * 8 + 1] = positions[i * 3 + 1];
    vertices[i * 8 + 2] = positions[i * 3 + 2];
    vertices[i * 8 + 3] = normals[i * 3 + 0];
    vertices[i * 8 + 4] = normals[i * 3 + 1];
    vertices[i * 8 + 5] = normals[i * 3 + 2];
    vertices[i * 8 + 6] = uvs[i * 2 + 0];
    vertices[i * 8 + 7] = uvs[i * 2 + 1];
  }
  const idx = new Uint32Array(indices);
  return { vertices, indices: idx, indexCount: idx.length };
}

export function makeCube(size = 1): Mesh {
  const h = size / 2;
  const p: number[] = [];
  const n: number[] = [];
  const u: number[] = [];
  const idx: number[] = [];
  // 6 个面，每面 4 顶点
  const faces = [
    { nrm: [0, 0, 1], verts: [[-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]] }, // +Z
    { nrm: [0, 0, -1], verts: [[h, -h, -h], [-h, -h, -h], [-h, h, -h], [h, h, -h]] }, // -Z
    { nrm: [1, 0, 0], verts: [[h, -h, h], [h, -h, -h], [h, h, -h], [h, h, h]] }, // +X
    { nrm: [-1, 0, 0], verts: [[-h, -h, -h], [-h, -h, h], [-h, h, h], [-h, h, -h]] }, // -X
    { nrm: [0, 1, 0], verts: [[-h, h, h], [h, h, h], [h, h, -h], [-h, h, -h]] }, // +Y
    { nrm: [0, -1, 0], verts: [[-h, -h, -h], [h, -h, -h], [h, -h, h], [-h, -h, h]] }, // -Y
  ];
  faces.forEach((f, fi) => {
    const base = fi * 4;
    const uvs = [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 1],
    ];
    for (let i = 0; i < 4; i++) {
      p.push(...f.verts[i]);
      n.push(...f.nrm);
      u.push(...uvs[i]);
    }
    idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
  });
  return buildMesh(p, n, u, idx);
}

export function makePlane(size = 20): Mesh {
  const h = size / 2;
  const p = [
    -h, 0, -h, h, 0, -h, h, 0, h, -h, 0, h,
  ];
  const n = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0];
  const u = [0, 0, 1, 0, 1, 1, 0, 1];
  // 绕序必须与 +Y 的顶点法线同向：0→2→1 / 0→3→2。
  // 写成自然顺序 0→1→2 / 0→2→3 时几何法线是 -Y，地面会在 cullMode:"back" 下
  // 从上方被整片剔除——地面「看不见」而不是「变黑」。
  const idx = [0, 2, 1, 0, 3, 2];
  return buildMesh(p, n, u, idx);
}

export function makeSphere(radius = 1, segments = 32, rings = 24): Mesh {
  const p: number[] = [];
  const n: number[] = [];
  const u: number[] = [];
  const idx: number[] = [];
  for (let y = 0; y <= rings; y++) {
    const v = y / rings;
    const theta = v * Math.PI;
    for (let x = 0; x <= segments; x++) {
      const uu = x / segments;
      const phi = uu * Math.PI * 2;
      const nx = Math.sin(theta) * Math.cos(phi);
      const ny = Math.cos(theta);
      const nz = Math.sin(theta) * Math.sin(phi);
      p.push(nx * radius, ny * radius, nz * radius);
      n.push(nx, ny, nz);
      u.push(uu, v);
    }
  }
  const stride = segments + 1;
  for (let y = 0; y < rings; y++) {
    for (let x = 0; x < segments; x++) {
      const a = y * stride + x;
      const b = a + 1;
      const c = a + stride;
      const d = c + 1;
      // 绕序必须让几何法线朝外：a→b→c / b→d→c。
      // 写成 a→c→b / b→c→d 会让整个球反面朝外。目前只有透明度管线
      // (cullMode:"none") 用到球体，所以不会消失——但一旦被当不透明物体用就会整颗不见。
      idx.push(a, b, c, b, d, c);
    }
  }
  return buildMesh(p, n, u, idx);
}

// ---------- 轨道相机 ----------

export class OrbitCamera {
  target: Vec3 = [0, 1, 0];
  distance = 11;
  yaw = 0.7;
  pitch = 0.35;
  fov = (50 * Math.PI) / 180;
  near = 0.1;
  far = 100;
  aspect = 1;

  eye(): Vec3 {
    const cp = Math.cos(this.pitch);
    return vec3Add(this.target, [
      cp * Math.sin(this.yaw) * this.distance,
      Math.sin(this.pitch) * this.distance,
      cp * Math.cos(this.yaw) * this.distance,
    ]);
  }

  private view(): Mat4 {
    return mat4LookAt(this.eye(), this.target, [0, 1, 0]);
  }

  private proj(): Mat4 {
    return mat4Perspective(this.fov, this.aspect, this.near, this.far);
  }

  /** 未带抖动的 view-projection。 */
  viewProj(): Mat4 {
    return mat4Multiply(this.proj(), this.view());
  }

  /** 相机前向（eye → target 的单位向量）。CSM 的视深度与级联选择依赖它。 */
  forward(): Vec3 {
    return vec3Normalize(vec3Sub(this.target, this.eye()));
  }
}

export function computeModel(r: Renderable): Mat4 {
  return mat4TRS(r.position, r.rotationY, r.scale);
}

export const IDENTITY: Mat4 = mat4Identity();