// Forward+ 的分块光源剔除（Tiled Light Culling）。
//
// 每个 workgroup 处理一个 TILE×TILE 的屏幕分块：
//   1. 先并行读入分块内所有像素的深度，用 atomic 求线性深度的 min/max
//   2. 由分块边界构造四条侧向视锥平面（过相机原点的平面）
//   3. 用「光源包围球 vs 视锥」做剔除，输出该分块可见的光源索引列表
//
// 输出采用固定步长布局（每分块 lightsPerTile 个槽位），
// 因此无需全局原子计数器，也省掉每帧重置计数器的开销。

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
  gridParams    : vec4<f32>,  // x=tilesX y=tilesY z=lightsPerTile w=pad
};

struct Light {
  position  : vec4<f32>,  // xyz 位置, w 衰减半径
  direction : vec4<f32>,  // xyz 方向, w cos(outer)
  color     : vec4<f32>,  // rgb 颜色, a 强度
  params    : vec4<f32>,  // x type, y matBase, z matCount, w baseBias
  shadowP   : vec4<f32>,
  cone      : vec4<f32>,
  csmSizes  : vec4<f32>,
  misc      : vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame : Frame;
@group(0) @binding(1) var depthTex : texture_depth_2d;
@group(0) @binding(2) var<storage, read> lights : array<Light>;
@group(0) @binding(3) var<storage, read_write> lightGrid : array<u32>;
@group(0) @binding(4) var<storage, read_write> lightIndices : array<u32>;

const TILE : u32 = 16u;

var<workgroup> tileMinDepth : atomic<u32>;
var<workgroup> tileMaxDepth : atomic<u32>;

fn linearizeDepth(d : f32) -> f32 {
  let zNear = frame.params.y;
  let zFar = frame.params.z;
  return (zFar * zNear) / max(zFar - d * (zFar - zNear), 1e-5);
}

/** 由 NDC 坐标反投影出该方向在视空间的射线（远平面 z_ndc = 1）。 */
fn rayDir(x : f32, y : f32) -> vec3<f32> {
  let p = frame.invProj * vec4<f32>(x, y, 1.0, 1.0);
  return normalize(p.xyz / p.w);
}

/** 构造过相机原点、由两条射线张成的平面法线，并统一朝向分块内部。 */
fn planeNormal(a : vec3<f32>, b : vec3<f32>, center : vec3<f32>) -> vec3<f32> {
  var n = normalize(cross(a, b));
  if (dot(n, center) < 0.0) { n = -n; }
  return n;
}

@compute @workgroup_size(16, 16, 1)
fn cs(@builtin(global_invocation_id) gid : vec3<u32>,
      @builtin(local_invocation_index) lindex : u32,
      @builtin(workgroup_id) wid : vec3<u32>) {
  if (lindex == 0u) {
    atomicStore(&tileMinDepth, 0x7F7FFFFFu); // +FLT_MAX
    atomicStore(&tileMaxDepth, 0u);
  }
  workgroupBarrier();

  // ---- 1. 求分块的深度范围 ----
  let dim = vec2<i32>(textureDimensions(depthTex));
  let pc = vec2<i32>(gid.xy);
  if (pc.x < dim.x && pc.y < dim.y) {
    let d = textureLoad(depthTex, pc, 0);
    if (d < 1.0) {
      // 线性深度恒为正，IEEE-754 正浮点数的序与整数序一致，可直接 atomicMin/Max
      let lin = linearizeDepth(d);
      atomicMin(&tileMinDepth, bitcast<u32>(lin));
      atomicMax(&tileMaxDepth, bitcast<u32>(lin));
    }
  }
  workgroupBarrier();

  // ---- 2. 由线程 0 完成剔除与写回 ----
  if (lindex != 0u) { return; }

  let tilesX = u32(frame.gridParams.x);
  let stride = u32(frame.gridParams.z);
  let tileIdx = wid.y * tilesX + wid.x;

  let minD = bitcast<f32>(atomicLoad(&tileMinDepth));
  var maxD = bitcast<f32>(atomicLoad(&tileMaxDepth));
  if (maxD <= 0.0) {
    lightGrid[tileIdx] = 0u;   // 分块内无可见几何
    return;
  }
  maxD = maxD * 1.05;          // 留一点余量，避免边界抖动导致漏光

  // ---- 3. 构造分块视锥的四个侧面 ----
  let W = f32(dim.x);
  let H = f32(dim.y);
  let x0 = (f32(wid.x * TILE) / W) * 2.0 - 1.0;
  let x1 = (f32((wid.x + 1u) * TILE) / W) * 2.0 - 1.0;
  let yTop = 1.0 - (f32(wid.y * TILE) / H) * 2.0;
  let yBot = 1.0 - (f32((wid.y + 1u) * TILE) / H) * 2.0;

  let rTL = rayDir(x0, yTop);
  let rTR = rayDir(x1, yTop);
  let rBL = rayDir(x0, yBot);
  let rBR = rayDir(x1, yBot);
  let center = rayDir((x0 + x1) * 0.5, (yTop + yBot) * 0.5);

  let nTop = planeNormal(rTL, rTR, center);
  let nBot = planeNormal(rBL, rBR, center);
  let nLeft = planeNormal(rTL, rBL, center);
  let nRight = planeNormal(rTR, rBR, center);

  // ---- 4. 逐光源剔除 ----
  let count = u32(frame.params.x);
  var outCount = 0u;
  for (var i = 0u; i < count; i = i + 1u) {
    let lt = lights[i];
    let ltype = lt.params.x;
    var visible = true;

    if (ltype != 0.0) {
      var c : vec3<f32>;
      var r : f32;
      if (ltype == 1.0) {
        // 点光源：包围球就是衰减球
        c = (frame.view * vec4<f32>(lt.position.xyz, 1.0)).xyz;
        r = max(lt.position.w, 0.01);
      } else {
        // 聚光灯：用包住整个锥体的保守包围球
        //   球心在锥体半长处，半径取顶点/底面圆环的最远距离
        let range = max(lt.position.w, 0.01);
        let dirW = normalize(lt.direction.xyz);
        let half = 0.5 * range;
        c = (frame.view * vec4<f32>(lt.position.xyz + dirW * half, 1.0)).xyz;
        let cosO = clamp(lt.direction.w, -1.0, 1.0);
        let tanO = sqrt(max(1.0 - cosO * cosO, 0.0)) / max(abs(cosO), 1e-3);
        r = half * sqrt(1.0 + 4.0 * tanO * tanO);
      }

      // 深度区间剔除
      if (-c.z - r > maxD || -c.z + r < minD) { visible = false; }
      // 四个侧面剔除：球与平面有交集的条件是 dot(n, c) > -r
      if (visible) {
        if (dot(nTop, c) < -r || dot(nBot, c) < -r ||
            dot(nLeft, c) < -r || dot(nRight, c) < -r) {
          visible = false;
        }
      }
    }
    // 平行光（ltype == 0）无条件可见

    if (visible && outCount < stride) {
      lightIndices[tileIdx * stride + outCount] = i;
      outCount = outCount + 1u;
    }
  }

  lightGrid[tileIdx] = outCount;
}
