// 光源模型：平行光（级联阴影）/ 点光源（立方体阴影）/ 聚光灯（单张阴影）。
//
// 所有光源共享一张 GPU storage buffer；阴影贴图按「类型分组 + 槽位编号」组织：
//   - 平行光：CSM_SLOTS 张 2D 阴影图，逐级联递增 depthBias 抵消摩尔纹
//   - 点光源：每盏占 CUBE_SLOTS(=6) 张 2D 阴影图，逐面递增 depthBias
//   - 聚光灯：每盏占 1 张 2D 阴影图
// 槽位在所有光源之间连续分配，Shader 侧只需 matBase / matCount / baseBias。

import { Vec3 } from "./Math";

export const CSM_SLOTS = 4; // 平行光级联数上限
export const CUBE_SLOTS = 6; // 点光源立方体阴影面数

export const LIGHT_TYPE_DIRECTIONAL = 0;
export const LIGHT_TYPE_POINT = 1;
export const LIGHT_TYPE_SPOT = 2;

// 每个光源 128 字节：mat4(64) + 4×vec4
export const LIGHT_STRIDE_FLOATS = 32;

export type LightType =
  | typeof LIGHT_TYPE_DIRECTIONAL
  | typeof LIGHT_TYPE_POINT
  | typeof LIGHT_TYPE_SPOT;

export interface CommonLight {
  color: Vec3;
  intensity: number;
  castShadow: boolean;
}

export interface DirectionalLight extends CommonLight {
  type: typeof LIGHT_TYPE_DIRECTIONAL;
  /** 光线传播方向（从光源指向场景），无需归一化。 */
  direction: Vec3;
  /** 级联数，1..CSM_SLOTS。 */
  cascades: number;
  /** 每级联的覆盖半径倍率，1.0 为默认。 */
  cascadeBlend: number;
}

export interface PointLight extends CommonLight {
  type: typeof LIGHT_TYPE_POINT;
  position: Vec3;
  /** 衰减半径，超出后无光照。 */
  range: number;
  /** 近裁剪面，避免立方体阴影自身被裁掉。 */
  near?: number;
}

export interface SpotLight extends CommonLight {
  type: typeof LIGHT_TYPE_SPOT;
  position: Vec3;
  /** 光线传播方向。 */
  direction: Vec3;
  range: number;
  /** 内锥半角（弧度），用于锥形衰减的软过渡。 */
  innerAngle: number;
  /** 外锥半角（弧度），超出即无光照。 */
  outerAngle: number;
  near?: number;
}

export type Light = DirectionalLight | PointLight | SpotLight;

export interface LightSlot {
  /** 该光源的阴影矩阵在 storage buffer 中的起始下标。 */
  matBase: number;
  matCount: number;
  /** depthBias 基准值，逐槽递增以抑制自阴影摩尔纹。 */
  baseBias: number;
  type: LightType;
  /** 平行光为级联数；点光源为 6；聚光灯为 1。 */
  slotCount: number;
  /**
   * 在「本地阴影数组」(点光源/聚光灯) 中的起始层。
   * 平行光不使用该字段（其层号即级联索引）。
   */
  localBase: number;
}

export function isDirectional(l: Light): l is DirectionalLight {
  return l.type === LIGHT_TYPE_DIRECTIONAL;
}
export function isPoint(l: Light): l is PointLight {
  return l.type === LIGHT_TYPE_POINT;
}
export function isSpot(l: Light): l is SpotLight {
  return l.type === LIGHT_TYPE_SPOT;
}

/** 为一组光源分配连续的阴影矩阵槽位。返回与 lights 等长的分配表。 */
export function assignSlots(lights: Light[]): LightSlot[] {
  const out: LightSlot[] = [];
  let cursor = 0;
  let localCursor = 0;
  let bias = 0.0006;

  for (const l of lights) {
    if (!l.castShadow) {
      out.push({
        matBase: cursor,
        matCount: 0,
        baseBias: 0,
        type: l.type,
        slotCount: 0,
        localBase: localCursor,
      });
      continue;
    }

    let matCount: number;
    let localBase = localCursor;
    if (isDirectional(l)) {
      matCount = Math.max(1, Math.min(CSM_SLOTS, l.cascades | 0));
    } else if (isPoint(l)) {
      matCount = CUBE_SLOTS;
      localBase = localCursor;
      localCursor += CUBE_SLOTS;
    } else {
      matCount = 1;
      localBase = localCursor;
      localCursor += 1;
    }

    out.push({
      matBase: cursor,
      matCount,
      baseBias: bias,
      type: l.type,
      slotCount: matCount,
      localBase,
    });

    cursor += matCount;
    // 每盏灯之后抬升下一盏的基准偏移，缓解级联/面之间的自阴影接缝
    bias += 0.00035;
  }

  return out;
}