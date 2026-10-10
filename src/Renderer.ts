// 渲染器：四条运行时可切换的管线。
//
//                     G-buffer / 前置   光源裁剪   光照位置
//   Forward           无                无         逐物体片元着色器（全部光源）
//   Forward+          前置(深度+法线)    分块       逐物体片元着色器（分块列表）
//   Deferred          全量 G-buffer      无         全屏着色于（全部光源）
//   Tiled Deferred    全量 G-buffer      分块       全屏着色（分块列表）
//
// 通道顺序：
//   Forward         阴影 → AO(强制关闭) → 前向着色(含 velocity) → 透明 → TAA → Bloom → 合成
//   Forward+        阴影 → 前置通道 → AO → 分块剔除 → 前向着色 → 透明 → TAA → Bloom → 合成
//   Deferred        阴影 → G-buffer → AO → 延迟光照 → 透明 → TAA → Bloom → 合成
//   Tiled Deferred  阴影 → G-buffer → AO → 分块剔除 → 延迟光照(分块) → 透明 → TAA → Bloom → 合成
//
// 四条路径共用同一套阴影资源、光源数据与着色模型（见 Lighting.wgsl /
// ForwardPlus.wgsl 里被两个入口点共用的 shadeLight()），因此画质应当一致，
// 差异只体现在带宽、遍历开销，以及 Forward 因缺 G-buffer 而拿不到屏幕空间 AO。
//
// 传统 Forward 不写 G-buffer，因此 AO 没有深度/法线可读，只能强制为「关闭」。

import { createBuffer, createTexture } from "./Gpu";
import {
  Mat4,
  Vec3,
  cubeFaceView,
  halton,
  mat4Invert,
  mat4LookAt,
  mat4Multiply,
  mat4NormalFromModel,
  mat4Ortho,
  mat4Perspective,
  cascadeBounds,
  cascadeMatrix,
  vec3Normalize,
} from "./Math";
import {
  CUBE_SLOTS,
  CSM_SLOTS,
  Light,
  LightSlot,
  LIGHT_STRIDE_FLOATS,
  LIGHT_TYPE_DIRECTIONAL,
  assignSlots,
  isDirectional,
  isPoint,
} from "./Lights";
import { OrbitCamera, Renderable, Scene } from "./Scene";
import {
  WGSL_AO,
  WGSL_BLOOM,
  WGSL_COMPOSITE,
  WGSL_FORWARDPLUS,
  WGSL_GEOMETRY,
  WGSL_LIGHTCULL,
  WGSL_LIGHTING,
  WGSL_PREPASS,
  WGSL_SHADOW,
  WGSL_SKY,
  WGSL_TAA,
  WGSL_TRANSPARENT,
} from "./Shaders";

/** 可选渲染路径。 */
export type RenderPath = "forward" | "forward+" | "deferred" | "tiled-deferred";
/** 可选环境光遮蔽算法。 */
export type AoMethod = "off" | "ssao" | "hbao" | "hbaoplus" | "gtao";

/** 顺序即面板按钮顺序与快捷键 P 的循环顺序：先前向、再延迟，各自从无裁剪到分块裁剪。 */
export const RENDER_PATHS: RenderPath[] = ["forward", "forward+", "deferred", "tiled-deferred"];
export const AO_METHODS: AoMethod[] = ["off", "ssao", "hbao", "hbaoplus", "gtao"];

/** 该路径是否需要写 G-buffer / 前置通道（即是否能做屏幕空间 AO）。 */
export function pathHasGBuffer(p: RenderPath): boolean {
  return p === "forward+" || p === "deferred" || p === "tiled-deferred";
}
/** 该路径是否使用分块光源剔除。 */
export function pathIsTiled(p: RenderPath): boolean {
  return p === "forward+" || p === "tiled-deferred";
}
/** 该路径的光照是否在片元着色器里前向完成（否则为全屏延迟光照）。 */
export function pathIsForward(p: RenderPath): boolean {
  return p === "forward" || p === "forward+";
}

const FORMAT_ALBEDO = "rgba8unorm" as GPUTextureFormat;
const FORMAT_HDR = "rgba16float" as GPUTextureFormat;
const FORMAT_VEL = "rg16float" as GPUTextureFormat;
const FORMAT_AO = "r16float" as GPUTextureFormat; // 存「可见度」，1 = 无遮挡
const FORMAT_DEPTH = "depth32float" as GPUTextureFormat;
const FORMAT_SHADOW = "depth32float" as GPUTextureFormat;

const CSM_RES = 2048; // 级联阴影单级分辨率
const LOCAL_SHADOW_RES = 1024; // 点光源立方体 / 聚光灯阴影分辨率
const MAX_CSM_LAYERS = CSM_SLOTS;
const MAX_LOCAL_LAYERS = 12; // 6(立方体) + 6(备用聚光灯)
const MIP_COUNT = 5;

/** 分块光源剔除的屏幕分块边长，必须与 LightCull.wgsl 的 TILE 常量一致。 */
const TILE_SIZE = 16;
/** 每个分块最多记录的光源数（固定步长布局，不需要全局原子计数器）。 */
const LIGHTS_PER_TILE = 32;

// Frame 布局（float 偏移）：
//   viewProj 0  prevViewProj 16  invViewProj 32  view 48  proj 64  invProj 80
//   camPos 96   camFwd 100       sunDir 104      sunColor 108
//   params 112  csmSplits 116    screen 120      aoParams 124  misc 128  gridParams 132
const FRAME_FLOATS = 136;
const OFF_VIEWPROJ = 0;
const OFF_PREVVIEWPROJ = 16;
const OFF_INVVIEWPROJ = 32;
const OFF_VIEW = 48;
const OFF_PROJ = 64;
const OFF_INVPROJ = 80;
const OFF_CAMPOS = 96;
const OFF_CAMFWD = 100;
const OFF_SUNDIR = 104;
const OFF_SUNCOLOR = 108;
const OFF_PARAMS = 112;
const OFF_CSMSPLITS = 116;
const OFF_SCREEN = 120;
const OFF_AOPARAMS = 124;
const OFF_MISC = 128;
const OFF_GRIDPARAMS = 132;

const OBJECT_FLOATS = 56; // 224 字节
const MAX_LIGHTS = 16;
const SLOT_MAT_FLOATS = 16;
const MAX_SHADOW_SLOTS = MAX_CSM_LAYERS + MAX_LOCAL_LAYERS;

const VERTEX_LAYOUT: GPUVertexBufferLayout = {
  arrayStride: 32,
  attributes: [
    { shaderLocation: 0, offset: 0, format: "float32x3" },
    { shaderLocation: 1, offset: 12, format: "float32x3" },
    { shaderLocation: 2, offset: 24, format: "float32x2" },
  ],
};

interface GPUObject {
  renderable: Renderable;
  buffer: GPUBuffer;
  bindGroup: GPUBindGroup;
  vbo: GPUBuffer;
  ibo: GPUBuffer;
  indexCount: number;
  transparent: boolean;
  prevModel: Mat4;
}

interface BloomStep {
  uniform: GPUBuffer;
  bindGroup: GPUBindGroup; // 引用 srcTex + uniform
}

export class Renderer {
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private format: GPUTextureFormat;

  // 采样器
  private samplerLinear: GPUSampler;
  private samplerComparison: GPUSampler;

  // 显式绑定组布局
  private frameBGL: GPUBindGroupLayout;
  private objectBGL: GPUBindGroupLayout;
  private slotBGL: GPUBindGroupLayout;
  private gbufferBGL: GPUBindGroupLayout;
  private shadowBGL: GPUBindGroupLayout;
  private lightBGL: GPUBindGroupLayout;
  private taaBGL: GPUBindGroupLayout;
  private bloomBGL: GPUBindGroupLayout;
  private compBGL: GPUBindGroupLayout;
  // AO：深度 + 法线 + albedo
  private aoBGL: GPUBindGroupLayout;
  // 分块剔除：frame + 深度 + 光源 + 两个输出 storage
  private cullBGL: GPUBindGroupLayout;
  // Forward+：阴影资源 + AO 纹理 + 线性采样器
  private shadowAoBGL: GPUBindGroupLayout;
  // Forward+：光源 + 阴影矩阵 + 分块列表
  private lightPlusBGL: GPUBindGroupLayout;

  // 管线
  private pGeometry!: GPURenderPipeline;
  private pShadow!: GPURenderPipeline;
  private pLighting!: GPURenderPipeline;
  /** Tiled Deferred：全屏光照但只遍历分块光源列表。 */
  private pLightingTiled!: GPURenderPipeline;
  private pTransparent!: GPURenderPipeline;
  private pTAA!: GPURenderPipeline;
  private pBloomPre!: GPURenderPipeline;
  private pBloomDown!: GPURenderPipeline;
  private pBloomUp!: GPURenderPipeline;
  private pComposite!: GPURenderPipeline;
  /** 天空背景（仅前向路径需要：前向着色不会执行到背景像素）。 */
  private pSky!: GPURenderPipeline;
  // Forward+ / 传统 Forward 管线
  private pPrepass!: GPURenderPipeline;
  private pLightCull!: GPUComputePipeline;
  private pForwardPlus!: GPURenderPipeline;
  /** 传统 Forward：单趟绘制、全部光源、额外写 velocity。 */
  private pForwardTrad!: GPURenderPipeline;
  // AO：每种算法一条管线，入口点同名
  private pAO = new Map<AoMethod, GPURenderPipeline>();

  // 帧 uniform
  private frameBuffer: GPUBuffer;
  private frameData = new Float32Array(FRAME_FLOATS);
  private frameBindGroup!: GPUBindGroup;

  // 场景对象
  private scene: Scene | null = null;
  private objects: GPUObject[] = [];
  private meshCache = new Map<unknown, { vbo: GPUBuffer; ibo: GPUBuffer }>();

  // 尺寸相关资源
  private width = 1;
  private height = 1;
  private gAlbedo!: GPUTexture;
  private gNormal!: GPUTexture;
  private gPosition!: GPUTexture;
  private gVelocity!: GPUTexture;
  private mainDepth!: GPUTexture;
  private csmArray!: GPUTexture;
  private localShadowArray!: GPUTexture;
  private hdrScene!: GPUTexture;
  private aoTex!: GPUTexture;
  private history: GPUTexture[] = [];
  private bloomMips: GPUTexture[] = [];
  private mipRes: { w: number; h: number }[] = [];

  // 分块光源网格（Forward+），尺寸随分辨率变化
  private lightGrid!: GPUBuffer;
  private lightIndices!: GPUBuffer;
  private tilesX = 1;
  private tilesY = 1;

  // 光源数据
  private lights: Light[] = [];
  private lightSlots: LightSlot[] = [];
  private lightBuffer!: GPUBuffer;
  private lightData = new Float32Array(MAX_LIGHTS * LIGHT_STRIDE_FLOATS);
  private shadowMatBuffer!: GPUBuffer;
  private shadowMatData = new Float32Array(MAX_SHADOW_SLOTS * SLOT_MAT_FLOATS);
  private slotUniforms: GPUBuffer[] = [];
  private slotBindGroups: GPUBindGroup[] = [];

  private lightingBindGroup!: GPUBindGroup;
  private shadowResourceBindGroup!: GPUBindGroup;
  private lightDataBindGroup!: GPUBindGroup;
  private taaBindGroup: GPUBindGroup[] = [];
  private prefilterBindGroup: GPUBindGroup[] = [];
  private downSteps: BloomStep[] = [];
  private upSteps: BloomStep[] = [];
  private compositeBindGroup: GPUBindGroup[] = [];
  // AO / Forward+ 绑定组
  private aoBindGroup!: GPUBindGroup;
  private cullBindGroup!: GPUBindGroup;
  private shadowAoBindGroup!: GPUBindGroup;
  private lightPlusBindGroup!: GPUBindGroup;

  private taaUniform: GPUBuffer;
  private compUniform: GPUBuffer;

  // 状态
  private readIndex = 0;
  private prevJitteredVP: Mat4 | null = null;
  private needReset = true;
  private frameCount = 0;

  // 可调参数
  bloomThreshold = 1.1;
  bloomKnee = 0.6;
  bloomIntensity = 0.9;
  exposure = 1.0;
  taaBlend = 0.1; // 历史混合权重（越小越稳定但越糊）
  /** 平行光的角半径（弧度）。PCSS 半影宽度由此物理推导：tan(θ)。 */
  lightAngularSize = 0.06;
  /** 阴影浓度，1 = 全黑阴影。 */
  shadowStrength = 1.0;

  /** 当前渲染路径，运行时可切换。 */
  renderPath: RenderPath = "deferred";
  /** 当前 AO 算法，运行时可切换。 */
  aoMethod: AoMethod = "gtao";
  /** AO 采样半径（世界单位）。值越大遮蔽范围越广，但也越容易糊。 */
  aoRadius = 0.8;
  /** AO 强度：遮挡量乘以该系数后再转成可见度。 */
  aoIntensity = 1.0;
  /** AO 幂次：>1 加深暗部对比，<1 变淡。 */
  aoPower = 1.3;

  /** 分块网格尺寸（分块路径），仅用于 HUD 显示。 */
  get tileGrid(): string {
    return `${this.tilesX}×${this.tilesY} @ ${TILE_SIZE}px`;
  }
  /** 场景中参与着色的光源数（上限 MAX_LIGHTS），仅用于 HUD 显示。 */
  get lightCount(): number {
    return this.lights.length;
  }

  /** 切换渲染路径（切换后重置 TAA 历史，避免两套路径的残影互相污染）。 */
  setRenderPath(p: RenderPath) {
    if (p === this.renderPath) return;
    this.renderPath = p;
    this.needReset = true;
  }
  /** 切换 AO 算法。 */
  setAoMethod(m: AoMethod) {
    if (m === this.aoMethod) return;
    this.aoMethod = m;
    this.needReset = true;
  }
  /** 在两条路径之间循环切换。 */
  cycleRenderPath(): RenderPath {
    const i = RENDER_PATHS.indexOf(this.renderPath);
    this.setRenderPath(RENDER_PATHS[(i + 1) % RENDER_PATHS.length]);
    return this.renderPath;
  }
  /** 在五种 AO 设置之间循环切换。 */
  cycleAoMethod(): AoMethod {
    const i = AO_METHODS.indexOf(this.aoMethod);
    this.setAoMethod(AO_METHODS[(i + 1) % AO_METHODS.length]);
    return this.aoMethod;
  }

  constructor(device: GPUDevice, context: GPUCanvasContext, format: GPUTextureFormat) {
    this.device = device;
    this.context = context;
    this.format = format;

    this.samplerLinear = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });
    this.samplerComparison = device.createSampler({
      compare: "less",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
      magFilter: "linear",
      minFilter: "linear",
    });

    // ---- 绑定组布局 ----
    // COMPUTE 也要可见：分块光源剔除的计算着色器同样读 Frame
    this.frameBGL = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility:
            GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
      ],
    });
    this.objectBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ],
    });
    // 阴影通道：逐槽位传入该槽位的投影矩阵
    this.slotBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      ],
    });
    this.gbufferBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      ],
    });
    // 阴影资源：CSM 数组 + 本地阴影数组 + 比较采样器
    this.shadowBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth", viewDimension: "2d-array" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth", viewDimension: "2d-array" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "comparison" } },
      ],
    });
    // 光源数据：光源数组 + 阴影矩阵数组
    this.lightBGL = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "read-only-storage" },
        },
      ],
    });

    // AO：只需深度 + 世界法线 + albedo（GTAO 的多重散射要用 albedo 亮度）
    this.aoBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      ],
    });

    // 分块剔除：全部集中在 group(0)
    this.cullBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "depth" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    // Forward+ 的 group(2)：阴影资源 + AO + 线性采样器
    this.shadowAoBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth", viewDimension: "2d-array" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth", viewDimension: "2d-array" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "comparison" } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
      ],
    });

    // Forward+ 的 group(3)：光源 + 阴影矩阵 + 分块计数 + 分块索引
    this.lightPlusBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      ],
    });

    this.taaBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ],
    });
    this.bloomBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ],
    });
    this.compBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ],
    });

    // ---- 帧 uniform ----
    this.frameBuffer = device.createBuffer({
      size: FRAME_FLOATS * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.frameBindGroup = device.createBindGroup({
      layout: this.frameBGL,
      entries: [{ binding: 0, resource: { buffer: this.frameBuffer } }],
    });

    this.taaUniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.compUniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // ---- 光源与阴影矩阵 storage buffer ----
    this.lightBuffer = device.createBuffer({
      size: MAX_LIGHTS * LIGHT_STRIDE_FLOATS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.shadowMatBuffer = device.createBuffer({
      size: MAX_SHADOW_SLOTS * SLOT_MAT_FLOATS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // 逐槽位的投影矩阵 uniform（阴影顶点着色器用，避免占用 storage 绑定）
    for (let i = 0; i < MAX_SHADOW_SLOTS; i++) {
      const buf = device.createBuffer({
        size: SLOT_MAT_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.slotUniforms.push(buf);
      this.slotBindGroups.push(
        device.createBindGroup({
          layout: this.slotBGL,
          entries: [{ binding: 0, resource: { buffer: buf } }],
        })
      );
    }

    // 阴影贴图：CSM 与点光源/聚光灯分开存放，分辨率不同
    const shadowUsage =
      GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    this.csmArray = createTexture(
      device,
      CSM_RES,
      CSM_RES,
      FORMAT_SHADOW,
      shadowUsage,
      MAX_CSM_LAYERS
    );
    this.localShadowArray = createTexture(
      device,
      LOCAL_SHADOW_RES,
      LOCAL_SHADOW_RES,
      FORMAT_SHADOW,
      shadowUsage,
      MAX_LOCAL_LAYERS
    );

    this.shadowResourceBindGroup = device.createBindGroup({
      layout: this.shadowBGL,
      entries: [
        { binding: 0, resource: this.csmArray.createView({ dimension: "2d-array" }) },
        { binding: 1, resource: this.localShadowArray.createView({ dimension: "2d-array" }) },
        { binding: 2, resource: this.samplerComparison },
      ],
    });

    this.lightDataBindGroup = device.createBindGroup({
      layout: this.lightBGL,
      entries: [
        { binding: 0, resource: { buffer: this.lightBuffer } },
        { binding: 1, resource: { buffer: this.shadowMatBuffer } },
      ],
    });

    this.buildPipelines();
  }

  private buildPipelines() {
    const d = this.device;
    const frameObjLayout = d.createPipelineLayout({
      bindGroupLayouts: [this.frameBGL, this.objectBGL],
    });
    const shadowLayout = d.createPipelineLayout({
      bindGroupLayouts: [this.frameBGL, this.objectBGL, this.slotBGL],
    });
    // 延迟光照：group(3) 用「光源 + 阴影矩阵 + 分块计数 + 分块索引」四 buffer 的布局。
    // 注意 Deferred 与 Tiled Deferred 必须用**同一个**布局：两个入口点共用 shadePixel()，
    // 里面的 `if (tiled) { lightGrid[...] }` 是运行时分支，因此 lightGrid/lightIndices
    // 从 fs 也是静态可达的 —— 若给 fs 用只含两 buffer 的布局，管线创建会被 Dawn 直接拒绝
    // （"ResourceBinding { group: 3, binding: 2 } is not available in the pipeline layout"）。
    // 代价为零：Deferred 那一路运行时永远不读这两个 buffer。
    const litLayout = d.createPipelineLayout({
      bindGroupLayouts: [this.frameBGL, this.gbufferBGL, this.shadowBGL, this.lightPlusBGL],
    });
    // 透明：不接受阴影，因此不绑定阴影资源，只留光源数组
    const forwardLayout = d.createPipelineLayout({
      bindGroupLayouts: [this.frameBGL, this.objectBGL, this.lightBGL],
    });
    // Forward+ 前向着色
    const forwardPlusLayout = d.createPipelineLayout({
      bindGroupLayouts: [this.frameBGL, this.objectBGL, this.shadowAoBGL, this.lightPlusBGL],
    });
    // AO 全屏通道
    const aoLayout = d.createPipelineLayout({
      bindGroupLayouts: [this.frameBGL, this.aoBGL],
    });

    const geometryModule = d.createShaderModule({ code: WGSL_GEOMETRY });
    this.pGeometry = d.createRenderPipeline({
      layout: frameObjLayout,
      vertex: { module: geometryModule, entryPoint: "vs", buffers: [VERTEX_LAYOUT] },
      fragment: {
        module: geometryModule,
        entryPoint: "fs",
        targets: [
          { format: FORMAT_ALBEDO },
          { format: FORMAT_HDR },
          { format: FORMAT_HDR },
          { format: FORMAT_VEL },
        ],
      },
      primitive: { topology: "triangle-list", cullMode: "back" },
      depthStencil: { format: FORMAT_DEPTH, depthWriteEnabled: true, depthCompare: "less" },
    });

    const shadowModule = d.createShaderModule({ code: WGSL_SHADOW });
    this.pShadow = d.createRenderPipeline({
      layout: shadowLayout,
      vertex: { module: shadowModule, entryPoint: "vs", buffers: [VERTEX_LAYOUT] },
      primitive: { topology: "triangle-list", cullMode: "back" },
      depthStencil: { format: FORMAT_SHADOW, depthWriteEnabled: true, depthCompare: "less" },
    });

    // ---- 延迟光照：PCSS + CSM ----
    // 两个入口点共用同一份 shadePixel()，差别只在遍历全部光源还是分块列表
    const litModule = d.createShaderModule({ code: WGSL_LIGHTING });
    this.pLighting = d.createRenderPipeline({
      layout: litLayout,
      vertex: { module: litModule, entryPoint: "vs" },
      fragment: { module: litModule, entryPoint: "fs", targets: [{ format: FORMAT_HDR }] },
      primitive: { topology: "triangle-list" },
    });
    this.pLightingTiled = d.createRenderPipeline({
      layout: litLayout,
      vertex: { module: litModule, entryPoint: "vs" },
      fragment: { module: litModule, entryPoint: "fs_tiled", targets: [{ format: FORMAT_HDR }] },
      primitive: { topology: "triangle-list" },
    });

    const transparentModule = d.createShaderModule({ code: WGSL_TRANSPARENT });
    this.pTransparent = d.createRenderPipeline({
      layout: forwardLayout,
      vertex: { module: transparentModule, entryPoint: "vs", buffers: [VERTEX_LAYOUT] },
      fragment: {
        module: transparentModule,
        entryPoint: "fs",
        targets: [
          {
            format: FORMAT_HDR,
            blend: {
              color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
              alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
            },
          },
        ],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: FORMAT_DEPTH, depthWriteEnabled: false, depthCompare: "less-equal" },
    });

    // ---- Forward+ ----
    // 前置通道：写 albedo / normal / velocity + 深度，不写世界坐标
    const prepassModule = d.createShaderModule({ code: WGSL_PREPASS });
    this.pPrepass = d.createRenderPipeline({
      layout: frameObjLayout,
      vertex: { module: prepassModule, entryPoint: "vs", buffers: [VERTEX_LAYOUT] },
      fragment: {
        module: prepassModule,
        entryPoint: "fs",
        targets: [{ format: FORMAT_ALBEDO }, { format: FORMAT_HDR }, { format: FORMAT_VEL }],
      },
      primitive: { topology: "triangle-list", cullMode: "back" },
      depthStencil: { format: FORMAT_DEPTH, depthWriteEnabled: true, depthCompare: "less" },
    });

    // 分块光源剔除（计算通道）
    const cullModule = d.createShaderModule({ code: WGSL_LIGHTCULL });
    this.pLightCull = d.createComputePipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.cullBGL] }),
      compute: { module: cullModule, entryPoint: "cs" },
    });

    // 前向着色（Forward+）：复用前置通道的深度做 less-equal 测试，避免重复光栅化
    const fpModule = d.createShaderModule({ code: WGSL_FORWARDPLUS });
    this.pForwardPlus = d.createRenderPipeline({
      layout: forwardPlusLayout,
      vertex: { module: fpModule, entryPoint: "vs", buffers: [VERTEX_LAYOUT] },
      fragment: { module: fpModule, entryPoint: "fs", targets: [{ format: FORMAT_HDR }] },
      primitive: { topology: "triangle-list", cullMode: "back" },
      depthStencil: { format: FORMAT_DEPTH, depthWriteEnabled: true, depthCompare: "less-equal" },
    });

    // 传统 Forward：单趟绘制。没有前置通道，所以深度从 1.0 清空、比较用 less；
    // 也没有 G-buffer 帮忙写 velocity，因此这一趟自己输出 [HDR, VEL] 两个目标。
    this.pForwardTrad = d.createRenderPipeline({
      layout: forwardPlusLayout,
      vertex: { module: fpModule, entryPoint: "vs_forward", buffers: [VERTEX_LAYOUT] },
      fragment: {
        module: fpModule,
        entryPoint: "fs_forward",
        targets: [{ format: FORMAT_HDR }, { format: FORMAT_VEL }],
      },
      primitive: { topology: "triangle-list", cullMode: "back" },
      depthStencil: { format: FORMAT_DEPTH, depthWriteEnabled: true, depthCompare: "less" },
    });

    // ---- AO：四种算法各一条管线，共享顶点着色器与绑定布局 ----
    const aoModule = d.createShaderModule({ code: WGSL_AO });
    const aoEntries: [AoMethod, string][] = [
      ["off", "fs_off"],
      ["ssao", "fs_ssao"],
      ["hbao", "fs_hbao"],
      ["hbaoplus", "fs_hbaoplus"],
      ["gtao", "fs_gtao"],
    ];
    for (const [name, entry] of aoEntries) {
      this.pAO.set(
        name,
        d.createRenderPipeline({
          layout: aoLayout,
          vertex: { module: aoModule, entryPoint: "vs" },
          fragment: { module: aoModule, entryPoint: entry, targets: [{ format: FORMAT_AO }] },
          primitive: { topology: "triangle-list" },
        })
      );
    }

    const taaModule = d.createShaderModule({ code: WGSL_TAA });
    this.pTAA = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.taaBGL] }),
      vertex: { module: taaModule, entryPoint: "vs" },
      fragment: { module: taaModule, entryPoint: "fs", targets: [{ format: FORMAT_HDR }] },
      primitive: { topology: "triangle-list" },
    });

    const bloomModule = d.createShaderModule({ code: WGSL_BLOOM });
    const bloomLayout = d.createPipelineLayout({ bindGroupLayouts: [this.bloomBGL] });
    const bloomTargets = [{ format: FORMAT_HDR }];
    this.pBloomPre = d.createRenderPipeline({
      layout: bloomLayout,
      vertex: { module: bloomModule, entryPoint: "vs" },
      fragment: { module: bloomModule, entryPoint: "fs_pre", targets: bloomTargets },
      primitive: { topology: "triangle-list" },
    });
    this.pBloomDown = d.createRenderPipeline({
      layout: bloomLayout,
      vertex: { module: bloomModule, entryPoint: "vs" },
      fragment: { module: bloomModule, entryPoint: "fs_down", targets: bloomTargets },
      primitive: { topology: "triangle-list" },
    });
    this.pBloomUp = d.createRenderPipeline({
      layout: bloomLayout,
      vertex: { module: bloomModule, entryPoint: "vs" },
      fragment: {
        module: bloomModule,
        entryPoint: "fs_up",
        targets: [
          {
            format: FORMAT_HDR,
            blend: {
              color: { srcFactor: "one", dstFactor: "one", operation: "add" },
              alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
            },
          },
        ],
      },
      primitive: { topology: "triangle-list" },
    });

    // 天空背景：只读 Frame（取 screen 求 uv），全屏三角形铺满
    const skyModule = d.createShaderModule({ code: WGSL_SKY });
    this.pSky = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.frameBGL] }),
      vertex: { module: skyModule, entryPoint: "vs" },
      fragment: { module: skyModule, entryPoint: "fs", targets: [{ format: FORMAT_HDR }] },
      primitive: { topology: "triangle-list" },
    });

    const compModule = d.createShaderModule({ code: WGSL_COMPOSITE });    this.pComposite = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.compBGL] }),
      vertex: { module: compModule, entryPoint: "vs" },
      fragment: { module: compModule, entryPoint: "fs", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list" },
    });
  }

  setScene(scene: Scene) {
    this.scene = scene;
    this.meshCache.clear();
    this.objects = [];
    for (const r of scene.objects) {
      const cached = this.getMeshBuffers(r.mesh);
      const buffer = this.device.createBuffer({
        size: OBJECT_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const bindGroup = this.device.createBindGroup({
        layout: this.objectBGL,
        entries: [{ binding: 0, resource: { buffer } }],
      });
      this.objects.push({
        renderable: r,
        buffer,
        bindGroup,
        vbo: cached.vbo,
        ibo: cached.ibo,
        indexCount: r.mesh.indexCount,
        transparent: r.material.transparent,
        prevModel: this.modelOf(r),
      });
    }

    this.lights = scene.lights.slice(0, MAX_LIGHTS);
    this.lightSlots = assignSlots(this.lights);
  }

  private getMeshBuffers(mesh: { vertices: Float32Array; indices: Uint32Array }) {
    const hit = this.meshCache.get(mesh);
    if (hit) return hit;
    const vbo = createBuffer(
      this.device,
      mesh.vertices.byteLength,
      GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mesh.vertices
    );
    const ibo = createBuffer(
      this.device,
      mesh.indices.byteLength,
      GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      mesh.indices
    );
    const entry = { vbo, ibo };
    this.meshCache.set(mesh, entry);
    return entry;
  }

  private modelOf(r: Renderable): Mat4 {
    const m = mat4Multiply(
      mat4Multiply(this.translation(r.position), this.rotY(r.rotationY)),
      this.scale(r.scale)
    );
    return m;
  }
  private translation(t: Vec3): Mat4 {
    const m = new Float32Array(16);
    m[0] = m[5] = m[10] = m[15] = 1;
    m[12] = t[0];
    m[13] = t[1];
    m[14] = t[2];
    return m;
  }
  private rotY(a: number): Mat4 {
    const c = Math.cos(a),
      s = Math.sin(a);
    const m = new Float32Array(16);
    m[0] = c;
    m[2] = -s;
    m[5] = 1;
    m[8] = s;
    m[10] = c;
    m[15] = 1;
    return m;
  }
  private scale(s: Vec3): Mat4 {
    const m = new Float32Array(16);
    m[0] = s[0];
    m[5] = s[1];
    m[10] = s[2];
    m[15] = 1;
    return m;
  }

  resize(width: number, height: number) {
    this.width = Math.max(1, width | 0);
    this.height = Math.max(1, height | 0);
    const d = this.device;
    const RT = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;

    this.gAlbedo?.destroy();
    this.gNormal?.destroy();
    this.gPosition?.destroy();
    this.gVelocity?.destroy();
    this.mainDepth?.destroy();
    this.hdrScene?.destroy();
    this.aoTex?.destroy();
    this.history.forEach((t) => t.destroy());
    this.bloomMips.forEach((t) => t.destroy());
    this.lightGrid?.destroy();
    this.lightIndices?.destroy();

    const W = this.width,
      H = this.height;
    this.gAlbedo = createTexture(d, W, H, FORMAT_ALBEDO, RT);
    this.gNormal = createTexture(d, W, H, FORMAT_HDR, RT);
    this.gPosition = createTexture(d, W, H, FORMAT_HDR, RT);
    this.gVelocity = createTexture(d, W, H, FORMAT_VEL, RT);
    this.mainDepth = createTexture(d, W, H, FORMAT_DEPTH, RT);
    this.hdrScene = createTexture(d, W, H, FORMAT_HDR, RT);
    this.aoTex = createTexture(d, W, H, FORMAT_AO, RT);
    this.history = [
      createTexture(d, W, H, FORMAT_HDR, RT),
      createTexture(d, W, H, FORMAT_HDR, RT),
    ];

    // 分块光源网格：分块数向上取整，覆盖到屏幕边缘以外也无所谓（着色器已做边界钳制）
    this.tilesX = Math.max(1, Math.ceil(W / TILE_SIZE));
    this.tilesY = Math.max(1, Math.ceil(H / TILE_SIZE));
    const tileCount = this.tilesX * this.tilesY;
    this.lightGrid = d.createBuffer({
      size: tileCount * 4,
      usage: GPUBufferUsage.STORAGE,
    });
    this.lightIndices = d.createBuffer({
      size: tileCount * LIGHTS_PER_TILE * 4,
      usage: GPUBufferUsage.STORAGE,
    });

    this.mipRes = [];
    this.bloomMips = [];
    for (let i = 0; i < MIP_COUNT; i++) {
      const mw = Math.max(1, Math.floor(W / Math.pow(2, i + 1)));
      const mh = Math.max(1, Math.floor(H / Math.pow(2, i + 1)));
      this.mipRes.push({ w: mw, h: mh });
      this.bloomMips.push(createTexture(d, mw, mh, FORMAT_HDR, RT));
    }

    this.buildShadowLayerViews();
    this.buildSizeDependentBindGroups();
    this.needReset = true;
    this.readIndex = 0;
  }

  private buildSizeDependentBindGroups() {
    const d = this.device;

    this.lightingBindGroup = d.createBindGroup({
      layout: this.gbufferBGL,
      entries: [
        { binding: 0, resource: this.gAlbedo.createView() },
        { binding: 1, resource: this.gNormal.createView() },
        { binding: 2, resource: this.gPosition.createView() },
        { binding: 3, resource: this.mainDepth.createView() },
        { binding: 4, resource: this.samplerLinear },
        { binding: 5, resource: this.aoTex.createView() },
      ],
    });

    // AO：深度 + 世界法线 + albedo
    this.aoBindGroup = d.createBindGroup({
      layout: this.aoBGL,
      entries: [
        { binding: 0, resource: this.mainDepth.createView() },
        { binding: 1, resource: this.gNormal.createView() },
        { binding: 2, resource: this.gAlbedo.createView() },
      ],
    });

    // 分块剔除：frame + 深度 + 光源 + 两个输出 storage
    this.cullBindGroup = d.createBindGroup({
      layout: this.cullBGL,
      entries: [
        { binding: 0, resource: { buffer: this.frameBuffer } },
        { binding: 1, resource: this.mainDepth.createView() },
        { binding: 2, resource: { buffer: this.lightBuffer } },
        { binding: 3, resource: { buffer: this.lightGrid } },
        { binding: 4, resource: { buffer: this.lightIndices } },
      ],
    });

    // Forward+ 的阴影 + AO 资源
    this.shadowAoBindGroup = d.createBindGroup({
      layout: this.shadowAoBGL,
      entries: [
        { binding: 0, resource: this.csmArray.createView({ dimension: "2d-array" }) },
        { binding: 1, resource: this.localShadowArray.createView({ dimension: "2d-array" }) },
        { binding: 2, resource: this.samplerComparison },
        { binding: 3, resource: this.aoTex.createView() },
        { binding: 4, resource: this.samplerLinear },
      ],
    });

    // Forward+ 的光源数据 + 分块列表
    this.lightPlusBindGroup = d.createBindGroup({
      layout: this.lightPlusBGL,
      entries: [
        { binding: 0, resource: { buffer: this.lightBuffer } },
        { binding: 1, resource: { buffer: this.shadowMatBuffer } },
        { binding: 2, resource: { buffer: this.lightGrid } },
        { binding: 3, resource: { buffer: this.lightIndices } },
      ],
    });

    // TAA: 读取 read 历史，写入 write 历史
    this.taaBindGroup = [0, 1].map((read) =>
      d.createBindGroup({
        layout: this.taaBGL,
        entries: [
          { binding: 0, resource: this.hdrScene.createView() }, // current
          { binding: 1, resource: this.history[read].createView() }, // history
          { binding: 2, resource: this.gVelocity.createView() }, // velocity
          { binding: 3, resource: this.samplerLinear },
          { binding: 4, resource: { buffer: this.taaUniform } },
        ],
      })
    );

    // 合成: resolved = history[write] + mip0
    this.compositeBindGroup = [0, 1].map((resolved) =>
      d.createBindGroup({
        layout: this.compBGL,
        entries: [
          { binding: 0, resource: this.history[resolved].createView() },
          { binding: 1, resource: this.bloomMips[0].createView() },
          { binding: 2, resource: this.samplerLinear },
          { binding: 3, resource: { buffer: this.compUniform } },
        ],
      })
    );

    // Bloom 降采样 / 上采样步骤
    this.downSteps = [];
    this.upSteps = [];
    for (let k = 0; k < MIP_COUNT - 1; k++) {
      const u = d.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.writeBloomUniform(u, this.mipRes[k], this.mipRes[k + 1]);
      this.downSteps.push({
        uniform: u,
        bindGroup: d.createBindGroup({
          layout: this.bloomBGL,
          entries: [
            { binding: 0, resource: this.bloomMips[k].createView() },
            { binding: 1, resource: this.samplerLinear },
            { binding: 2, resource: { buffer: u } },
          ],
        }),
      });
      const u2 = d.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.writeBloomUniform(u2, this.mipRes[k + 1], this.mipRes[k]);
      this.upSteps.push({
        uniform: u2,
        bindGroup: d.createBindGroup({
          layout: this.bloomBGL,
          entries: [
            { binding: 0, resource: this.bloomMips[k + 1].createView() },
            { binding: 1, resource: this.samplerLinear },
            { binding: 2, resource: { buffer: u2 } },
          ],
        }),
      });
    }

    // 预过滤 uniform（源=全分辨率 HDR，目标=mip0）
    if (!this.preUniformBuffer) {
      this.preUniformBuffer = d.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }
    this.writeBloomUniform(this.preUniformBuffer, { w: this.width, h: this.height }, this.mipRes[0]);
    this.prefilterBindGroup = [0, 1].map((src) =>
      d.createBindGroup({
        layout: this.bloomBGL,
        entries: [
          { binding: 0, resource: this.history[src].createView() },
          { binding: 1, resource: this.samplerLinear },
          { binding: 2, resource: { buffer: this.preUniformBuffer! } },
        ],
      })
    );

    // 写入 Comp uniform
    const compData = new Float32Array([this.width, this.height, this.exposure, this.bloomIntensity]);
    this.device.queue.writeBuffer(this.compUniform, 0, compData);
  }

  private preUniformBuffer: GPUBuffer | null = null;

  private writeBloomUniform(buf: GPUBuffer, srcRes: { w: number; h: number }, dstRes: { w: number; h: number }) {
    const data = new Float32Array(8);
    data[0] = 1 / srcRes.w;
    data[1] = 1 / srcRes.h;
    data[2] = dstRes.w;
    data[3] = dstRes.h;
    data[4] = this.bloomThreshold;
    data[5] = this.bloomKnee;
    this.device.queue.writeBuffer(buf, 0, data);
  }

  // ---- 阴影贴图视图（按槽位类型选数组层） ----
  private csmLayerViews: GPUTextureView[] = [];
  private localLayerViews: GPUTextureView[] = [];

  private buildShadowLayerViews() {
    this.csmLayerViews = [];
    this.localLayerViews = [];
    for (let i = 0; i < MAX_CSM_LAYERS; i++) {
      this.csmLayerViews.push(
        this.csmArray.createView({ dimension: "2d", baseArrayLayer: i, arrayLayerCount: 1 })
      );
    }
    for (let i = 0; i < MAX_LOCAL_LAYERS; i++) {
      this.localLayerViews.push(
        this.localShadowArray.createView({ dimension: "2d", baseArrayLayer: i, arrayLayerCount: 1 })
      );
    }
  }

  /**
   * 解算所有光源的阴影矩阵并写入 GPU。
   * 返回本帧的 CSM 分界距离，写入帧 uniform 供着色器选级联。
   */
  private updateLightData(camera: OrbitCamera): Float32Array {
    const d = this.device;
    const eye = camera.eye();
    const fwd = camera.forward();
    const aspect = this.width / this.height;
    const tanHalf = Math.tan(camera.fov * 0.5);

    // 视锥切片包围球：在视空间求半径，构造世界空间的球心
    const sliceCenter = (n: number, f: number): { c: Vec3; r: number } => {
      const mid = (n + f) * 0.5;
      const halfH = mid * tanHalf;
      const halfW = halfH * aspect;
      const r = Math.hypot(halfW, halfH, (f - n) * 0.5);
      return {
        c: [eye[0] + fwd[0] * mid, eye[1] + fwd[1] * mid, eye[2] + fwd[2] * mid],
        r,
      };
    };

    this.shadowMatData.fill(0);
    this.lightData.fill(0);
    const csmSplits = new Float32Array(4);
    csmSplits.fill(camera.far);

    let matCursor = 0;

    for (let i = 0; i < this.lights.length; i++) {
      const l = this.lights[i];
      const slot = this.lightSlots[i];
      const o = i * LIGHT_STRIDE_FLOATS;
      const ld = this.lightData;

      ld[o + 0] = "position" in l ? l.position[0] : 0;
      ld[o + 1] = "position" in l ? l.position[1] : 0;
      ld[o + 2] = "position" in l ? l.position[2] : 0;
      ld[o + 3] = "range" in l ? l.range : 0;

      ld[o + 4] = "direction" in l ? l.direction[0] : 0;
      ld[o + 5] = "direction" in l ? l.direction[1] : -1;
      ld[o + 6] = "direction" in l ? l.direction[2] : 0;
      ld[o + 7] = 1; // cos(outer)，默认全角

      ld[o + 8] = l.color[0];
      ld[o + 9] = l.color[1];
      ld[o + 10] = l.color[2];
      ld[o + 11] = l.intensity;

      ld[o + 12] = l.type;
      ld[o + 13] = slot.matBase;
      ld[o + 14] = slot.matCount;
      ld[o + 15] = slot.baseBias;

      ld[o + 16] = 0; // lightSizeUV，稍后按类型回填
      ld[o + 17] = l.castShadow ? this.shadowStrength : 0;
      ld[o + 20] = 1; // cosInner

      let texelWorld = 1;
      let lightSizeUV = 0;
      let isOrtho = 0;

      if (isDirectional(l)) {
        // ---- 级联 ----
        const dir = vec3Normalize(l.direction);
        const count = Math.max(1, Math.min(CSM_SLOTS, l.cascades | 0));
        // bounds[c] = [splits[c-1], splits[c]]：级联矩阵按它构建，
        // 所以着色器必须按「v ≤ splits[c] 的最小 c」选级联（见 Math.selectCascade）。
        const { bounds } = cascadeBounds(camera.near * 4, camera.far, count, 0.7);

        // 正交投影下深度线性，着色器只能拿到一组 near/far 做 linearizeDepth，
        // 所以这里统一用「最远级联」的 (zNear, zFarBase)；各级联真实 far 与之的差异
        // 通过把比例 k 折进 lightSizeUV 补偿（见下）——不需要改 Light 结构体。
        const zNear = 0.1;
        const base = sliceCenter(bounds[0].n, bounds[count - 1].f);
        const zFarBase = base.r * 6 + 10;
        texelWorld = (base.r * 2) / CSM_RES;

        for (let c = 0; c < count; c++) {
          const s = sliceCenter(bounds[c].n, bounds[c].f);
          const rc = s.r * (l.cascadeBlend || 1);
          const zFarC = rc * 6 + 10;
          // texel 对齐必须用**本级联自己的** texel 尺寸：拿最远级联的粗 texel 去对齐近处级联，
          // 既对不齐近处自己的网格（抖动消不掉），又会把包围球中心推开最多半个粗 texel
          // —— 相对近处很小的半径，偏移可达半径的 1%+，足以把切片边缘挤出正交框。
          const res = cascadeMatrix(dir, s.c, rc, (rc * 2) / CSM_RES);
          this.shadowMatData.set(res.vp, matCursor * SLOT_MAT_FLOATS);
          // PCSS 半影：worldDiff × tanθ/(2r_c)。但着色器里的 worldDiff 是用共享的
          // (zNear, zFarBase) 线性化出的 ndc 差，比本级联真实 worldDiff 大
          // (zFarBase-zNear)/(zFarC-zNear) 倍，把该比例折进这里即可让各级联半影物理正确。
          // 不折的话近处级联半影会偏大近一个数量级（demo 相机下 k₀≈0.108 → 放大 9.3 倍），
          // 近处阴影被糊到 penumbraUV 上限，看起来就像「阴影没了」。
          const k = (zFarC - zNear) / (zFarBase - zNear);
          ld[o + 24 + c] = (Math.tan(this.lightAngularSize) / Math.max(rc * 2, 1e-3)) * k;
          csmSplits[c] = bounds[c].f;
          matCursor++;
        }
        // 未使用的级联槽位填远界，避免着色器读未初始化值
        for (let c = count; c < CSM_SLOTS; c++) csmSplits[c] = camera.far;

        lightSizeUV = ld[o + 24];
        isOrtho = 1;
        ld[o + 18] = zNear;
        ld[o + 19] = zFarBase;
      } else if (isPoint(l)) {
        // ---- 立方体阴影：六面各一张 ----
        const near = l.near ?? 0.08;
        const proj = mat4Perspective(Math.PI / 2, 1, near, l.range * 2 + 20);
        const dist = l.range * 2 + 20;
        for (let f = 0; f < CUBE_SLOTS; f++) {
          const view = cubeFaceView(f, l.position, dist);
          this.shadowMatData.set(mat4Multiply(proj, view), matCursor * SLOT_MAT_FLOATS);
          matCursor++;
        }
        // 面宽 = 2*tan(45°)*d = 2d，UV 上光源尺寸 ∝ 半径/(2*dist)
        lightSizeUV = this.lightAngularSize / (2 * dist);
        isOrtho = 0;
        ld[o + 18] = near;
        ld[o + 19] = dist;
        ld[o + 13] = slot.localBase;
        ld[o + 28] = 1.0;
      } else {
        // ---- 聚光灯：单张透视阴影 ----
        const near = l.near ?? 0.08;
        const outer = Math.max(l.outerAngle, 0.01);
        const fov = Math.min(2 * outer * 1.05, Math.PI * 0.95);
        const proj = mat4Perspective(fov, 1, near, l.range * 2);
        const dir = vec3Normalize(l.direction);
        const target: Vec3 = [
          l.position[0] + dir[0] * l.range,
          l.position[1] + dir[1] * l.range,
          l.position[2] + dir[2] * l.range,
        ];
        const up: Vec3 = Math.abs(dir[1]) > 0.98 ? [0, 0, 1] : [0, 1, 0];
        const view = mat4LookAt(l.position, target, up);
        this.shadowMatData.set(mat4Multiply(proj, view), matCursor * SLOT_MAT_FLOATS);
        matCursor++;

        lightSizeUV = this.lightAngularSize / (2 * Math.tan(fov * 0.5));
        isOrtho = 0;
        ld[o + 18] = near;
        ld[o + 19] = l.range * 2;
        ld[o + 7] = Math.cos(l.outerAngle);
        ld[o + 20] = Math.cos(l.innerAngle);
        ld[o + 13] = slot.localBase;
        ld[o + 28] = 1.0;
      }

      ld[o + 16] = lightSizeUV;
      ld[o + 21] = isOrtho;
      ld[o + 22] = texelWorld;
      ld[o + 29] = "range" in l ? l.range : 0;
    }

    d.queue.writeBuffer(this.lightBuffer, 0, this.lightData);
    d.queue.writeBuffer(this.shadowMatBuffer, 0, this.shadowMatData);

    // 阴影顶点着色器逐槽位读取：把 storage 中的矩阵同步到 slot uniform
    for (let s = 0; s < MAX_SHADOW_SLOTS; s++) {
      d.queue.writeBuffer(
        this.slotUniforms[s],
        0,
        this.shadowMatData,
        s * SLOT_MAT_FLOATS,
        SLOT_MAT_FLOATS
      );
    }

    return csmSplits;
  }

  render(camera: OrbitCamera, time: number) {
    if (!this.scene) return;
    const d = this.device;
    const W = this.width,
      H = this.height;
    const aspect = W / H;
    camera.aspect = aspect;

    // ---- 相机矩阵 + 子像素抖动 ----
    const baseVP = camera.viewProj();
    const jx = (halton(this.frameCount + 1, 2) - 0.5) * (2 / W);
    const jy = (halton(this.frameCount + 1, 3) - 0.5) * (2 / H);
    const proj = mat4Perspective(camera.fov, aspect, camera.near, camera.far);
    proj[8] -= jx;
    proj[9] -= jy;
    const view = mat4LookAt(camera.eye(), camera.target, [0, 1, 0]);
    const jitteredVP = mat4Multiply(proj, view);

    // ---- 光源矩阵与 CSM 分界 ----
    const csmSplits = this.updateLightData(camera);

    // ---- 写入帧 uniform ----
    // 布局见文件顶部的常量表。view / proj / invProj 一律用「抖动后」的投影矩阵：
    // 深度缓冲是抖动 VP 光栅化出来的，AO 的「视空间重建 ↔ 屏幕 UV」往返必须用同一组矩阵，
    // 否则采样点会与深度值错开半个像素，出现沿物体边缘的漏光。
    this.frameData.set(jitteredVP, OFF_VIEWPROJ);
    this.frameData.set(this.prevJitteredVP ?? jitteredVP, OFF_PREVVIEWPROJ);
    this.frameData.set(mat4Invert(baseVP), OFF_INVVIEWPROJ);
    this.frameData.set(view, OFF_VIEW);
    this.frameData.set(proj, OFF_PROJ);
    this.frameData.set(mat4Invert(proj), OFF_INVPROJ);

    const eye = camera.eye();
    this.frameData[OFF_CAMPOS + 0] = eye[0];
    this.frameData[OFF_CAMPOS + 1] = eye[1];
    this.frameData[OFF_CAMPOS + 2] = eye[2];
    this.frameData[OFF_CAMPOS + 3] = 1;

    const fwd = camera.forward();
    this.frameData[OFF_CAMFWD + 0] = fwd[0];
    this.frameData[OFF_CAMFWD + 1] = fwd[1];
    this.frameData[OFF_CAMFWD + 2] = fwd[2];
    this.frameData[OFF_CAMFWD + 3] = 0;

    // 主方向光：供透明着色器等旧路径使用，取第一盏平行光
    const dirLight = this.lights.find((l) => l.type === LIGHT_TYPE_DIRECTIONAL);
    if (dirLight && "direction" in dirLight) {
      const n = vec3Normalize(dirLight.direction);
      this.frameData[OFF_SUNDIR + 0] = n[0];
      this.frameData[OFF_SUNDIR + 1] = n[1];
      this.frameData[OFF_SUNDIR + 2] = n[2];
      this.frameData[OFF_SUNDIR + 3] = 0;
      this.frameData[OFF_SUNCOLOR + 0] = dirLight.color[0];
      this.frameData[OFF_SUNCOLOR + 1] = dirLight.color[1];
      this.frameData[OFF_SUNCOLOR + 2] = dirLight.color[2];
      this.frameData[OFF_SUNCOLOR + 3] = dirLight.intensity;
    }

    this.frameData[OFF_PARAMS + 0] = this.lights.length;
    this.frameData[OFF_PARAMS + 1] = camera.near;
    this.frameData[OFF_PARAMS + 2] = camera.far;
    this.frameData[OFF_PARAMS + 3] = time;
    this.frameData.set(csmSplits, OFF_CSMSPLITS);
    this.frameData[OFF_SCREEN + 0] = W;
    this.frameData[OFF_SCREEN + 1] = H;
    this.frameData[OFF_SCREEN + 2] = 1 / W;
    this.frameData[OFF_SCREEN + 3] = 1 / H;

    // AO 参数：噪声种子逐帧变化，采样噪声被 TAA 摊平而不是固定成图案
    this.frameData[OFF_AOPARAMS + 0] = this.aoRadius;
    this.frameData[OFF_AOPARAMS + 1] = this.aoIntensity;
    this.frameData[OFF_AOPARAMS + 2] = this.aoPower;
    this.frameData[OFF_AOPARAMS + 3] = (this.frameCount % 64) * 0.618;

    this.frameData[OFF_MISC + 0] = RENDER_PATHS.indexOf(this.renderPath);
    this.frameData[OFF_MISC + 1] = AO_METHODS.indexOf(this.aoMethod);
    this.frameData[OFF_MISC + 2] = TILE_SIZE;
    this.frameData[OFF_MISC + 3] = TILE_SIZE;

    this.frameData[OFF_GRIDPARAMS + 0] = this.tilesX;
    this.frameData[OFF_GRIDPARAMS + 1] = this.tilesY;
    this.frameData[OFF_GRIDPARAMS + 2] = LIGHTS_PER_TILE;
    this.frameData[OFF_GRIDPARAMS + 3] = 0;
    d.queue.writeBuffer(this.frameBuffer, 0, this.frameData);

    // TAA uniform
    const reset = this.needReset ? 1 : 0;
    const taaData = new Float32Array([W, H, this.taaBlend, reset]);
    d.queue.writeBuffer(this.taaUniform, 0, taaData);

    // ---- 写入物体 uniform（并更新动画/prevModel） ----
    for (const o of this.objects) {
      if (o.renderable.animate) o.renderable.animate(o.renderable, time);
      const model = this.modelOf(o.renderable);
      const normalM = mat4NormalFromModel(model);
      const od = new Float32Array(OBJECT_FLOATS);
      od.set(model, 0);
      od.set(o.prevModel, 16);
      od.set(normalM, 32);
      const bc = o.renderable.material.baseColor;
      od[48] = bc[0];
      od[49] = bc[1];
      od[50] = bc[2];
      od[51] = bc[3];
      od[52] = o.renderable.material.metallic;
      od[53] = o.renderable.material.roughness;
      od[54] = o.renderable.material.emissive;
      od[55] = o.renderable.material.opacity;
      d.queue.writeBuffer(o.buffer, 0, od);
    }

    const encoder = d.createCommandEncoder();

    // ===== 1. 阴影通道：逐槽位渲染 =====
    // CSM 槽位写 csmArray 对应层，点光源/聚光灯槽位写 localShadowArray 对应层。
    // 两类贴图分辨率不同（2048 vs 1024），但光源的 matBase 已按类型分好，
    // 因此这里只需按类型选择目标视图。
    for (let i = 0; i < this.lights.length; i++) {
      const l = this.lights[i];
      const slot = this.lightSlots[i];
      if (slot.matCount === 0) continue;

      const isDir = l.type === LIGHT_TYPE_DIRECTIONAL;
      const views = isDir ? this.csmLayerViews : this.localLayerViews;
      // 平行光的层号即级联索引；点/聚光灯使用其在本地数组中的基址
      const localBase = isDir ? 0 : slot.localBase;

      for (let s = 0; s < slot.matCount; s++) {
        const layer = isDir ? s : localBase + s;
        const view = views[layer];
        if (!view) continue;

        const pass = encoder.beginRenderPass({
          colorAttachments: [],
          depthStencilAttachment: {
            view,
            depthClearValue: 1.0,
            depthLoadOp: "clear",
            depthStoreOp: "store",
          },
        });
        pass.setPipeline(this.pShadow);
        pass.setBindGroup(0, this.frameBindGroup);
        pass.setBindGroup(2, this.slotBindGroups[slot.matBase + s]);
        for (const o of this.objects) {
          if (o.transparent) continue; // 仅不透明物体投射阴影
          pass.setBindGroup(1, o.bindGroup);
          pass.setVertexBuffer(0, o.vbo);
          pass.setIndexBuffer(o.ibo, "uint32");
          pass.drawIndexed(o.indexCount);
        }
        pass.end();
      }
    }

    // 四条路径在几何阶段分三种走法：
    //   全量 G-buffer（Deferred / Tiled Deferred）
    //   前置通道，只写深度+法线+albedo+velocity，不写世界坐标（Forward+）
    //   什么都不写，几何在着色那一趟才光栅化（传统 Forward）
    const hasGbuffer = pathHasGBuffer(this.renderPath);
    const tiled = pathIsTiled(this.renderPath);
    const drawOpaque = (pass: GPURenderPassEncoder) => {
      for (const o of this.objects) {
        if (o.transparent) continue;
        pass.setBindGroup(1, o.bindGroup);
        pass.setVertexBuffer(0, o.vbo);
        pass.setIndexBuffer(o.ibo, "uint32");
        pass.drawIndexed(o.indexCount);
      }
    };

    const tiledDeferredPath = this.renderPath === "tiled-deferred";
    const forwardPlusPath = this.renderPath === "forward+";

    if (forwardPlusPath) {
      // ===== 2a. 前置通道：albedo / normal / velocity + 深度（不写世界坐标） =====
      {
        const pass = encoder.beginRenderPass({
          colorAttachments: [
            { view: this.gAlbedo.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
            { view: this.gNormal.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
            { view: this.gVelocity.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
          ],
          depthStencilAttachment: {
            view: this.mainDepth.createView(),
            depthClearValue: 1.0,
            depthLoadOp: "clear",
            depthStoreOp: "store",
          },
        });
        pass.setPipeline(this.pPrepass);
        pass.setBindGroup(0, this.frameBindGroup);
        drawOpaque(pass);
        pass.end();
      }
    } else if (hasGbuffer) {
      // ===== 2b. G-buffer（不透明） =====
      {
        const pass = encoder.beginRenderPass({
          colorAttachments: [
            { view: this.gAlbedo.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
            { view: this.gNormal.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
            { view: this.gPosition.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
            { view: this.gVelocity.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
          ],
          depthStencilAttachment: {
            view: this.mainDepth.createView(),
            depthClearValue: 1.0,
            depthLoadOp: "clear",
            depthStoreOp: "store",
          },
        });
        pass.setPipeline(this.pGeometry);
        pass.setBindGroup(0, this.frameBindGroup);
        drawOpaque(pass);
        pass.end();
      }
    }
    // 传统 Forward 在此不做几何通道：它没有 G-buffer，几何只在第 5c 步光栅化一次。

    // ===== 3. AO → aoTex（存可见度，1 = 无遮挡） =====
    // 前三条路径在此之前的输出一致（albedo / world normal / depth），
    // 因此 AO 是路径无关的，同样的参数在它们下面得到同样的结果。
    // 传统 Forward 没有 G-buffer，屏幕空间 AO 无从下手（无深度/法线可读），
    // 因此强制用 fs_off（写全可见）。这一趟仍然要跑：它也是 aoTex 的清零动作。
    {
      const aoEntry: AoMethod = this.renderPath === "forward" ? "off" : this.aoMethod;
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          { view: this.aoTex.createView(), clearValue: { r: 1, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" },
        ],
      });
      pass.setPipeline(this.pAO.get(aoEntry)!);
      pass.setBindGroup(0, this.frameBindGroup);
      pass.setBindGroup(1, this.aoBindGroup);
      pass.draw(3);
      pass.end();
    }

    // ===== 4. 分块光源剔除（只有分块路径需要） =====
    if (tiled) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pLightCull);
      pass.setBindGroup(0, this.cullBindGroup);
      pass.dispatchWorkgroups(this.tilesX, this.tilesY, 1);
      pass.end();
    }

    // ===== 5. 天空背景（仅前向路径） =====
    // 前向着色只作用于几何体覆盖的像素，背景像素不会被执行到，
    // 因此天空得先铺一趟再让几何体覆盖上去。延迟路径不需要：它的全屏着色器
    // 本来就在 depth >= 1 处返回天空（见 Lighting.wgsl 的 shadePixel）。
    if (pathIsForward(this.renderPath)) {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          { view: this.hdrScene.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" },
        ],
      });
      pass.setPipeline(this.pSky);
      pass.setBindGroup(0, this.frameBindGroup);
      pass.draw(3);
      pass.end();
    }

    if (forwardPlusPath) {
      // ===== 6a. Forward+ 前向着色 → HDR =====
      // 复用前置通道的深度做 less-equal 测试：像素级零重绘，只对着色一次
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          { view: this.hdrScene.createView(), loadOp: "load", storeOp: "store" },
        ],
        depthStencilAttachment: {
          view: this.mainDepth.createView(),
          depthLoadOp: "load",
          depthStoreOp: "store",
        },
      });
      pass.setPipeline(this.pForwardPlus);
      pass.setBindGroup(0, this.frameBindGroup);
      pass.setBindGroup(2, this.shadowAoBindGroup);
      pass.setBindGroup(3, this.lightPlusBindGroup);
      drawOpaque(pass);
      pass.end();
    } else if (this.renderPath === "forward") {
      // ===== 6b. 传统 Forward：单趟 几何 + 光照 + velocity =====
      // 深度在这里从 1.0 清空并写入；没有前置通道，所以像素按绘制顺序着色，
      // 多光源遍历、阴影解算全部发生在这一个 fragment shader 里。
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          { view: this.hdrScene.createView(), loadOp: "load", storeOp: "store" },
          { view: this.gVelocity.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
        ],
        depthStencilAttachment: {
          view: this.mainDepth.createView(),
          depthClearValue: 1.0,
          depthLoadOp: "clear",
          depthStoreOp: "store",
        },
      });
      pass.setPipeline(this.pForwardTrad);
      pass.setBindGroup(0, this.frameBindGroup);
      pass.setBindGroup(2, this.shadowAoBindGroup);
      pass.setBindGroup(3, this.lightPlusBindGroup);
      drawOpaque(pass);
      pass.end();
    } else {
      // ===== 6c. 延迟光照 → HDR（Tiled Deferred 只多一层分块光源列表） =====
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          { view: this.hdrScene.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" },
        ],
      });
      pass.setPipeline(tiledDeferredPath ? this.pLightingTiled : this.pLighting);
      pass.setBindGroup(0, this.frameBindGroup);
      pass.setBindGroup(1, this.lightingBindGroup);
      pass.setBindGroup(2, this.shadowResourceBindGroup);
      // 两条延迟路径的 group(3) 布局相同（见 litLayout 的说明），故共用同一个绑定组
      pass.setBindGroup(3, this.lightPlusBindGroup);
      pass.draw(3);
      pass.end();
    }

    // ===== 7. 透明前向 → HDR（叠加混合，不接受阴影） =====
    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          { view: this.hdrScene.createView(), loadOp: "load", storeOp: "store" },
        ],
        depthStencilAttachment: {
          view: this.mainDepth.createView(),
          depthLoadOp: "load",
          depthStoreOp: "store",
          depthReadOnly: false,
        },
      });
      pass.setPipeline(this.pTransparent);
      pass.setBindGroup(0, this.frameBindGroup);
      pass.setBindGroup(2, this.lightDataBindGroup);
      for (const o of this.objects) {
        if (!o.transparent) continue;
        pass.setBindGroup(1, o.bindGroup);
        pass.setVertexBuffer(0, o.vbo);
        pass.setIndexBuffer(o.ibo, "uint32");
        pass.drawIndexed(o.indexCount);
      }
      pass.end();
    }

    // ===== 8. TAA（乒乓） =====
    const write = 1 - this.readIndex;
    const read = this.readIndex;
    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          { view: this.history[write].createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" },
        ],
      });
      pass.setPipeline(this.pTAA);
      pass.setBindGroup(0, this.taaBindGroup[read]);
      pass.draw(3);
      pass.end();
    }

    // ===== 9. Bloom =====
    // prefilter: history[write] → mip0
    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          { view: this.bloomMips[0].createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" },
        ],
      });
      pass.setPipeline(this.pBloomPre);
      pass.setBindGroup(0, this.prefilterBindGroup[write]);
      pass.draw(3);
      pass.end();
    }
    // downsample: mip[k] → mip[k+1]
    for (let k = 0; k < MIP_COUNT - 1; k++) {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          { view: this.bloomMips[k + 1].createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" },
        ],
      });
      pass.setPipeline(this.pBloomDown);
      pass.setBindGroup(0, this.downSteps[k].bindGroup);
      pass.draw(3);
      pass.end();
    }
    // upsample: mip[k+1] → mip[k]（加法混合，从粗到细）
    for (let k = MIP_COUNT - 2; k >= 0; k--) {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          { view: this.bloomMips[k].createView(), loadOp: "load", storeOp: "store" },
        ],
      });
      pass.setPipeline(this.pBloomUp);
      pass.setBindGroup(0, this.upSteps[k].bindGroup);
      pass.draw(3);
      pass.end();
    }

    // ===== 10. 合成 → swapchain =====
    {
      const swapView = this.context.getCurrentTexture().createView();
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          { view: swapView, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" },
        ],
      });
      pass.setPipeline(this.pComposite);
      pass.setBindGroup(0, this.compositeBindGroup[write]);
      pass.draw(3);
      pass.end();
    }

    d.queue.submit([encoder.finish()]);

    // ---- 状态推进 ----
    this.readIndex = write;
    this.prevJitteredVP = jitteredVP;
    for (const o of this.objects) o.prevModel = this.modelOf(o.renderable);
    this.needReset = false;
    this.frameCount++;
  }
}