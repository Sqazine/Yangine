// 渲染器：装配延迟渲染整条管线。
// 通道顺序：阴影 → G-buffer(不透明) → 延迟光照(HDR) → 透明前向(HDR) → TAA(乒乓) → Bloom → 合成。

import { createBuffer, createTexture } from "./Gpu";
import {
  Mat4,
  Vec3,
  halton,
  mat4Invert,
  mat4LookAt,
  mat4Multiply,
  mat4NormalFromModel,
  mat4Perspective,
} from "./Math";
import { OrbitCamera, Renderable, Scene } from "./Scene";
import {
  WGSL_BLOOM,
  WGSL_COMPOSITE,
  WGSL_GEOMETRY,
  WGSL_LIGHTING,
  WGSL_SHADOW,
  WGSL_TAA,
  WGSL_TRANSPARENT,
} from "./Shaders";

const FORMAT_ALBEDO = "rgba8unorm" as GPUTextureFormat;
const FORMAT_HDR = "rgba16float" as GPUTextureFormat;
const FORMAT_VEL = "rg16float" as GPUTextureFormat;
const FORMAT_DEPTH = "depth32float" as GPUTextureFormat;
const FORMAT_SHADOW = "depth32float" as GPUTextureFormat;

const SHADOW_SIZE = 2048;
const MIP_COUNT = 5;

const FRAME_FLOATS = 80; // 320 bytes
const OBJECT_FLOATS = 56; // 224 bytes

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
  private gbufferBGL: GPUBindGroupLayout;
  private taaBGL: GPUBindGroupLayout;
  private bloomBGL: GPUBindGroupLayout;
  private compBGL: GPUBindGroupLayout;

  // 管线
  private pGeometry!: GPURenderPipeline;
  private pShadow!: GPURenderPipeline;
  private pLighting!: GPURenderPipeline;
  private pTransparent!: GPURenderPipeline;
  private pTAA!: GPURenderPipeline;
  private pBloomPre!: GPURenderPipeline;
  private pBloomDown!: GPURenderPipeline;
  private pBloomUp!: GPURenderPipeline;
  private pComposite!: GPURenderPipeline;

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
  private shadowMap!: GPUTexture;
  private hdrScene!: GPUTexture;
  private history: GPUTexture[] = [];
  private bloomMips: GPUTexture[] = [];
  private mipRes: { w: number; h: number }[] = [];

  private lightingBindGroup!: GPUBindGroup;
  private taaBindGroup: GPUBindGroup[] = [];
  private prefilterBindGroup: GPUBindGroup[] = [];
  private downSteps: BloomStep[] = [];
  private upSteps: BloomStep[] = [];
  private compositeBindGroup: GPUBindGroup[] = [];

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
    this.frameBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ],
    });
    this.objectBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ],
    });
    this.gbufferBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "comparison" } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
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

    this.buildPipelines();
  }

  private buildPipelines() {
    const d = this.device;
    const frameObjLayout = d.createPipelineLayout({
      bindGroupLayouts: [this.frameBGL, this.objectBGL],
    });
    const lightingLayout = d.createPipelineLayout({
      bindGroupLayouts: [this.frameBGL, this.gbufferBGL],
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
      layout: frameObjLayout,
      vertex: { module: shadowModule, entryPoint: "vs", buffers: [VERTEX_LAYOUT] },
      primitive: { topology: "triangle-list", cullMode: "back" },
      depthStencil: { format: FORMAT_SHADOW, depthWriteEnabled: true, depthCompare: "less" },
    });

    const lightingModule = d.createShaderModule({ code: WGSL_LIGHTING });
    this.pLighting = d.createRenderPipeline({
      layout: lightingLayout,
      vertex: { module: lightingModule, entryPoint: "vs" },
      fragment: { module: lightingModule, entryPoint: "fs", targets: [{ format: FORMAT_HDR }] },
      primitive: { topology: "triangle-list" },
    });

    const transparentModule = d.createShaderModule({ code: WGSL_TRANSPARENT });
    this.pTransparent = d.createRenderPipeline({
      layout: frameObjLayout,
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

    const compModule = d.createShaderModule({ code: WGSL_COMPOSITE });
    this.pComposite = d.createRenderPipeline({
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
    this.history.forEach((t) => t.destroy());
    this.bloomMips.forEach((t) => t.destroy());

    const W = this.width,
      H = this.height;
    this.gAlbedo = createTexture(d, W, H, FORMAT_ALBEDO, RT);
    this.gNormal = createTexture(d, W, H, FORMAT_HDR, RT);
    this.gPosition = createTexture(d, W, H, FORMAT_HDR, RT);
    this.gVelocity = createTexture(d, W, H, FORMAT_VEL, RT);
    this.mainDepth = createTexture(d, W, H, FORMAT_DEPTH, RT);
    this.hdrScene = createTexture(d, W, H, FORMAT_HDR, RT);
    this.history = [
      createTexture(d, W, H, FORMAT_HDR, RT),
      createTexture(d, W, H, FORMAT_HDR, RT),
    ];

    this.mipRes = [];
    this.bloomMips = [];
    for (let i = 0; i < MIP_COUNT; i++) {
      const mw = Math.max(1, Math.floor(W / Math.pow(2, i + 1)));
      const mh = Math.max(1, Math.floor(H / Math.pow(2, i + 1)));
      this.mipRes.push({ w: mw, h: mh });
      this.bloomMips.push(createTexture(d, mw, mh, FORMAT_HDR, RT));
    }

    if (!this.shadowMap) {
      this.shadowMap = createTexture(
        d,
        SHADOW_SIZE,
        SHADOW_SIZE,
        FORMAT_SHADOW,
        GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
      );
    }

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
        { binding: 3, resource: this.shadowMap.createView() },
        { binding: 4, resource: this.samplerComparison },
        { binding: 5, resource: this.mainDepth.createView() },
        { binding: 6, resource: this.samplerLinear },
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

  // 方向光正交投影（WebGPU z∈[0,1]）
  private ortho(l: number, r: number, b: number, t: number, n: number, f: number): Mat4 {
    const m = new Float32Array(16);
    m[0] = 2 / (r - l);
    m[5] = 2 / (t - b);
    m[10] = 1 / (f - n);
    m[12] = -(r + l) / (r - l);
    m[13] = -(t + b) / (t - b);
    m[14] = n / (n - f);
    m[15] = 1;
    return m;
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

    // ---- 方向光矩阵 ----
    const lightDir: Vec3 = [0.5, -1.0, 0.35];
    const ld = [lightDir[0], lightDir[1], lightDir[2]];
    const ll = Math.hypot(ld[0], ld[1], ld[2]);
    const ldn: Vec3 = [ld[0] / ll, ld[1] / ll, ld[2] / ll];
    const lightEye: Vec3 = [
      camera.target[0] - ldn[0] * 25,
      camera.target[1] - ldn[1] * 25,
      camera.target[2] - ldn[2] * 25,
    ];
    const lightView = mat4LookAt(lightEye, camera.target, [0, 1, 0]);
    const lightProj = this.ortho(-16, 16, -16, 16, 0.5, 60);
    const lightVP = mat4Multiply(lightProj, lightView);

    // ---- 写入帧 uniform ----
    this.frameData.set(jitteredVP, 0);
    this.frameData.set(this.prevJitteredVP ?? jitteredVP, 16);
    this.frameData.set(lightVP, 32);
    this.frameData.set(mat4Invert(baseVP), 48);

    const eye = camera.eye();
    this.frameData[64] = eye[0];
    this.frameData[65] = eye[1];
    this.frameData[66] = eye[2];
    this.frameData[67] = 0;
    this.frameData[68] = ldn[0];
    this.frameData[69] = ldn[1];
    this.frameData[70] = ldn[2];
    this.frameData[71] = 0;
    this.frameData[72] = 1.0;
    this.frameData[73] = 0.96;
    this.frameData[74] = 0.86;
    this.frameData[75] = 3.0; // 光强
    this.frameData[76] = time;
    this.frameData[77] = 0;
    this.frameData[78] = 0;
    this.frameData[79] = 0;
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

    // ===== 1. 阴影通道 =====
    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [],
        depthStencilAttachment: {
          view: this.shadowMap.createView(),
          depthClearValue: 1.0,
          depthLoadOp: "clear",
          depthStoreOp: "store",
        },
      });
      pass.setPipeline(this.pShadow);
      pass.setBindGroup(0, this.frameBindGroup);
      for (const o of this.objects) {
        if (o.transparent) continue; // 仅不透明物体投射阴影
        pass.setBindGroup(1, o.bindGroup);
        pass.setVertexBuffer(0, o.vbo);
        pass.setIndexBuffer(o.ibo, "uint32");
        pass.drawIndexed(o.indexCount);
      }
      pass.end();
    }

    // ===== 2. G-buffer（不透明） =====
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
      for (const o of this.objects) {
        if (o.transparent) continue;
        pass.setBindGroup(1, o.bindGroup);
        pass.setVertexBuffer(0, o.vbo);
        pass.setIndexBuffer(o.ibo, "uint32");
        pass.drawIndexed(o.indexCount);
      }
      pass.end();
    }

    // ===== 3. 延迟光照 → HDR =====
    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          { view: this.hdrScene.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" },
        ],
      });
      pass.setPipeline(this.pLighting);
      pass.setBindGroup(0, this.frameBindGroup);
      pass.setBindGroup(1, this.lightingBindGroup);
      pass.draw(3);
      pass.end();
    }

    // ===== 4. 透明前向 → HDR（叠加混合） =====
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
      for (const o of this.objects) {
        if (!o.transparent) continue;
        pass.setBindGroup(1, o.bindGroup);
        pass.setVertexBuffer(0, o.vbo);
        pass.setIndexBuffer(o.ibo, "uint32");
        pass.drawIndexed(o.indexCount);
      }
      pass.end();
    }

    // ===== 5. TAA（乒乓） =====
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

    // ===== 6. Bloom =====
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

    // ===== 7. 合成 → swapchain =====
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