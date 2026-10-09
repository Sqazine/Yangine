// WebGPU 设备/上下文初始化与少量辅助函数。

export interface GPUContext {
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
  canvas: HTMLCanvasElement;
}

export async function initWebGPU(
  canvas: HTMLCanvasElement
): Promise<GPUContext> {
  if (!("gpu" in navigator)) {
    throw new Error("此浏览器不支持 WebGPU，请使用 Chrome/Edge 113+ 或开启了 WebGPU 的浏览器。");
  }
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter) {
    throw new Error("无法获取 GPUAdapter（requestAdapter 返回 null）。");
  }
  const device = await adapter.requestDevice();
  device.lost.then((info) => {
    console.error("WebGPU device lost:", info.message);
  });

  const context = canvas.getContext("webgpu");
  if (!context) {
    throw new Error("无法获取 WebGPU canvas context。");
  }
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format,
    alphaMode: "opaque",
  });

  return { device, context, format, canvas };
}

/** 创建 GPUBuffer 并可选写入初始数据。 */
export function createBuffer(
  device: GPUDevice,
  byteLength: number,
  usage: GPUBufferUsageFlags,
  data?: Float32Array | Uint32Array | Int32Array
): GPUBuffer {
  const buf = device.createBuffer({ size: byteLength, usage });
  if (data) device.queue.writeBuffer(buf, 0, data.buffer, data.byteOffset, data.byteLength);
  return buf;
}

/** 创建二维纹理（含全部常用 usage）。 */
export function createTexture(
  device: GPUDevice,
  width: number,
  height: number,
  format: GPUTextureFormat,
  usage: GPUTextureUsageFlags
): GPUTexture {
  return device.createTexture({
    size: { width, height },
    format,
    usage,
  });
}
