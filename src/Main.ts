// 入口：初始化 WebGPU、构建演示场景、处理输入与渲染循环。

import { initWebGPU } from "./Gpu";
import { OrbitCamera, Renderable, Scene, makeCube, makePlane, makeSphere } from "./Scene";
import { Renderer } from "./Renderer";

const canvas = document.getElementById("app") as HTMLCanvasElement;
const hud = document.getElementById("hud") as HTMLDivElement;
const errEl = document.getElementById("err") as HTMLDivElement;

function fail(msg: string) {
  errEl.style.display = "flex";
  errEl.textContent = msg;
  hud.style.display = "none";
  console.error(msg);
}

async function main() {
  let gpu;
  try {
    gpu = await initWebGPU(canvas);
  } catch (e) {
    fail((e as Error).message);
    return;
  }
  const { device, context, format } = gpu;

  const renderer = new Renderer(device, context, format);

  // ---------- 演示场景 ----------
  const scene = new Scene();
  const cube = makeCube(1);
  const sphere = makeSphere(1, 40, 28);
  const plane = makePlane(30);

  // 地面（不透明）
  scene.add({
    mesh: plane,
    position: [0, 0, 0],
    rotationY: 0,
    scale: [1, 1, 1],
    material: {
      baseColor: [0.32, 0.34, 0.38, 1],
      metallic: 0.0,
      roughness: 0.95,
      emissive: 0.0,
      opacity: 1,
      transparent: false,
    },
  });

  // 彩色不透明立方体阵列
  const colors: [number, number, number][] = [
    [0.85, 0.25, 0.2],
    [0.2, 0.6, 0.85],
    [0.3, 0.8, 0.35],
    [0.9, 0.75, 0.2],
  ];
  colors.forEach((c, i) => {
    const ang = (i / colors.length) * Math.PI * 2;
    scene.add({
      mesh: cube,
      position: [Math.cos(ang) * 4.2, 0.6, Math.sin(ang) * 4.2],
      rotationY: ang,
      scale: [1.2, 1.2, 1.2],
      material: {
        baseColor: [c[0], c[1], c[2], 1],
        metallic: 0.1,
        roughness: 0.4,
        emissive: 0.0,
        opacity: 1,
        transparent: false,
      },
    });
  });

  // 自发光中心方块
  scene.add({
    mesh: cube,
    position: [0, 1.0, 0],
    rotationY: 0,
    scale: [0.9, 0.9, 0.9],
    material: {
      baseColor: [1.0, 0.55, 0.12, 1],
      metallic: 0.0,
      roughness: 0.5,
      emissive: 2.0,
      opacity: 1,
      transparent: false,
    },
    animate: (r, t) => {
      r.rotationY = t * 0.6;
      r.position[1] = 1.0 + Math.sin(t * 1.5) * 0.25;
    },
  });

  // 透明玻璃球（前向渲染）
  const glassColors: [number, number, number][] = [
    [0.4, 0.8, 1.0],
    [1.0, 0.5, 0.8],
    [0.7, 1.0, 0.5],
  ];
  glassColors.forEach((c, i) => {
    const ang = (i / glassColors.length) * Math.PI * 2 + 0.5;
    scene.add({
      mesh: sphere,
      position: [Math.cos(ang) * 2.4, 1.1, Math.sin(ang) * 2.4],
      rotationY: 0,
      scale: [0.9, 0.9, 0.9],
      material: {
        baseColor: [c[0], c[1], c[2], 1],
        metallic: 0.0,
        roughness: 0.1,
        emissive: 0.0,
        opacity: 0.45,
        transparent: true,
      },
      animate: (r, t) => {
        r.position[1] = 1.1 + Math.sin(t * 1.2 + i) * 0.4;
      },
    });
  });

  renderer.setScene(scene);

  const camera = new OrbitCamera();
  camera.target = [0, 1, 0];
  camera.distance = 12;
  camera.yaw = 0.6;
  camera.pitch = 0.4;

  // ---------- 尺寸 ----------
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      renderer.resize(w, h);
    }
  }
  resize();
  window.addEventListener("resize", resize);

  // ---------- 输入：轨道相机 ----------
  let dragging = false;
  let lastX = 0,
    lastY = 0;
  canvas.addEventListener("pointerdown", (e) => {
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointerup", (e) => {
    dragging = false;
    canvas.releasePointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    camera.yaw -= dx * 0.005;
    camera.pitch = Math.max(-1.3, Math.min(1.3, camera.pitch + dy * 0.005));
  });
  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      camera.distance = Math.max(4, Math.min(40, camera.distance + e.deltaY * 0.01));
    },
    { passive: false }
  );

  // ---------- 渲染循环 ----------
  let start = performance.now();
  let last = start;
  let frames = 0;
  let fps = 0;
  function frame(now: number) {
    const time = (now - start) / 1000;
    resize();
    renderer.render(camera, time);

    frames++;
    if (now - last > 500) {
      fps = Math.round((frames * 1000) / (now - last));
      frames = 0;
      last = now;
    }
    hud.textContent =
      `WebGPU 延迟渲染\n` +
      `FPS: ${fps}\n` +
      `拖拽: 旋转  滚轮: 缩放\n` +
      `通道: 阴影→G-buffer→光照→透明→TAA→Bloom→合成\n` +
      `分辨率: ${canvas.width}×${canvas.height} (DPR ${devicePixelRatio.toFixed(1)})`;

    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main();