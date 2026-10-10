// 入口：初始化 WebGPU、构建演示场景、处理输入与渲染循环。

import { initWebGPU } from "./Gpu";
import { OrbitCamera, Scene, makeCube, makePlane, makeSphere } from "./Scene";
import {
  AO_METHODS,
  AoMethod,
  RENDER_PATHS,
  RenderPath,
  Renderer,
} from "./Renderer";
import {
  LIGHT_TYPE_DIRECTIONAL,
  LIGHT_TYPE_POINT,
  LIGHT_TYPE_SPOT,
} from "./Lights";

const canvas = document.getElementById("app") as HTMLCanvasElement;
const hud = document.getElementById("hud") as HTMLDivElement;
const panel = document.getElementById("panel") as HTMLDivElement;
const errEl = document.getElementById("err") as HTMLDivElement;

const AO_LABEL: Record<AoMethod, string> = {
  off: "关闭",
  ssao: "SSAO",
  hbao: "HBAO",
  hbaoplus: "HBAO+",
  gtao: "GTAO",
};
const PATH_LABEL: Record<RenderPath, string> = {
  forward: "Forward",
  "forward+": "Forward+",
  deferred: "Deferred",
  "tiled-deferred": "Tiled Deferred",
};
/** 每条路径的通道顺序（HUD 显示用）。 */
const PATH_ORDER: Record<RenderPath, string> = {
  forward: "阴影→AO(强制关)→前向着色(含velocity)→透明→TAA→Bloom→合成",
  "forward+": "阴影→前置通道→AO→分块剔除→前向着色→透明→TAA→Bloom→合成",
  deferred: "阴影→G-buffer→AO→延迟光照→透明→TAA→Bloom→合成",
  "tiled-deferred": "阴影→G-buffer→AO→分块剔除→延迟光照(分块)→透明→TAA→Bloom→合成",
};

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
  const plane = makePlane(60);

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

  // ---------- 光源：平行光(CSM) + 点光源(Cube) + 聚光灯 ----------
  // 地面增大到 60 以覆盖 4 级级联的远端，保证远处仍有阴影细节
  scene.addLight({
    type: LIGHT_TYPE_DIRECTIONAL,
    color: [1.0, 0.96, 0.86],
    intensity: 3.0,
    castShadow: true,
    direction: [0.55, -1.0, 0.4],
    cascades: 4,
    cascadeBlend: 1.0,
  });

  scene.addLight({
    type: LIGHT_TYPE_POINT,
    color: [1.0, 0.62, 0.28],
    intensity: 22.0,
    castShadow: true,
    position: [-7.5, 6.0, -6.5],
    range: 22.0,
  });

  scene.addLight({
    type: LIGHT_TYPE_SPOT,
    color: [0.45, 0.72, 1.0],
    intensity: 26.0,
    castShadow: true,
    position: [7.0, 8.5, 6.0],
    direction: [-0.55, -1.0, -0.5],
    range: 26.0,
    innerAngle: (18 * Math.PI) / 180,
    outerAngle: (34 * Math.PI) / 180,
  });

  renderer.setScene(scene);

  // ---------- 切换面板：渲染路径 + AO 算法 ----------
  const pathButtons = new Map<RenderPath, HTMLButtonElement>();
  const aoButtons = new Map<AoMethod, HTMLButtonElement>();

  function syncPanel() {
    pathButtons.forEach((b, k) => b.classList.toggle("on", k === renderer.renderPath));
    aoButtons.forEach((b, k) => b.classList.toggle("on", k === renderer.aoMethod));
  }

  /** 建一个「标题 + 若干按钮」的分组。 */
  function buildGroup<T>(title: string, items: T[], label: (v: T) => string): Map<T, HTMLButtonElement> {
    const g = document.createElement("div");
    g.className = "grp";
    const l = document.createElement("span");
    l.className = "lbl";
    l.textContent = title;
    g.appendChild(l);
    panel.appendChild(g);

    const map = new Map<T, HTMLButtonElement>();
    for (const v of items) {
      const b = document.createElement("button");
      b.textContent = label(v);
      g.appendChild(b);
      map.set(v, b);
    }
    return map;
  }

  const pg = buildGroup("渲染路径（快捷键 P）", RENDER_PATHS, (p) => PATH_LABEL[p]);
  pg.forEach((b, p) => {
    b.onclick = () => {
      renderer.setRenderPath(p);
      syncPanel();
    };
    pathButtons.set(p, b);
  });

  const ag = buildGroup("环境光遮蔽（快捷键 O）", AO_METHODS, (m) => AO_LABEL[m]);
  ag.forEach((b, m) => {
    b.onclick = () => {
      renderer.setAoMethod(m);
      syncPanel();
    };
    aoButtons.set(m, b);
  });

  {
    const g = document.createElement("div");
    g.className = "grp";
    const l = document.createElement("span");
    l.className = "lbl";
    l.textContent = "快捷键";
    g.appendChild(l);
    const t = document.createElement("span");
    t.textContent = "P 路径 · O 算法 · [ ] 半径 · - = 强度";
    g.appendChild(t);
    panel.appendChild(g);
  }

  syncPanel();

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

  // ---------- 快捷键 ----------
  window.addEventListener("keydown", (e) => {
    if (e.repeat) return;
    const k = e.key.toLowerCase();
    if (k === "p") {
      renderer.cycleRenderPath();
      syncPanel();
    } else if (k === "o") {
      renderer.cycleAoMethod();
      syncPanel();
    } else if (k === "[" || k === "]") {
      // AO 半径：0.1 ~ 4.0（世界单位）
      const step = k === "]" ? 1.12 : 1 / 1.12;
      renderer.aoRadius = Math.min(4, Math.max(0.1, renderer.aoRadius * step));
    } else if (k === "-" || k === "=") {
      // AO 强度：0 ~ 2
      const step = k === "=" ? 0.08 : -0.08;
      renderer.aoIntensity = Math.min(2, Math.max(0, renderer.aoIntensity + step));
    }
  });

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
    const path = renderer.renderPath;
    // 传统 Forward 没有 G-buffer，屏幕空间 AO 无深度/法线可读，AO 被强制关闭
    const aoText =
      path === "forward"
        ? `AO: 不可用（Forward 无 G-buffer）`
        : `AO: ${AO_LABEL[renderer.aoMethod]}  r=${renderer.aoRadius.toFixed(2)}  强度=${renderer.aoIntensity.toFixed(2)}`;
    hud.textContent =
      `WebGPU ${PATH_LABEL[path]} 渲染\n` +
      `FPS: ${fps}\n` +
      `拖拽: 旋转  滚轮: 缩放\n` +
      `通道: ${PATH_ORDER[path]}\n` +
      aoText +
      `\n` +
      `阴影: 平行光 4 级 CSM + 点光源 Cube + 聚光灯，均为 PCSS（半透不接收）\n` +
      `光源角半径: ${(renderer.lightAngularSize * 1000).toFixed(0)} mrad\n` +
      `光源: ${renderer.lightCount} 盏` +
      (path === "forward" ? "（逐片元全遍历）" : path === "forward+" || path === "tiled-deferred" ? `（分块 ${renderer.tileGrid}）` : "（全屏全遍历）") +
      `\n` +
      `分辨率: ${canvas.width}×${canvas.height} (DPR ${devicePixelRatio.toFixed(1)})`;

    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main();