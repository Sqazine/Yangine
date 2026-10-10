# Yangine 项目长期记忆

WebGPU + TypeScript + Vite 渲染引擎（`D:\.sc\Yangine`）。**不用任何第三方渲染库**，全部自研：
`Math`(mat4/vec3) · `Gpu`(设备与纹理封装) · `Scene`(几何生成 + 轨道相机) · `Lights`(光源与阴影槽位) ·
`Shaders.ts`(把 .wgsl 以 ?raw 汇总) · `Renderer`(管线与通道编排) · `Main`(面板 / HUD / 输入)。
详细过程见 `.workbuddy/memory/YYYY-MM-DD.md` 日志。

## 三条铁律
1. **WGSL 的错误 tsc 一个都发现不了**。着色器改动必须跑 `npm run verify`（含 Dawn 编译 + 真管线创建）。
   「编译通过 ≠ 管线能创建」—— 绑定组布局、采样器类型只有 createRenderPipeline 才校验。
2. `npm run verify` 全绿才算完成：typecheck → wgsl → pipelines → layout → winding → csm → lights → ao。
3. 注释写**为什么**（尤其记录踩过的坑），不写「是什么」。中文。

## 4 条可切换渲染路径（快捷键 P）
| 路径 | G-buffer / 前置 | 光源裁剪 | 光照位置 |
|---|---|---|---|
| forward | 无 | 无 | 逐物体 FS，遍历全部光源 |
| forward+ | 前置（不写世界坐标） | 分块 | 逐物体 FS，遍历分块列表 |
| deferred | 全量 G-buffer | 无 | 全屏 FS，遍历全部光源 |
| tiled-deferred | 全量 G-buffer | 分块 | 全屏 FS，遍历分块列表 |

`Lighting.wgsl` 的 `fs`/`fs_tiled`、`ForwardPlus.wgsl` 的 `fs`/`fs_forward` **共用** `shadeLight()`/`shadePixel()`，
不得各写一份光照；路径语义由 `pathHasGBuffer/pathIsTiled/pathIsForward` 三个纯函数概括。
`forward` 无 G-buffer ⇒ AO 强制走 `fs_off`（同时充当 aoTex 清零），HUD 显式标注不可用。
通道编号：1 阴影 → 2 前置/G-buffer → 3 AO → 4 分块剔除 → 5 天空(仅前向) → 6 着色 → 7 透明 → 8 TAA → 9 Bloom → 10 合成。

## 资源布局（改动须同步 `verify_layout.py`）
- `Frame`：16 成员 / 544 字节，8 个着色器声明必须一致，偏移由 `Renderer.OFF_*` 常量给出。
- `ObjectU` 5 成员 / 224B；`Light` 128B；CSM 2048²×4 层；局部阴影 1024²×12 层。
- `TILE_SIZE=16`、`LIGHTS_PER_TILE=32`、`MAX_LIGHTS=16`、`FRAME_FLOATS=136`。

## 必须成对出现的约定（历史上都曾单边改错）
- **绕序 ↔ 顶点法线**：右手法则几何法线 `cross(v1−v0, v2−v0)` 必须与顶点法线同向。
  不一致的症状是**物体从外面直接消失**（被 `cullMode:"back"` 剔掉），不是变黑 —— 极易误诊。
- **`csmSplits[c]` = 级联 c 的远界**，级联矩阵按 `[splits[c-1], splits[c]]` 构建，
  因此选集必须是「第一个满足 `v ≤ splits[c]` 的 c」。把它当近界会让远处阴影静默消失（走 `return 1.0`）。
  半影也依赖：着色器只有一组 near/far 做 linearizeDepth，需把比例 k 折进 `csmSizes[c]` 补偿。
- **深度采样/反投影**：深度值属于「像素中心那条视线」，必须先用像素中心 uv 反投影。
  用精确行进 uv 会在掠射角被深度梯度放大成假遮挡，且结果随采样步数漂移。
- **AO 仰角以切平面为基准**（`asin(dot(S−P,N)/|S−P|)`），不是视平面；距离衰减用三维距离（旋转不变）。
  越界兜底为安全值（如 `return 1.0`）的分支，必须配一条覆盖性测试 —— 否则 bug 是静默的。
- **共享函数体 + 运行时分支 ⇒ 布局不能更小**：`if (tiled) { lightGrid[...] }` 让分块缓冲从两个入口点
  都静态可达，Dawn 会拒绝更小的管线布局。Deferred 与 Tiled Deferred 必须共用同一个 litLayout。
- 天空渐变在 `Sky.wgsl`（前向路径）与 `Lighting.wgsl`（延迟路径）各有一份，必须同色（已有守卫）。

## 验证套件（`tools/`，逐条对应 `npm run verify:*`）
`validate_wgsl.py`(Dawn 编译) · `validate_pipelines.py`(真建 16 条管线) · `verify_layout.py`(结构体偏移 /
入口点 / 路径成套 / 天空一致性) · `verify_winding.mjs`(13) · `verify_csm.mjs`(49) · `verify_lights.mjs`(51) ·
`verify_ao.mjs`(53)。`.mjs` 需 esbuild 打包：`Math.ts` 用无扩展名导入，Node 直接跑解析不了。

## 环境坑
- `wgpu.utils` 不能 import：它会初始化画布后端，在无显示环境下**静默 exit 0**（无异常无输出），排查极具迷惑性。只 import `wgpu`。
- 本机 Edge headless 拿不到 GPUAdapter，改用 wgpu-py 走原生 Dawn 后端。
- 图片取证可用托管 venv（`C:/Users/Sqazi/.workbuddy/binaries/python/envs/default`）里的 PIL + numpy；
  脚本别放 `/tmp`（那里有同名 `inspect.py` 会遮蔽标准库）。
