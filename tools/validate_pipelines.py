"""构建完整渲染管线以校验绑定组布局一致性。

着色器编译通过 ≠ 管线能创建成功。WebGPU 会在管线创建时校验：
绑定组布局与着色器 @group/@binding 声明是否匹配、采样器类型是否匹配
（如 textureSampleCompareLevel 必须配 comparison sampler）、
派生控制流中是否误用了需要 uniform 控制流的采样操作。

本脚本镜像 Renderer.ts 的布局声明，用真实 Dawn 设备逐条创建管线。
用法： python tools/validate_pipelines.py
"""

import sys
import pathlib

import wgpu

ROOT = pathlib.Path(__file__).resolve().parent.parent
SHADERS = ROOT / "src" / "Shaders"

HDR = "rgba16float"
ALBEDO = "rgba8unorm"
VEL = "rg16float"
AO = "r16float"
DEPTH = "depth32float"

# GPUShaderStage 位标志
V = 0x1  # VERTEX
F = 0x2  # FRAGMENT
C = 0x4  # COMPUTE

VERTEX_LAYOUT = {
    "array_stride": 32,
    "attributes": [
        {"shader_location": 0, "offset": 0, "format": "float32x3"},
        {"shader_location": 1, "offset": 12, "format": "float32x3"},
        {"shader_location": 2, "offset": 24, "format": "float32x2"},
    ],
}


def log(m):
    print(m, flush=True)


def vs_fullscreen(module, entry="vs"):
    return {
        "module": module,
        "entry_point": entry,
    }


def main() -> int:
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        log("[FAIL] 无可用 GPU 后端")
        return 2
    device = adapter.request_device_sync()

    def load(name):
        return device.create_shader_module(
            code=(SHADERS / name).read_text(encoding="utf-8")
        )

    # ---- 绑定组布局（与 Renderer.ts 保持一致） ----
    frame_bgl = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": V | F | C,  # 计算着色器（分块剔除）也读 Frame
                "buffer": {"type": "uniform"},
            }
        ]
    )
    object_bgl = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": V | F,
                "buffer": {"type": "uniform"},
            }
        ]
    )
    slot_bgl = device.create_bind_group_layout(
        entries=[{"binding": 0, "visibility": V, "buffer": {"type": "uniform"}}]
    )
    gbuffer_bgl = device.create_bind_group_layout(
        entries=[
            {"binding": 0, "visibility": F, "texture": {"sample_type": "float"}},
            {"binding": 1, "visibility": F, "texture": {"sample_type": "float"}},
            {"binding": 2, "visibility": F, "texture": {"sample_type": "float"}},
            {"binding": 3, "visibility": F, "texture": {"sample_type": "depth"}},
            {"binding": 4, "visibility": F, "sampler": {"type": "filtering"}},
            {"binding": 5, "visibility": F, "texture": {"sample_type": "float"}},  # AO
        ]
    )
    shadow_bgl = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": 0x2,
                "texture": {"sample_type": "depth", "view_dimension": "2d-array"},
            },
            {
                "binding": 1,
                "visibility": 0x2,
                "texture": {"sample_type": "depth", "view_dimension": "2d-array"},
            },
            {"binding": 2, "visibility": 0x2, "sampler": {"type": "comparison"}},
        ]
    )
    light_bgl = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": 0x2,
                "buffer": {"type": "read-only-storage"},
            },
            {
                "binding": 1,
                "visibility": 0x2,
                "buffer": {"type": "read-only-storage"},
            },
        ]
    )
    bloom_bgl = device.create_bind_group_layout(
        entries=[
            {"binding": 0, "visibility": F, "texture": {"sample_type": "float"}},
            {"binding": 2, "visibility": F, "buffer": {"type": "uniform"}},
        ]
    )

    # AO：深度 + 世界法线 + albedo
    ao_bgl = device.create_bind_group_layout(
        entries=[
            {"binding": 0, "visibility": F, "texture": {"sample_type": "depth"}},
            {"binding": 1, "visibility": F, "texture": {"sample_type": "float"}},
            {"binding": 2, "visibility": F, "texture": {"sample_type": "float"}},
        ]
    )
    # 分块剔除：frame + 深度 + 光源 + 计数 + 索引
    cull_bgl = device.create_bind_group_layout(
        entries=[
            {"binding": 0, "visibility": C, "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": C, "texture": {"sample_type": "depth"}},
            {"binding": 2, "visibility": C, "buffer": {"type": "read-only-storage"}},
            {"binding": 3, "visibility": C, "buffer": {"type": "storage"}},
            {"binding": 4, "visibility": C, "buffer": {"type": "storage"}},
        ]
    )
    # Forward+ group(2)：阴影 + AO + 线性采样器
    shadow_ao_bgl = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": F,
                "texture": {"sample_type": "depth", "view_dimension": "2d-array"},
            },
            {
                "binding": 1,
                "visibility": F,
                "texture": {"sample_type": "depth", "view_dimension": "2d-array"},
            },
            {"binding": 2, "visibility": F, "sampler": {"type": "comparison"}},
            {"binding": 3, "visibility": F, "texture": {"sample_type": "float"}},
            {"binding": 4, "visibility": F, "sampler": {"type": "filtering"}},
        ]
    )
    # Forward+ group(3)：光源 + 阴影矩阵 + 分块计数 + 分块索引
    light_plus_bgl = device.create_bind_group_layout(
        entries=[
            {"binding": 0, "visibility": F, "buffer": {"type": "read-only-storage"}},
            {"binding": 1, "visibility": F, "buffer": {"type": "read-only-storage"}},
            {"binding": 2, "visibility": F, "buffer": {"type": "read-only-storage"}},
            {"binding": 3, "visibility": F, "buffer": {"type": "read-only-storage"}},
        ]
    )

    frame_obj = device.create_pipeline_layout(
        bind_group_layouts=[frame_bgl, object_bgl]
    )
    shadow_layout = device.create_pipeline_layout(
        bind_group_layouts=[frame_bgl, object_bgl, slot_bgl]
    )
    # 延迟光照：Deferred 与 Tiled Deferred **共用同一个布局**。
    # 两个入口点共用 shadePixel()，其中 `if (tiled) { lightGrid[...] }` 是运行时分支，
    # 因此 lightGrid/lightIndices 从 fs 也静态可达 —— 给 fs 用只含两 buffer 的布局
    # 会被 Dawn 拒绝（ResourceBinding { group: 3, binding: 2 } is not available）。
    lit_layout = device.create_pipeline_layout(
        bind_group_layouts=[frame_bgl, gbuffer_bgl, shadow_bgl, light_plus_bgl]
    )
    # 透明：不接受阴影，只绑光源数组
    forward_layout = device.create_pipeline_layout(
        bind_group_layouts=[frame_bgl, object_bgl, light_bgl]
    )
    forward_plus_layout = device.create_pipeline_layout(
        bind_group_layouts=[frame_bgl, object_bgl, shadow_ao_bgl, light_plus_bgl]
    )
    ao_layout = device.create_pipeline_layout(bind_group_layouts=[frame_bgl, ao_bgl])
    cull_layout = device.create_pipeline_layout(bind_group_layouts=[cull_bgl])
    bloom_layout = device.create_pipeline_layout(bind_group_layouts=[bloom_bgl])

    cases = []

    # ---- G-buffer ----
    geo = load("Geometry.wgsl")
    cases.append(
        (
            "pGeometry",
            device.create_render_pipeline,
            dict(
                layout=frame_obj,
                vertex={"module": geo, "entry_point": "vs", "buffers": [VERTEX_LAYOUT]},
                fragment={
                    "module": geo,
                    "entry_point": "fs",
                    "targets": [
                        {"format": ALBEDO},
                        {"format": HDR},
                        {"format": HDR},
                        {"format": VEL},
                    ],
                },
                primitive={"topology": "triangle-list", "cull_mode": "back"},
                depth_stencil={
                    "format": DEPTH,
                    "depth_write_enabled": True,
                    "depth_compare": "less",
                },
            ),
        )
    )

    # ---- 阴影通道（逐槽位矩阵） ----
    sh = load("Shadow.wgsl")
    cases.append(
        (
            "pShadow",
            device.create_render_pipeline,
            dict(
                layout=shadow_layout,
                vertex={"module": sh, "entry_point": "vs", "buffers": [VERTEX_LAYOUT]},
                primitive={"topology": "triangle-list", "cull_mode": "back"},
                depth_stencil={
                    "format": DEPTH,
                    "depth_write_enabled": True,
                    "depth_compare": "less",
                },
            ),
        )
    )

    # ---- 延迟光照：PCSS + CSM ----
    # 两个入口点共用同一份 shadePixel()，差别只在遍历全部光源还是分块列表
    li = load("Lighting.wgsl")
    cases.append(
        (
            "pLighting",
            device.create_render_pipeline,
            dict(
                layout=lit_layout,
                vertex=vs_fullscreen(li),
                fragment={
                    "module": li,
                    "entry_point": "fs",
                    "targets": [{"format": HDR}],
                },
                primitive={"topology": "triangle-list"},
            ),
        )
    )
    cases.append(
        (
            "pLightingTiled",
            device.create_render_pipeline,
            dict(
                layout=lit_layout,
                vertex=vs_fullscreen(li),
                fragment={
                    "module": li,
                    "entry_point": "fs_tiled",
                    "targets": [{"format": HDR}],
                },
                primitive={"topology": "triangle-list"},
            ),
        )
    )

    # ---- 透明前向 ----
    tr = load("Transparent.wgsl")
    cases.append(
        (
            "pTransparent",
            device.create_render_pipeline,
            dict(
                layout=forward_layout,
                vertex={"module": tr, "entry_point": "vs", "buffers": [VERTEX_LAYOUT]},
                fragment={
                    "module": tr,
                    "entry_point": "fs",
                    "targets": [
                        {
                            "format": HDR,
                            "blend": {
                                "color": {
                                    "src_factor": "src-alpha",
                                    "dst_factor": "one-minus-src-alpha",
                                    "operation": "add",
                                },
                                "alpha": {
                                    "src_factor": "one",
                                    "dst_factor": "one-minus-src-alpha",
                                    "operation": "add",
                                },
                            },
                        }
                    ],
                },
                primitive={"topology": "triangle-list", "cull_mode": "none"},
                depth_stencil={
                    "format": DEPTH,
                    "depth_write_enabled": False,
                    "depth_compare": "less-equal",
                },
            ),
        )
    )

    # ---- Forward+：前置通道（不写世界坐标） ----
    pre = load("Prepass.wgsl")
    cases.append(
        (
            "pPrepass",
            device.create_render_pipeline,
            dict(
                layout=frame_obj,
                vertex={"module": pre, "entry_point": "vs", "buffers": [VERTEX_LAYOUT]},
                fragment={
                    "module": pre,
                    "entry_point": "fs",
                    "targets": [{"format": ALBEDO}, {"format": HDR}, {"format": VEL}],
                },
                primitive={"topology": "triangle-list", "cull_mode": "back"},
                depth_stencil={
                    "format": DEPTH,
                    "depth_write_enabled": True,
                    "depth_compare": "less",
                },
            ),
        )
    )

    # ---- Forward+：分块光源剔除（计算通道） ----
    cull = load("LightCull.wgsl")
    cases.append(
        (
            "pLightCull",
            device.create_compute_pipeline,
            dict(
                layout=cull_layout,
                compute={"module": cull, "entry_point": "cs"},
            ),
        )
    )

    # ---- Forward+ / 传统 Forward：同一个模块的两个入口点 ----
    fp = load("ForwardPlus.wgsl")
    cases.append(
        (
            "pForwardPlus",
            device.create_render_pipeline,
            dict(
                layout=forward_plus_layout,
                vertex={"module": fp, "entry_point": "vs", "buffers": [VERTEX_LAYOUT]},
                fragment={
                    "module": fp,
                    "entry_point": "fs",
                    "targets": [{"format": HDR}],
                },
                primitive={"topology": "triangle-list", "cull_mode": "back"},
                depth_stencil={
                    "format": DEPTH,
                    "depth_write_enabled": True,
                    "depth_compare": "less-equal",
                },
            ),
        )
    )
    # 传统 Forward：单趟绘制，遍历全部光源；没有前置通道所以额外输出 velocity，
    # 深度比较用 less（这一趟自己清空并写深度）
    cases.append(
        (
            "pForwardTrad",
            device.create_render_pipeline,
            dict(
                layout=forward_plus_layout,
                vertex={"module": fp, "entry_point": "vs_forward", "buffers": [VERTEX_LAYOUT]},
                fragment={
                    "module": fp,
                    "entry_point": "fs_forward",
                    "targets": [{"format": HDR}, {"format": VEL}],
                },
                primitive={"topology": "triangle-list", "cull_mode": "back"},
                depth_stencil={
                    "format": DEPTH,
                    "depth_write_enabled": True,
                    "depth_compare": "less",
                },
            ),
        )
    )

    # ---- AO：四种算法 + 关闭，共五条管线 ----
    ao = load("Ao.wgsl")
    for label, entry in (
        ("pAO:off", "fs_off"),
        ("pAO:ssao", "fs_ssao"),
        ("pAO:hbao", "fs_hbao"),
        ("pAO:hbaoplus", "fs_hbaoplus"),
        ("pAO:gtao", "fs_gtao"),
    ):
        cases.append(
            (
                label,
                device.create_render_pipeline,
                dict(
                    layout=ao_layout,
                    vertex=vs_fullscreen(ao),
                    fragment={
                        "module": ao,
                        "entry_point": entry,
                        "targets": [{"format": AO}],
                    },
                    primitive={"topology": "triangle-list"},
                ),
            )
        )

    # ---- 天空背景（仅前向路径用：前向着色不会执行到背景像素） ----
    sky_bgl = frame_bgl  # 只读 Frame
    sky_layout = device.create_pipeline_layout(bind_group_layouts=[sky_bgl])
    sky = load("Sky.wgsl")
    cases.append(
        (
            "pSky",
            device.create_render_pipeline,
            dict(
                layout=sky_layout,
                vertex=vs_fullscreen(sky),
                fragment={
                    "module": sky,
                    "entry_point": "fs",
                    "targets": [{"format": HDR}],
                },
                primitive={"topology": "triangle-list"},
            ),
        )
    )

    # ---- Bloom / TAA / 合成 ----
    # TAA 用自己的布局：current + history + velocity + sampler + uniform
    taa_bgl = device.create_bind_group_layout(
        entries=[
            {"binding": 0, "visibility": 0x2, "texture": {"sample_type": "float"}},
            {"binding": 1, "visibility": 0x2, "texture": {"sample_type": "float"}},
            {"binding": 2, "visibility": 0x2, "texture": {"sample_type": "float"}},
            {"binding": 3, "visibility": 0x2, "sampler": {"type": "filtering"}},
            {"binding": 4, "visibility": 0x2, "buffer": {"type": "uniform"}},
        ]
    )
    taa_layout = device.create_pipeline_layout(bind_group_layouts=[taa_bgl])
    taa = load("Taa.wgsl")
    cases.append(
        (
            "pTAA",
            device.create_render_pipeline,
            dict(
                layout=taa_layout,
                vertex=vs_fullscreen(taa),
                fragment={"module": taa, "entry_point": "fs", "targets": [{"format": HDR}]},
                primitive={"topology": "triangle-list"},
            ),
        )
    )

    failed = False
    for name, factory, kwargs in cases:
        try:
            factory(**kwargs)
            log(f"[ok]   {name}")
        except Exception as e:
            log(f"[FAIL] {name}: {e}")
            failed = True

    log("[result] " + ("存在管线创建失败" if failed else "全部管线创建成功"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())