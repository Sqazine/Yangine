"""校验 WGSL 结构体布局与 Renderer.ts 中的 CPU 侧偏移常量一致。

着色器编译通过 ≠ 数据解读正确。WebGPU 不提供任何反射机制：
uniform buffer 就是一段字节，着色器按声明顺序算偏移，CPU 按手写常量写字节。
两边只要有一处不同步（新增字段、调换顺序、改类型），画面会静默错乱而没有任何报错。
本脚本把 WGSL 的布局规则实现一遍，独立算出偏移，再与 Renderer.ts 的常量逐一比对。

用法： python tools/verify_layout.py
退出码： 0 = 一致，1 = 存在不一致，2 = 文件缺失
"""

import sys
import re
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SHADERS = ROOT / "src" / "Shaders"
RENDERER = ROOT / "src" / "Renderer.ts"
LIGHTS_TS = ROOT / "src" / "Lights.ts"

# WGSL 在 uniform / storage 地址空间下的 (字节大小, 字节对齐)
WGSL_SIZES = {
    "mat4x4<f32>": (64, 16),
    "mat3x3<f32>": (48, 16),
    "mat2x2<f32>": (16, 8),
    "vec4<f32>": (16, 16),
    "vec3<f32>": (12, 16),
    "vec2<f32>": (8, 8),
    "f32": (4, 4),
    "u32": (4, 4),
    "i32": (4, 4),
}


def log(m):
    print(m, flush=True)


def align_up(x, a):
    return (x + a - 1) // a * a


def parse_struct(text, name):
    """从 WGSL 源码中取出 `struct <name> { ... };` 的成员 [(字段名, 类型)]。"""
    m = re.search(r"struct\s+" + name + r"\s*\{(.*?)\n\};", text, re.S)
    if not m:
        return None
    body = m.group(1)
    # 去掉整行注释与行尾注释
    body = re.sub(r"//[^\n]*", "", body)
    # WGSL 结构体成员以逗号分隔（本项目的结构体不含带逗号的类型）
    out = []
    for stmt in body.split(","):
        stmt = stmt.strip()
        if not stmt:
            continue
        mm = re.match(r"(\w+)\s*:\s*([A-Za-z_][\w<>]*)\s*$", stmt)
        if not mm:
            raise SyntaxError(f"无法解析成员声明：{stmt!r}")
        out.append((mm.group(1), mm.group(2)))
    return out


def layout_of(members):
    """按 WGSL 规则算出每个成员的字节偏移，返回 [(字段名, 类型, 字节偏移)] 与总大小。"""
    off = 0
    max_align = 1
    rows = []
    for fname, ftype in members:
        if ftype not in WGSL_SIZES:
            raise KeyError(f"未知类型 {ftype}（请补充到 WGSL_SIZES）")
        size, align = WGSL_SIZES[ftype]
        max_align = max(max_align, align)
        off = align_up(off, align)
        rows.append((fname, ftype, off))
        off += size
    return rows, align_up(off, max_align)


def ts_const(text, name):
    m = re.search(r"\b" + name + r"\s*=\s*(-?\d+)", text)
    return int(m.group(1)) if m else None


def fn_body(text, header):
    """取出以 header 开头的函数的完整函数体（按大括号配对，避免切在 `-> @location(0)` 上）。"""
    i = text.find(header)
    if i < 0:
        return ""
    j = text.find("{", i)
    if j < 0:
        return ""
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j : k + 1]
    return ""


def main() -> int:
    for p in (RENDERER, LIGHTS_TS):
        if not p.exists():
            log(f"[FAIL] 缺少文件 {p}")
            return 2

    rts = RENDERER.read_text(encoding="utf-8")
    lts = LIGHTS_TS.read_text(encoding="utf-8")

    frame_floats = ts_const(rts, "FRAME_FLOATS")
    tile_size = ts_const(rts, "TILE_SIZE")
    lights_per_tile = ts_const(rts, "LIGHTS_PER_TILE")
    max_lights = ts_const(rts, "MAX_LIGHTS")
    light_stride = ts_const(lts, "LIGHT_STRIDE_FLOATS")
    object_floats = ts_const(rts, "OBJECT_FLOATS")

    failures = []

    # ---- 收集每个着色器里的 Frame / Light / ObjectU ----
    frame_variants = {}
    light_variants = {}
    obj_variants = {}
    for path in sorted(SHADERS.glob("*.wgsl")):
        text = path.read_text(encoding="utf-8")
        for name, bucket in (
            ("Frame", frame_variants),
            ("Light", light_variants),
            ("ObjectU", obj_variants),
        ):
            members = parse_struct(text, name)
            if members:
                bucket[path.name] = members

    def check_uniform(name, variants, total_floats, label):
        """同一 struct 在所有着色器里必须完全一致，且总大小等于 CPU 侧常量。"""
        if not variants:
            log(f"[skip] 未在任何着色器中找到 struct {name}")
            return
        reference = None
        for fname, members in sorted(variants.items()):
            key = tuple(members)
            if reference is None:
                reference = (fname, key, members)
            elif key != reference[1]:
                failures.append(
                    f"{name} 在 {fname} 与 {reference[0]} 中声明不一致：\n"
                    f"    {fname}: {[m[0] for m in members]}\n"
                    f"    {reference[0]}: {[m[0] for m in reference[2]]}"
                )
        rows, total_bytes = layout_of(reference[2])
        if total_bytes != total_floats * 4:
            failures.append(
                f"{name} 实际 {total_bytes} 字节，但 CPU 侧 {label}={total_floats} floats"
                f"（{total_floats * 4} 字节）"
            )
        log(f"[ok]   struct {name}: {len(rows)} 个成员 / {total_bytes} 字节，"
            f"{len(variants)} 个着色器声明一致")
        return rows

    frame_rows = check_uniform("Frame", frame_variants, frame_floats, "FRAME_FLOATS")
    check_uniform("Light", light_variants, light_stride, "LIGHT_STRIDE_FLOATS")
    check_uniform("ObjectU", obj_variants, object_floats, "OBJECT_FLOATS")

    # ---- Frame 每个成员的偏移必须等于 Renderer.ts 的 OFF_* 常量 ----
    if frame_rows:
        for fname, ftype, byte_off in frame_rows:
            const_name = "OFF_" + fname.upper()
            expect = ts_const(rts, const_name)
            got = byte_off // 4
            if expect is None:
                failures.append(f"Renderer.ts 缺少常量 {const_name}（Frame.{fname} 偏移 {got}）")
            elif expect != got:
                failures.append(
                    f"Frame.{fname}: WGSL 偏移 {got} floats，Renderer.ts {const_name}={expect}"
                )
        log(f"[ok]   Frame 的 {len(frame_rows)} 个成员偏移与 Renderer.ts 全部对齐")

    # ---- 分块剔除参数：着色器常量必须与 CPU 一致 ----
    cull = SHADERS / "LightCull.wgsl"
    if cull.exists():
        m = re.search(r"const\s+TILE\s*:\s*u32\s*=\s*(\d+)u", cull.read_text(encoding="utf-8"))
        if not m:
            failures.append("LightCull.wgsl 未找到 TILE 常量")
        elif int(m.group(1)) != tile_size:
            failures.append(
                f"LightCull.wgsl 的 TILE={m.group(1)}，Renderer.ts 的 TILE_SIZE={tile_size}"
            )
        else:
            log(f"[ok]   TILE 一致：{tile_size}")

    if lights_per_tile is not None and max_lights is not None:
        if lights_per_tile < max_lights:
            failures.append(
                f"LIGHTS_PER_TILE({lights_per_tile}) < MAX_LIGHTS({max_lights})，"
                "分块索引槽位不足，会丢光源"
            )
        else:
            log(f"[ok]   LIGHTS_PER_TILE={lights_per_tile} ≥ MAX_LIGHTS={max_lights}")

    # ---- AO：五个入口点必须与 AoMethod 枚举一一对应 ----
    ao = SHADERS / "Ao.wgsl"
    if ao.exists():
        text = ao.read_text(encoding="utf-8")
        entries = set(re.findall(r"fn\s+(fs_\w+)\s*\(", text))
        expect_entries = {"fs_off", "fs_ssao", "fs_hbao", "fs_hbaoplus", "fs_gtao"}
        missing = expect_entries - entries
        if missing:
            failures.append(f"Ao.wgsl 缺少入口点：{sorted(missing)}")
        else:
            log(f"[ok]   Ao.wgsl 五个入口点齐备：{sorted(expect_entries)}")

    # Renderer.ts 必须为每种 AoMethod 建一条管线
    m = re.search(r'export const AO_METHODS: AoMethod\[\] = \[(.*?)\];', rts, re.S)
    if not m:
        failures.append("Renderer.ts 未导出 AO_METHODS")
    else:
        methods = re.findall(r'"(\w+)"', m.group(1))
        for name in methods:
            if name == "off":
                entry = "fs_off"
            else:
                entry = "fs_" + name
            if f'"{entry}"' not in rts:
                failures.append(f'Renderer.ts 未为 AoMethod "{name}" 建管线（入口点 {entry}）')
        log(f"[ok]   AO_METHODS={methods} 均有对应管线入口点")

    # ---- 四条渲染路径：入口点、管线字段、布局必须成套存在 ----
    # 四条路径的组合最容易出的错是「加了路径名但忘了建对应管线/入口点」，
    # 那种情况 tsc 不会报（RenderPath 只是个字符串联合），只会在运行时黑屏。
    m = re.search(r'export const RENDER_PATHS: RenderPath\[\] = \[(.*?)\];', rts, re.S)
    if not m:
        failures.append("Renderer.ts 未导出 RENDER_PATHS")
    else:
        paths = re.findall(r'"([\w+-]+)"', m.group(1))
        # 每条路径 → 它必须用到的管线字段
        need = {
            "forward": ["pForwardTrad"],
            "forward+": ["pPrepass", "pLightCull", "pForwardPlus"],
            "deferred": ["pGeometry", "pLighting"],
            "tiled-deferred": ["pGeometry", "pLightCull", "pLightingTiled"],
        }
        missing_paths = [p for p in paths if p not in need]
        if missing_paths:
            failures.append(f"RENDER_PATHS 含未登记组合的路径：{missing_paths}")
        else:
            for name in paths:
                for field in need[name]:
                    if not re.search(rf"private\s+{field}\s*!", rts) or f"this.{field} " not in rts:
                        failures.append(f"路径 {name} 需要管线字段 {field}，但 Renderer.ts 中没有")
            log(f"[ok]   RENDER_PATHS={paths} 各自的管线字段齐备")

        # 两个「分块」路径必须都用到 pLightCull；两个延迟路径必须共用同一个光照布局
        if "pLightingTiled" in rts and "pLighting" in rts:
            layouts = set(re.findall(r"layout:\s*(litLayout|litTiledLayout)", rts))
            if layouts != {"litLayout"}:
                failures.append(
                    f"两条延迟光照管线必须共用同一个布局 litLayout（共享 shadePixel 使分块缓冲"
                    f"从 fs 也静态可达，Dawn 会拒绝更小的布局），实际用到 {sorted(layouts)}"
                )
            else:
                log("[ok]   两条延迟光照管线共用 litLayout（分块缓冲对 fs 静态可达）")

    # ForwardPlus.wgsl 必须同时提供分块入口与传统 Forward 入口，且二者共用 shadePixel
    fp = SHADERS / "ForwardPlus.wgsl"
    if fp.exists():
        text = fp.read_text(encoding="utf-8")
        for sig in ("fn vs(", "fn vs_forward(", "fn fs(", "fn fs_forward(", "fn shadePixel("):
            if sig not in text:
                failures.append(f"ForwardPlus.wgsl 缺少 {sig}")
        if "fn shadePixel(" in text:
            # 两个入口点都必须走 shadePixel，不得各写一份光照
            for entry in ("fn fs(", "fn fs_forward("):
                seg = fn_body(text, entry)
                if "shadePixel(" not in seg:
                    failures.append(f"ForwardPlus.wgsl 的 {entry} 未走 shadePixel()（光照代码会漂移）")
            log("[ok]   ForwardPlus.wgsl：vs/vs_forward + fs/fs_forward，两个入口点共用 shadePixel()")

    # Lighting.wgsl 必须提供 fs 与 fs_tiled，且二者共用 shadePixel
    lw = SHADERS / "Lighting.wgsl"
    if lw.exists():
        text = lw.read_text(encoding="utf-8")
        if "fn fs_tiled(" not in text:
            failures.append("Lighting.wgsl 缺少 fs_tiled 入口点（Tiled Deferred）")
        elif "fn shadePixel(" not in text:
            failures.append("Lighting.wgsl 缺少 shadePixel，两个入口点无法共用光照代码")
        else:
            for entry in ("fn fs(", "fn fs_tiled("):
                seg = fn_body(text, entry)
                if "shadePixel(" not in seg:
                    failures.append(f"Lighting.wgsl 的 {entry} 未走 shadePixel()（光照代码会漂移）")
            log("[ok]   Lighting.wgsl：fs / fs_tiled 共用 shadePixel()")

    # ---- 天空渐变：Sky.wgsl（前向路径）与 Lighting.wgsl（延迟路径）必须同色 ----
    # 前向路径靠独立的天空通道铺背景，延迟路径在光照着色器里顺手写，
    # 两处公式若不一致，同一场景会在不同路径下呈现不同背景。
    sky = SHADERS / "Sky.wgsl"
    lw2 = SHADERS / "Lighting.wgsl"
    if sky.exists() and lw2.exists():
        pat = re.compile(
            r"mix\(vec3<f32>\(([^)]*)\),\s*vec3<f32>\(([^)]*)\),\s*t\)"
        )

        def sky_pair(text):
            m = pat.search(text)
            if not m:
                return None
            return tuple(x.strip() for x in m.groups())

        a = sky_pair(sky.read_text(encoding="utf-8"))
        b = sky_pair(lw2.read_text(encoding="utf-8"))
        if a is None or b is None:
            failures.append("Sky.wgsl 或 Lighting.wgsl 中找不到天空渐变常量")
        elif a != b:
            failures.append(f"天空渐变不一致：Sky.wgsl {a} vs Lighting.wgsl {b}")
        else:
            log(f"[ok]   天空渐变两处一致：{a[0]} → {a[1]}")

    # ---- 半透物体不得再引用阴影资源 ----
    tr = SHADERS / "Transparent.wgsl"
    if tr.exists():
        text = tr.read_text(encoding="utf-8")
        leaked = [b for b in ("shadowCsmTex", "shadowLocalTex", "shadowSamp", "shadowSimple")
                  if b in text]
        if leaked:
            failures.append(
                f"Transparent.wgsl 仍引用 {leaked}——半透物体不应接收阴影"
            )
        else:
            log("[ok]   Transparent.wgsl 不引用任何阴影资源（半透不接受阴影）")

    if failures:
        log("")
        for f in failures:
            log(f"[FAIL] {f}")
        log(f"[result] {len(failures)} 项不一致")
        return 1

    log("[result] 布局全部一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
