"""用 Dawn(wgpu) 校验全部 WGSL 着色器：语法、类型、入口点、资源绑定声明。

静态检查（tsc）覆盖不到 WGSL，此脚本补上这一环：
TypeScript 全部通过 ≠ 着色器能编译。

用法： python tools/validate_wgsl.py
退出码： 0 = 全部通过，1 = 存在编译错误，2 = 无可用 GPU 后端。
"""

import sys
import pathlib

# 只 import wgpu 本体：wgpu.utils 会初始化画布后端，在无显示环境下会静默终止进程。
import wgpu

ROOT = pathlib.Path(__file__).resolve().parent.parent
SHADERS = ROOT / "src" / "Shaders"


def log(msg):
    print(msg, flush=True)


def main() -> int:
    log(f"[scan] {SHADERS}")
    files = sorted(SHADERS.glob("*.wgsl"))
    if not files:
        log(f"[FAIL] 未找到任何 .wgsl：{SHADERS}")
        return 1
    log(f"[scan] 共 {len(files)} 个着色器")

    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    except Exception as e:
        log(f"[FAIL] request_adapter 失败：{type(e).__name__}: {e}")
        return 2
    if adapter is None:
        log("[FAIL] 无法获取 GPUAdapter（需要可用的 Vulkan/DX12 后端）")
        return 2

    try:
        device = adapter.request_device_sync()
    except Exception as e:
        log(f"[FAIL] request_device 失败：{type(e).__name__}: {e}")
        return 2

    failed = False
    for path in files:
        code = path.read_text(encoding="utf-8")
        try:
            module = device.create_shader_module(code=code)
        except Exception as e:
            log(f"[FAIL] {path.name}: {e}")
            failed = True
            continue
        # create_shader_module 成功即代表 naga 校验通过；此处再取详细诊断
        try:
            for msg in module.get_compilation_info_sync():
                if msg.type == "error":
                    log(f"[FAIL] {path.name}:{msg.line_num}:{msg.line_pos} {msg.message}")
                    failed = True
                elif msg.type == "warning":
                    log(f"[warn] {path.name}:{msg.line_num}: {msg.message}")
        except Exception:
            pass
        log(f"[ok]   {path.name}  ({len(code)} bytes)")

    log("[result] " + ("存在编译错误" if failed else "全部通过"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())