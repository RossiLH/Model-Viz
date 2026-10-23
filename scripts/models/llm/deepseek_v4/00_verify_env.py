"""环境自检脚本：验证 DeepSeekV4 分析/实现所需的运行环境是否就绪。

背景
----
本项目原先的默认解释器是 Python 3.7.9 + torch 1.11.0+cpu，
而两个 DeepSeekV4 参考仓库分别要求 Python >= 3.10（deepseek-v4-mini-pytorch）
与 >= 3.9（nano-deepseek-v4），Model-Viz 自身 pyproject 也声明 requires-python >= 3.11。
因此我们新建了 Python 3.11 虚拟环境 `Model-Viz/.venv`，
本脚本用于确认该环境满足全部分析工作的前置条件。

用法
----
    # 用 3.11 venv 运行（推荐）
    python scripts\\deepseek_v4\\00_verify_env.py

检查项
------
1. Python 版本 >= 3.11（两个参考仓库的语法/类型标注下限）
2. 关键第三方库可导入，并打印版本（torch / numpy / matplotlib / yaml / safetensors）
3. torch 可用且能跑一次最小 CPU 前向（确认不是坏轮子）
4. 两个参考仓库可被 import（这是后续所有探针脚本的基础）

退出码
------
0 = 全部通过；1 = 存在失败项（失败项会逐条打印原因）
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

# 两个参考仓库的根目录（相对本文件定位，避免依赖当前工作目录）
_REPO_ROOT = Path(__file__).resolve().parents[3]
REF_MINI = _REPO_ROOT / "Reference" / "DeepSeekV4" / "deepseek-v4-mini-pytorch-main"
REF_NANO = _REPO_ROOT / "Reference" / "DeepSeekV4" / "nano-deepseek-v4-main"

# 需要检查的第三方库；safetensors 仅 nano 仓库的 checkpoint 模块需要
_REQUIRED_MODULES = ("torch", "numpy", "matplotlib", "yaml", "safetensors", "pytest")

_failures: list[str] = []


def _check_python_version() -> None:
    """检查解释器版本 >= 3.11。"""
    want = (3, 11)
    got = sys.version_info[:2]
    ok = got >= want
    print(f"[{'OK ' if ok else 'FAIL'}] Python {got[0]}.{got[1]} (需要 >= {want[0]}.{want[1]})")
    print(f"       解释器路径: {sys.executable}")
    if not ok:
        _failures.append(
            f"Python 版本过低: {got[0]}.{got[1]}；请使用 Model-Viz/.venv 下的 3.11 解释器"
        )


def _check_modules() -> None:
    """逐个导入关键第三方库并打印版本。"""
    for name in _REQUIRED_MODULES:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "(无 __version__)")
            print(f"[OK ] import {name:<12} {ver}")
        except Exception as exc:  # noqa: BLE001 - 自检脚本需捕获任何导入错误
            print(f"[FAIL] import {name:<12} {type(exc).__name__}: {exc}")
            _failures.append(f"缺少依赖 {name}: {exc}")


def _check_torch_forward() -> None:
    """跑一次最小 CPU 前向，确认 torch 真正可用（而非只是能 import）。"""
    try:
        import torch

        torch.manual_seed(0)
        layer = torch.nn.Linear(8, 4)
        out = layer(torch.randn(2, 8))
        assert out.shape == (2, 4), out.shape
        print(f"[OK ] torch CPU 前向可用，输出 shape={tuple(out.shape)}")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] torch CPU 前向失败 {type(exc).__name__}: {exc}")
        _failures.append(f"torch 前向失败: {exc}")


def _check_reference_repo(label: str, root: Path, probe: str) -> None:
    """把参考仓库根目录加入 sys.path 后尝试导入其模块。

    Args:
        label: 打印用的仓库名
        root:  仓库根目录
        probe: 用于探测的模块名（import 成功即视为该仓库可用）
    """
    if not root.is_dir():
        print(f"[FAIL] {label}: 目录不存在 {root}")
        _failures.append(f"{label} 目录缺失: {root}")
        return
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        importlib.import_module(probe)
        print(f"[OK ] {label}: import {probe} 成功")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {label}: import {probe} 失败 {type(exc).__name__}: {exc}")
        _failures.append(f"{label} 导入失败: {exc}")


def main() -> int:
    print("=" * 68)
    print("DeepSeekV4 分析环境自检")
    print("=" * 68)

    print("\n-- 1. 解释器版本 --")
    _check_python_version()

    print("\n-- 2. 第三方依赖 --")
    _check_modules()

    print("\n-- 3. torch 可用性 --")
    _check_torch_forward()

    print("\n-- 4. 参考仓库可导入性 --")
    # mini 仓库以 src/ 为包根（pyproject 里 pythonpath = [".", "src"]）
    _check_reference_repo("deepseek-v4-mini-pytorch (主参考)", REF_MINI, "src.deepseek_components")
    _check_reference_repo("nano-deepseek-v4 (次参考)", REF_NANO, "nano_deepseek_v4.modeling")

    print("\n" + "=" * 68)
    if _failures:
        print(f"结论：{len(_failures)} 项失败")
        for i, msg in enumerate(_failures, 1):
            print(f"  {i}. {msg}")
        return 1
    print("结论：全部通过，环境就绪")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
