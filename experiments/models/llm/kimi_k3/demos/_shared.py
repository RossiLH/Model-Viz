"""Demo 共享工具模块（对齐 kimi3-master demos/_shared.py）。

提供模型加载、文本生成、配置加载等 demo 通用功能。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

#: 项目根目录（Model-Viz/）
ROOT = Path(__file__).resolve().parent.parent.parent

# 确保 src 在导入路径中
_src = str(ROOT / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from models.llm.kimi_k3.data.tokenizer import ByteTokenizer, SPECIAL_TOKENS  # noqa: E402
from models.llm.kimi_k3.config import (  # noqa: E402
    Kimi3Config, Kimi3ModelConfig, KDAConfig, MLAConfig, MoEConfig,
    AttnResConfig, MTPConfig, ViTConfig, count_params, flops_per_token,
    ParamCount,
)
from models.llm.kimi_k3.model import Kimi3ForCausalLM  # noqa: E402

#: 所有 checkpoint 都在 seq_len=256 下训练
SEQ_LEN = 256


def load_config_from_yaml(name: str) -> Kimi3Config:
    """从 configs/ 目录加载 YAML 配置。

    使用简单的 YAML 解析（不依赖 pyyaml），将 YAML 转为 Kimi3Config。

    Args:
        name: 配置名称（如 "k3_nano"）

    Returns:
        Kimi3Config 实例
    """
    path = ROOT / "configs" / f"{name}.yaml"
    if not path.exists():
        raise SystemExit(f"config not found: {path}")

    # 简单 YAML 解析（支持 K3 配置的子集）
    cfg_dict = _parse_yaml_simple(path)
    return _dict_to_config(cfg_dict)


def _parse_yaml_simple(path: Path) -> dict:
    """极简 YAML 解析器（仅支持 K3 配置格式）。

    支持：嵌套 dict、int/float/str/bool/list、注释忽略。
    不支持：锚点、多文档、复杂标量。
    """
    import re

    result = {}
    stack = [result]
    indent_stack = [-1]

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip()
            if not stripped or stripped.lstrip().startswith("#"):
                continue

            indent = len(line) - len(line.lstrip())
            content = stripped.lstrip()

            # 计算缩进层级
            while indent <= indent_stack[-1]:
                stack.pop()
                indent_stack.pop()

            if ":" in content:
                key, _, val = content.partition(":")
                key = key.strip()
                val = val.strip()

                # 去除行内注释（# 后面的内容）
                if "#" in val:
                    val = val[:val.index("#")].strip()

                if val == "" or val == "|":
                    # 子 dict
                    new_dict = {}
                    stack[-1][key] = new_dict
                    stack.append(new_dict)
                    indent_stack.append(indent)
                else:
                    stack[-1][key] = _parse_value(val)

    return result


def _parse_value(val: str):
    """解析 YAML 值。"""
    import re
    val = val.strip().rstrip(",")

    # 布尔
    if val.lower() in ("true", "yes"):
        return True
    if val.lower() in ("false", "no"):
        return False

    # None
    if val.lower() in ("null", "~", "none"):
        return None

    # 列表 [1, 2, 3]
    if val.startswith("[") and val.endswith("]"):
        inner = val[1:-1].strip()
        if not inner:
            return []
        items = [_parse_value(x.strip()) for x in inner.split(",")]
        return items

    # 字符串（带引号）
    if (val.startswith('"') and val.endswith('"')) or \
       (val.startswith("'") and val.endswith("'")):
        return val[1:-1]

    # 数字
    try:
        if "." in val or "e" in val.lower():
            return float(val)
        return int(val)
    except ValueError:
        return val


def _dict_to_config(d: dict) -> Kimi3Config:
    """将嵌套 dict 转为 Kimi3Config。"""
    model_d = d.get("model", {})

    # 子配置
    kda_d = model_d.get("kda", {})
    mla_d = model_d.get("mla", {})
    moe_d = model_d.get("moe", {})
    attn_res_d = model_d.get("attn_res", {})
    mtp_d = model_d.get("mtp", {})
    vit_d = model_d.get("vit", {})

    kda = KDAConfig(**{k: v for k, v in kda_d.items() if hasattr(KDAConfig, k)})
    mla = MLAConfig(**{k: v for k, v in mla_d.items() if hasattr(MLAConfig, k)})
    moe = MoEConfig(**{k: v for k, v in moe_d.items() if hasattr(MoEConfig, k)})
    attn_res = AttnResConfig(**{k: v for k, v in attn_res_d.items() if hasattr(AttnResConfig, k)})
    mtp = MTPConfig(**{k: v for k, v in mtp_d.items() if hasattr(MTPConfig, k)})
    vit = ViTConfig(**{k: v for k, v in vit_d.items() if hasattr(ViTConfig, k)})

    # 模型配置：只取 Kimi3ModelConfig 的 dataclass 字段
    from dataclasses import fields as dc_fields
    model_field_names = {f.name for f in dc_fields(Kimi3ModelConfig)}
    sub_config_names = {"kda", "mla", "moe", "attn_res", "mtp", "vit"}
    model_kwargs = {k: v for k, v in model_d.items()
                    if k in model_field_names and k not in sub_config_names}
    model = Kimi3ModelConfig(
        kda=kda, mla=mla, moe=moe, attn_res=attn_res, mtp=mtp, vit=vit,
        **model_kwargs
    )

    return Kimi3Config(name=d.get("name", "k3"), model=model)


def nano_config() -> Kimi3Config:
    """加载 nano 配置。"""
    return load_config_from_yaml("k3_nano")


def resolve(name: str):
    """查找 checkpoint 文件。"""
    checkpoints = {
        "800": [ROOT / "checkpoints" / "nano-800.pt"],
        "1400": [ROOT / "checkpoints" / "nano-1400.pt"],
        "3000": [ROOT / "checkpoints" / "nano-3000.pt"],
    }
    for candidate in checkpoints.get(name, [Path(name)]):
        if Path(candidate).exists():
            return Path(candidate)
    return None


def load(name: str = "3000", *, device: str = "cpu"):
    """加载模型和分词器。

    Args:
        name: checkpoint 名称
        device: 设备

    Returns:
        (model, tokenizer) 元组
    """
    path = resolve(name)
    if path is None:
        available = [k for k in ("800", "1400", "3000") if resolve(k)]
        raise SystemExit(
            f"no checkpoint found for {name!r}\n"
            f"  looked in: {path}\n"
            f"  available: {available or '(none)'}"
        )
    cfg = nano_config()
    model = Kimi3ForCausalLM(cfg).to(device)
    blob = torch.load(path, map_location=device, weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict({k: v.float() if v.is_floating_point() else v
                           for k, v in state.items()})
    model.eval()
    return model, ByteTokenizer(512)


@torch.no_grad()
def complete(model, tok, prompt: str, *, n: int = 90, temperature: float = 0.7,
             top_p: float = 0.9, seed: int = 7, device: str = "cpu") -> str:
    """续写文本。固定种子，demo 可复现。"""
    torch.manual_seed(seed)
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens=n, temperature=temperature, top_p=top_p)
    return tok.decode(out[0].tolist())


def head(text: str, lines: int = 3, width: int = 92) -> str:
    """截取前 N 行。"""
    kept = [ln[:width] for ln in text.split("\n")[:lines]]
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)


def rule(title: str = "", width: int = 78) -> None:
    """打印分隔线。"""
    if not title:
        print(f"\n{'─' * width}")
    else:
        print(f"\n── {title} " + "─" * max(0, width - len(title) - 4))
