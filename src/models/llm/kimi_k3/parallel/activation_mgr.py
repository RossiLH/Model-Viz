r"""统一激活管理器（论文 §5.2.2）。

在 2.8T 训练里，激活值总量往往比权重还占显存——所以 K3 用一个统一抽象
把所有"为反向保存的中间张量"按**存储后端**归类，重计算 / 量化 / offload /
远程 offload 都只是同一套抽象下的**策略**，通过轻量标注声明、与模型代码
完全解耦——这正是"存储决策不影响架构"的关键。

五个策略（Policy）：
  - KEEP        驻留，原始精度（用于小张量 / 需要高精度的张量）。
  - FP8         驻留但逐块 FP8 量化（省一半，常见主力：K3 大多数激活）。
  - OFFLOAD     移去 CPU，反向按需取回（速率慢但显存归零）。
  - REMOTE_OFFLOAD  挪到 peer pipeline rank 的内存（远程 offload，配激活均衡）。
  - RECOMPUTE   不存——反向后向重算（梯度 checkpointing 的机制）。

配合资源：
  - ``quant`` 提供了 BlockFP8 / quantize / dequantize（新增在本模块内，与
    kimi3-master/quant/fp8_blockwise.py 一致）。
  - activation_balance（pipeline.py）负责跨 rank 的远程拿放，本管理器只负责
    记账与恢复——真实远程引擎在上层接入。

使用方式（轻量注解）：
    mgr = ActivationManager(default_policy=Policy.FP8)
    mgr.annotate("attn.hidden", Policy.OFFLOAD)         # 声明策略
    mgr.annotate("mlp.elementwise", Policy.RECOMPUTE)
    entry = mgr.save("attn.hidden", t)                # forward 保存
    back = mgr.restore("attn.hidden")                 # backward 取回
    mgr.release("attn.hidden")                        # 用毕释放
    mgr.report()  # {'gpu_bytes': .., 'if_all_kept': .., 'reduction': ..}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch
from torch import Tensor

__all__ = [
    "Policy",
    "SavedTensor",
    "ActivationManager",
    "BlockFP8",
    "quantize_blockwise",
    "dequantize_blockwise",
    "quantization_error",
]


# ---------------------------------------------------------------------------
# 块级 FP8 量化（来自 quant 的概念移植；block 沿最后一维）
# ---------------------------------------------------------------------------

_E4M3_MAX = 448.0  #: e4m3 格式的最大有限值（取块缩放后 range 足够）


def fp8_dtype() -> torch.dtype:
    return getattr(torch, "float8_e4m3fn", torch.int8)


@dataclass
class BlockFP8:
    """量化 tensor + 逐块 scale。"""

    data: Tensor
    scales: Tensor
    shape: tuple[int, ...]
    block_size: int
    orig_dtype: torch.dtype

    def nbytes(self) -> int:
        return self.data.numel() + self.scales.numel() * 4

    def dequantize(self) -> Tensor:
        return dequantize_blockwise(self)


def quantize_blockwise(x: Tensor, block_size: int = 128) -> BlockFP8:
    """沿最后维度逐块量化（每 block 一个 scale）。"""
    orig_shape = tuple(x.shape)
    orig_dtype = x.dtype
    flat = x.reshape(-1, orig_shape[-1]).float()
    n, d = flat.shape
    pad = (-d) % block_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    blocks = flat.view(n, -1, block_size)

    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    dtype = fp8_dtype()
    if dtype == torch.int8:  # 老 torch 无 fp8：用 int8 量级重映射（scale 取 amax/127）
        scales = amax / 127.0
        q = (blocks / scales).clamp(-127.0, 127.0).round().to(torch.int8)
    else:
        scales = amax / _E4M3_MAX
        q = (blocks / scales).clamp(-_E4M3_MAX, _E4M3_MAX).to(dtype)
    return BlockFP8(q, scales.squeeze(-1), orig_shape, block_size, orig_dtype)


def dequantize_blockwise(q: BlockFP8) -> Tensor:
    n = q.data.shape[0]
    blocks = q.data.float().view(n, -1, q.block_size)
    out = (blocks * q.scales.unsqueeze(-1)).view(n, -1)
    d = q.shape[-1]
    return out[:, :d].reshape(q.shape).to(q.orig_dtype)


def quantization_error(x: Tensor, block_size: int = 128) -> float:
    y = dequantize_blockwise(quantize_blockwise(x, block_size))
    return float((y - x).norm() / x.norm().clamp_min(1e-12))


# ---------------------------------------------------------------------------
# 统一激活管理器
# ---------------------------------------------------------------------------


class Policy(str, Enum):
    KEEP = "keep"                 # 常驻、原始精度
    FP8 = "fp8"                   # 常驻、逐块 FP8
    OFFLOAD = "offload"           # 移 CPU，按需取回
    REMOTE_OFFLOAD = "remote"     # 移到 peer rank 内存
    RECOMPUTE = "recompute"       # 不保存，反向重算


@dataclass
class SavedTensor:
    """一张被保存的激活张量的入口。"""

    name: str
    policy: Policy
    payload: Tensor | BlockFP8 | None = None
    recompute_fn: object = None
    shape: tuple[int, ...] = ()
    dtype: torch.dtype = torch.float32
    device: torch.device | None = None

    def bytes_on_gpu(self) -> int:
        if self.policy in (Policy.OFFLOAD, Policy.REMOTE_OFFLOAD, Policy.RECOMPUTE):
            return 0
        if isinstance(self.payload, BlockFP8):
            return self.payload.nbytes()
        if isinstance(self.payload, Tensor):
            return self.payload.numel() * self.payload.element_size()
        return 0

    def restore(self) -> Tensor:
        if self.policy == Policy.RECOMPUTE:
            assert callable(self.recompute_fn), f"{self.name} 没有重算函数"
            return self.recompute_fn()
        if isinstance(self.payload, BlockFP8):
            return self.payload.dequantize()
        assert isinstance(self.payload, Tensor)
        if self.policy in (Policy.OFFLOAD, Policy.REMOTE_OFFLOAD):
            return self.payload.to(self.device or self.payload.device, non_blocking=True)
        return self.payload


@dataclass
class ActivationManager:
    """单一池子 + 逐 tensor 策略。

    peer_store 在真实运行里由远程传输引擎提供；这里用 dict 占位，
    但"哪些字节离开本设备"的记账与真实一致。
    """

    default_policy: Policy = Policy.KEEP
    fp8_block_size: int = 128
    saved: dict[str, SavedTensor] = field(default_factory=dict)
    peer_store: dict[str, Tensor] = field(default_factory=dict)
    policies: dict[str, Policy] = field(default_factory=dict)

    def annotate(self, name: str, policy: Policy) -> None:
        """按名字声明张量的存储策略——这就是"轻量标注"，把存储决策留在模型外。"""
        self.policies[name] = policy

    def save(self, name: str, tensor: Tensor, *, recompute_fn=None) -> SavedTensor:
        policy = self.policies.get(name, self.default_policy)
        entry = SavedTensor(
            name=name,
            policy=policy,
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            device=tensor.device,
            recompute_fn=recompute_fn,
        )
        if policy == Policy.RECOMPUTE:
            if recompute_fn is None:
                raise ValueError(f"tensor '{name}' 标为 recompute 却没有重算函数")
        elif policy == Policy.FP8:
            entry.payload = quantize_blockwise(tensor, self.fp8_block_size)
        elif policy == Policy.OFFLOAD:
            entry.payload = tensor.detach().to("cpu", non_blocking=True)
        elif policy == Policy.REMOTE_OFFLOAD:
            self.peer_store[name] = tensor.detach().to("cpu", non_blocking=True)
            entry.payload = self.peer_store[name]
        else:
            entry.payload = tensor.detach()
        self.saved[name] = entry
        return entry

    def restore(self, name: str) -> Tensor:
        return self.saved[name].restore()

    def release(self, name: str) -> None:
        self.saved.pop(name, None)
        self.peer_store.pop(name, None)

    def clear(self) -> None:
        self.saved.clear()
        self.peer_store.clear()

    # -- 记账 ----------------------------------------------------------------

    def gpu_bytes(self) -> int:
        return sum(e.bytes_on_gpu() for e in self.saved.values())

    def resident_bytes_if_kept(self) -> int:
        out = 0
        for e in self.saved.values():
            n = 1
            for d in e.shape:
                n *= d
            out += n * torch.finfo(e.dtype).bits // 8 if e.dtype.is_floating_point else n
        return out

    def report(self) -> dict[str, float]:
        kept = self.resident_bytes_if_kept()
        actual = self.gpu_bytes()
        return {
            "tensors": float(len(self.saved)),
            "gpu_bytes": float(actual),
            "if_all_kept": float(kept),
            "reduction": float(kept / actual) if actual else float("inf"),
        }


def moe_permuted_prob_grad(act_output: Tensor, doutput: Tensor, weights: Tensor) -> Tensor:
    """路由权重梯度，重写成不保存 layer 的 output（跟随 SonicMoE 的优化路径）。

    原生 MoE 里 permuted 概率的梯度依赖整层的 output，为此 output 必须
    存活到 backward；SonicMoe 用中间的 act_output 与上游 doutput 表达同一
    量，代价是一次轻量的逐元素计算——把一个很大的张量从激活集里拿掉。
    """
    return (act_output * doutput).sum(dim=-1, keepdim=True) * torch.ones_like(weights)