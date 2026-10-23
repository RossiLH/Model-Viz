from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpaceTimeSiglipEncoder(nn.Module):
    """SpaceTimeSiglip 视频编码器（MEM 时空压缩的简化实现）

    π0.7 / MEM 的核心思想：**任意长度历史帧 → 固定数量 token**。
    视频通过一个"空间编码 + 因果时间注意力"的两段式编码器：
    1. 每帧独立过空间 SigLIP 编码器，得到 patch 序列；
    2. 对每个 patch 位置做**因果时间注意力**（时间维上 causal），把历史帧的
       信息聚合进当前帧的表示；
    3. 只保留当前帧（t = T-1）的 tokens —— token 数固定为 ``num_patches``，
       与历史长度 T 无关。

    与 OpenTau 的 ``SpaceTimeSiglipVideoEncoder``（video_encoder.py）对齐：
    - 空间部分复用每帧的 patch embedding + 空间位置编码；
    - 时间部分按 patch 位置拆分，时间维做 causal attention；
    - 输出只取当前帧（MEM 论文："we only pass the representation computed
      for the current timestep onwards"）；
    - ``obs_history_is_pad`` 用于屏蔽 padded 历史帧的时间注意力（纯像素置零
      不够——patch embedding 的 bias 与时间 PE 在 t < T-1 时非零）。

    简化点（相对 OpenTau 真实实现）：
    - OpenTau 每隔 4 层把 SigLIP 层的空间注意力替换为空间+时间交替；
      这里用一个轻量的时间注意力层（``temporal_layers`` 层）一次性混合历史。
    - 不加载真实 Gemma3-SigLIP 权重，用项目内 SimplifiedSigLIP 的小模型。
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        temporal_layers: int = 1,
        temporal_heads: int = 2,
        max_num_frames: int = 6,
    ):
        super().__init__()
        self.max_num_frames = max_num_frames
        self.patch_size = patch_size
        n_patches = (image_size // patch_size) ** 2

        # ---- 空间部分：复用 pi05 的 SimplifiedSigLIP 结构 ----
        from models.vla.core.modules.siglip import SimplifiedSigLIP
        self.spatial = SimplifiedSigLIP(
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
        )
        # SimplifiedSigLIP 会加 cls token，这里记录 patch 数（不含 cls）
        self.num_patches = n_patches
        self.hidden_size = dim

        # ---- 时间部分：每层对每个 patch 位置做 causal temporal attention ----
        from models.vla.core.standard.attention import StandardVLAAttention
        self.temporal_layers = nn.ModuleList()
        for _ in range(temporal_layers):
            self.temporal_layers.append(
                StandardVLAAttention(
                    dim=dim, n_heads=temporal_heads, n_kv_heads=1,
                    head_dim=dim // temporal_heads, bidirectional=False,
                )
            )
        self.temporal_norm = nn.LayerNorm(dim)

    def forward(
        self,
        video: torch.Tensor,
        obs_history_is_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """编码一段视频，返回当前帧的 tokens。

        Args:
            video: ``(B, T, C, H, W)`` 视频帧，像素 [0, 1]。
            obs_history_is_pad: ``(B, T)`` bool，True = 该历史帧是 padding
                （帧序列起始前的零填充）。默认 None → 除最后一帧外全视为 padding。

        Returns:
            ``(B, num_patches + 1, dim)`` 当前帧 tokens（含 cls）。
        """
        b, t, c, h, w = video.shape
        if t > self.max_num_frames:
            raise ValueError(f"Expected T <= max_num_frames ({self.max_num_frames}), got {t}")

        # 1) 逐帧过空间编码器：flat (B*T, C, H, W) -> (B*T, S, D)
        flat = video.reshape(b * t, c, h, w)
        feats = self.spatial(flat)                 # (B*T, S, D)
        s = feats.shape[1]
        feats = feats.reshape(b, t, s, -1)         # (B, T, S, D)

        # 2) 时间注意力：每个 patch 位置（含 cls）独立做 causal temporal attn
        #    (B, T, S, D) -> (B*S, T, D)
        feats_t = feats.transpose(1, 2).reshape(b * s, t, -1)

        # 构造时间因果 mask：query i 只能看 key j <= i
        causal = torch.triu(
            torch.full((t, t), float("-inf"), device=video.device, dtype=feats_t.dtype),
            diagonal=1,
        )
        if obs_history_is_pad is not None:
            # 屏蔽 padded 帧作为 key。
            # 关键：掩码必须作用在 **key 轴**（dim=-1）。softmax 在 key 维归一化，
            # 要让第 j 帧"不被任何 query 看见"，就得把所有 query 行的第 j 列置 -inf。
            # 若误作用在 query 轴（把第 i 行整行置 -inf），既堵不住泄漏，
            # 还会让该行 softmax 分母为 0 而输出 NaN。
            key_valid = (~obs_history_is_pad).clone()      # (B, T)，clone 避免原地改调用方张量
            key_valid[:, -1] = True                        # 当前帧恒有效
            # (B, T) -> (B*S, T)：每个 patch 位置共享同一份帧有效性
            key_valid = key_valid.repeat_interleave(s, dim=0)
            # (B*S, 1, T)：dim=1 是 query 轴（广播），dim=2 是 key 轴（被掩）
            key_valid = key_valid.unsqueeze(1)
            neg_inf = torch.full_like(causal, float("-inf"))
            # 先按 causal 铺底，再把无效 key 列整列压成 -inf
            time_mask = torch.where(key_valid, causal.unsqueeze(0), neg_inf.unsqueeze(0))
        else:
            time_mask = causal.unsqueeze(0).expand(b * s, -1, -1)

        h_t = feats_t
        for layer in self.temporal_layers:
            h_t = h_t + layer(self.temporal_norm(h_t), attn_mask=time_mask)

        # 3) 只保留当前帧 t=T-1 的表示
        h_t = h_t.reshape(b, s, t, -1)
        current = h_t[:, :, -1, :]                    # (B, S, D)
        return current