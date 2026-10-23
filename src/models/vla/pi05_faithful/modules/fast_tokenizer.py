"""FAST 动作 tokenizer（自包含，无 scipy / BPE）

设计规格 §4：方案 (a) DCT + 均匀量化。
算法权威来源：`Reference/PI/fast/processing_action_tokenizer.py`（官方 physical-intelligence/fast，
158 行）。官方 `__call__` 用 `dct(x, axis=1, norm="ortho")`（带 batch，时间轴是 axis=1）；
`fit`/`decode` 用 `axis=0`。**本实现直接对单 chunk (H, A) 沿时间轴（axis=0）做正交 DCT-II。**

流程：
    编码:  正交 DCT-II（沿时间轴，每动作维度独立）→ round(coeff * scale) → -min_token 偏移
           → 每系数一个整数 token → 截断到 max_tokens
    解码:  +min_token → reshape (H, A) → idct(coeff / scale, axis=0)（正交矩阵转置即逆）
    拟合:  对一组动作块跑 DCT+量化 → 统计 min/max token → 校验范围 ≤ vocab_size

与官方的差异（方案 (b) 记录不选）：用均匀量化（min_token 偏移）替代 BPE 聚类，不引入
`tokenizers`/scipy 依赖（见设计规格 §11 不入范围）。
"""
import numpy as np


def _dct_matrix(n):
    """构造 n×n 正交 DCT-II 矩阵（float64）。

    与 scipy.fft.dct(x, axis=0, norm="ortho") 一致的正交 DCT-II：
        C[k, m] = sqrt(2/n) * cos(pi * k * (2m+1) / (2n))   (k > 0)
        C[0, m] = sqrt(1/n)
    正交矩阵满足 C.T @ C = I，故逆 DCT = C.T @ x。

    Args:
        n: 时间步长度（动作块的时间轴长度）

    Returns:
        (n, n) float64 矩阵
    """
    k = np.arange(n)[:, None]
    m = np.arange(n)[None, :]
    mat = np.sqrt(2.0 / n) * np.cos(np.pi * k * (2 * m + 1) / (2.0 * n))
    mat[0, :] = np.sqrt(1.0 / n)
    return mat


class FastTokenizer:
    """自包含 FAST tokenizer：DCT + 均匀量化（设计规格 §4.5）"""

    def __init__(self, scale=10.0, min_token=0, max_tokens=None,
                 time_horizon=None, action_dim=None):
        """初始化

        Args:
            scale: DCT 系数量化尺度（官方 FAST+ 默认 10）
            min_token: token 偏移量（fit 确定），保证 token 非负
            max_tokens: token 序列长度上限（None = 不截断）
            time_horizon: 解码所需的时间步数（H）
            action_dim: 解码所需的动作维度（A）
        """
        self.scale = scale
        self.min_token = min_token
        self.max_tokens = max_tokens
        self.time_horizon = time_horizon
        self.action_dim = action_dim

    def _dct(self, chunk):
        """对 (H, A) chunk 沿时间轴（axis=0）做正交 DCT-II。

        Args:
            chunk: (H, A) float64

        Returns:
            (H, A) float64 系数矩阵
        """
        H = chunk.shape[0]
        return _dct_matrix(H) @ chunk

    def encode(self, action_chunk):
        """编码：DCT + 量化 + 偏移 → 整数 token 序列。

        Args:
            action_chunk: (H, A) 动作块（float）

        Returns:
            (K,) int64 token id，K = min(H*A, max_tokens)
        """
        chunk = np.asarray(action_chunk, dtype=np.float64)
        H, A = chunk.shape
        self.time_horizon = H
        self.action_dim = A
        coeff = self._dct(chunk)                       # (H, A)
        quant = np.round(coeff * self.scale)           # 量化
        tokens = (quant - self.min_token).astype(np.int64)  # 偏移保证非负
        tokens = tokens.flatten()
        if self.max_tokens is not None:
            tokens = tokens[:self.max_tokens]          # 截断
        return tokens

    def decode(self, tokens):
        """解码：+min_token → reshape (H, A) → 逆 DCT / scale。

        对应官方 decode：BPE → ord → +min_token → reshape → idct(coeff/scale, axis=0)。
        缺失的系数按 0 补齐（截断信息不可恢复）。

        Args:
            tokens: (K,) 整数 token id

        Returns:
            (H, A) 还原的动作块
        """
        if self.time_horizon is None or self.action_dim is None:
            raise ValueError("未初始化 time_horizon/action_dim，请先调用 encode() 或传入参数")
        H, A = self.time_horizon, self.action_dim
        tokens = np.asarray(tokens, dtype=np.int64).flatten()
        coeff = np.zeros(H * A, dtype=np.float64)
        n = min(len(tokens), H * A)
        coeff[:n] = (tokens[:n] + self.min_token) / self.scale
        coeff = coeff.reshape(H, A)
        return _dct_matrix(H).T @ coeff                 # 逆 DCT（正交矩阵转置）

    @classmethod
    def fit(cls, action_data, scale=10.0, vocab_size=512, max_tokens=None,
            time_horizon=None, action_dim=None):
        """拟合：统计 token 范围并确定 min_token。

        对应官方 fit：dct(a, axis=0, norm="ortho").flatten() → 统计 min/max token
        → 校验 min_vocab_size ≤ vocab_size。

        Args:
            action_data: list[(H, A) 动作块]
            scale: 量化尺度
            vocab_size: 词表大小上界
            max_tokens: token 序列长度上限
            time_horizon / action_dim: 解码参数（None 时取第一块形状）

        Returns:
            FastTokenizer（已设置 min_token）
        """
        all_quant = []
        for a in action_data:
            chunk = np.asarray(a, dtype=np.float64)
            if time_horizon is None:
                time_horizon, action_dim = chunk.shape
            all_quant.append(np.round(cls._dct_static(chunk) * scale).flatten())
        flat = np.concatenate(all_quant)
        min_token = int(flat.min())
        max_token = int(flat.max())
        if (max_token - min_token) >= vocab_size:
            raise ValueError(
                f"token 范围 {max_token - min_token} 超过 vocab_size {vocab_size}，"
                f"请增大 vocab_size 或调小 scale"
            )
        return cls(scale=scale, min_token=min_token, max_tokens=max_tokens,
                   time_horizon=time_horizon, action_dim=action_dim)

    @staticmethod
    def _dct_static(chunk):
        """供 fit 使用的静态 DCT（无实例状态）。"""
        return _dct_matrix(chunk.shape[0]) @ chunk
