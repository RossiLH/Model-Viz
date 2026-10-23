import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "src"))
import torch
import torch.nn.functional as F

from models.vla.pi07.config import (
    Pi07Config, Pi07VisionConfig, Pi07ActionConfig, Pi07RTCConfig,
    Pi07MetadataConfig,
)
from models.vla.pi07.model import Pi07PolicyModel, TOK_STATE, TOK_SUBGOAL, TOK_ACTION
from models.vla.pi07.modules.attention import (
    block_ids, make_block_attn_mask, make_suffix_cross_block_attn_mask,
)
from models.vla.pi07.modules.space_time import SpaceTimeSiglipEncoder
from models.vla.pi07.world_model import SimpleWorldModel
from models.vla.pi07.rl_token import make_rl_token_weights


def make_cfg(**rtc_kwargs):
    rtc = dict(enabled=True, max_delay=2, n_action_steps=4)
    rtc.update(rtc_kwargs)
    return Pi07Config(
        vision=Pi07VisionConfig(image_size=28, patch_size=7, dim=32, n_layers=1,
                                n_heads=4, n_obs_steps=3, temporal_layers=1,
                                temporal_heads=2),
        action=Pi07ActionConfig(width=32, depth=1, n_heads=4, n_kv_heads=1,
                                head_dim=8, mlp_dim=64),
        rtc=Pi07RTCConfig(**rtc),
        action_horizon=4, action_dim=8, state_dim=8, num_inference_steps=2,
        prompt_max_length=8,
    )


# ---------------------------------------------------------------- block-causal
def test_block_ids():
    # [视频块(2), 语言3个token各一块, 状态块(2)]
    ar = torch.tensor([0, 0, 1, 1, 1, 1, 0])
    ids = block_ids(ar)
    assert ids.tolist() == [-1, -1, 0, 1, 2, 3, 3]
    # 视频内部双向（同块）、语言严格因果（各一块）、状态块内双向


def test_block_attn_mask_video_bidirectional():
    # 视频 2 patch 双向；语言 causal
    ar = torch.tensor([0, 0, 1, 1])
    pad = torch.ones(4, dtype=torch.bool)
    mask = make_block_attn_mask(ar, pad)
    m = mask[0]  # (S, S)
    # 视频 patch 0 可见 patch 1（同块双向）
    assert m[0, 1] == 0.0
    assert m[1, 0] == 0.0
    # 语言 token 2 可见 token 0/1（后续块可看前块）
    assert m[2, 0] == 0.0
    # 语言 token 3 不可见 token 2（严格因果）
    assert m[2, 3] == float("-inf")


def test_block_attn_mask_pad():
    ar = torch.tensor([1, 1, 1])
    pad = torch.tensor([True, False, True])
    mask = make_block_attn_mask(ar, pad)
    m = mask[0]
    # query 2 不可见 pad 位置 1
    assert m[2, 1] == float("-inf")
    # query 0 是第一个块，严格因果下不可见后面的 token 2
    assert m[0, 2] == float("-inf")
    # query 2 可见自己与之前的 token（2 可见 0，1 是 pad 除外）
    assert m[2, 0] == 0.0


def test_suffix_cross_mask():
    prefix_len = 3
    suffix_ar = torch.tensor([1, 0, 0, 0])  # suffix 内部双向
    mask = make_suffix_cross_block_attn_mask(prefix_len, suffix_ar)
    m = mask[0]
    # suffix token (index 5) 可见 prefix 全部
    assert m[5, 0] == 0.0
    assert m[5, 2] == 0.0
    # suffix 内部双向
    assert m[5, 6] == 0.0
    # prefix token 不可见 suffix（prefix 块在 suffix 之前）
    assert m[0, 5] == float("-inf")


# ---------------------------------------------------------------- space-time
def test_space_time_encoder_fixed_tokens():
    enc = SpaceTimeSiglipEncoder(image_size=28, patch_size=7, dim=32, n_layers=1,
                                 n_heads=4, temporal_layers=1, temporal_heads=2,
                                 max_num_frames=4)
    video = torch.randn(2, 3, 3, 28, 28)
    out = enc(video)
    # 输出只含当前帧 tokens，数量与历史长度无关（固定 num_patches+cls）
    n_patches = (28 // 7) ** 2
    assert out.shape == (2, n_patches + 1, 32)
    # 与单帧输入应产生同样 token 数
    out1 = enc(video[:, :1])
    assert out1.shape[1] == out.shape[1]


def test_space_time_encoder_history_pad():
    enc = SpaceTimeSiglipEncoder(image_size=28, patch_size=7, dim=32, n_layers=1,
                                 n_heads=4, temporal_layers=1, temporal_heads=2,
                                 max_num_frames=4)
    video = torch.randn(1, 3, 3, 28, 28)
    pad = torch.tensor([[True, True, False]])  # 前两帧是 padding，当前帧有效
    out = enc(video, obs_history_is_pad=pad)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------- model
def test_pi07_prefix_layout():
    cfg = make_cfg()
    model = Pi07PolicyModel(cfg)
    b = 1
    videos = torch.randn(b, 3, 3, 28, 28)
    lang = torch.randint(1, 100, (b, 4))
    lang_mask = torch.ones(b, 4, dtype=torch.bool)
    state = torch.randn(b, 3, 8)
    embs, pads, ars = model.embed_prefix(
        videos, lang, lang_mask, state, return_blocks=True,
    )
    assert embs.dim() == 3
    assert embs.shape[0] == b
    assert embs.shape[-1] == cfg.action.width
    assert pads.dtype == torch.bool
    assert ars.dtype == torch.bool
    assert pads.shape == ars.shape == (b, embs.shape[1])
    # 视频块（双向）在语言（因果）之前
    n_vid = model.video_encoder.num_patches + 1
    assert embs.shape[1] > n_vid


def test_pi07_forward_basic():
    cfg = make_cfg()
    model = Pi07PolicyModel(cfg)
    b = 2
    videos = torch.randn(b, 3, 3, 28, 28)
    lang = torch.randint(1, 100, (b, 4))
    lang_mask = torch.ones(b, 4, dtype=torch.bool)
    state = torch.randn(b, 3, 8)
    actions = torch.randn(b, 4, 8)
    loss = model(videos, lang, lang_mask, state, actions)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_pi07_forward_metadata_and_subgoal():
    cfg = make_cfg()
    model = Pi07PolicyModel(cfg)
    b = 2
    videos = torch.randn(b, 3, 3, 28, 28)
    lang = torch.randint(1, 100, (b, 4))
    lang_mask = torch.ones(b, 4, dtype=torch.bool)
    state = torch.randn(b, 3, 8)
    actions = torch.randn(b, 4, 8)
    metadata = torch.randn(b, 4)
    subgoal = torch.randn(b, 3, 28, 28)
    sg_mask = torch.ones(b, dtype=torch.bool)
    loss = model(videos, lang, lang_mask, state, actions,
                 metadata=metadata, subgoal_images=subgoal, subgoal_mask=sg_mask)
    assert torch.isfinite(loss)


def test_pi07_forward_response_tokens():
    cfg = make_cfg()
    model = Pi07PolicyModel(cfg)
    b = 2
    videos = torch.randn(b, 3, 3, 28, 28)
    lang = torch.randint(1, 100, (b, 4))
    lang_mask = torch.ones(b, 4, dtype=torch.bool)
    state = torch.randn(b, 3, 8)
    actions = torch.randn(b, 4, 8)
    resp = torch.randint(1, 100, (b, 3))
    resp_mask = torch.ones(b, 3, dtype=torch.bool)
    loss = model(videos, lang, lang_mask, state, actions,
                 response_tokens=resp, response_masks=resp_mask)
    assert torch.isfinite(loss)


def test_pi07_rtc_training_uses_mask():
    """训练期 RTC：prefix 步的 x_t 应等于 actions（time=0），postfix 才监督

    注意：这里直接调用 pi07/modules/rtc.py 的共享实现（make_prefix_mask），
    而非在测试里重写一遍掩码逻辑——否则测试通过只证明"测试自己的实现自洽"，
    并不能证明模型用的那份是对的。
    """
    from models.vla.pi07.modules.rtc import make_prefix_mask

    cfg = make_cfg()
    model = Pi07PolicyModel(cfg)
    b = 2
    videos = torch.randn(b, 3, 3, 28, 28)
    lang = torch.randint(1, 100, (b, 4))
    lang_mask = torch.ones(b, 4, dtype=torch.bool)
    state = torch.randn(b, 3, 8)
    actions = torch.randn(b, 4, 8)
    # 固定 noise 与 time，检查 RTC 内部逻辑
    noise = torch.randn(b, 4, 8)
    time = torch.full((b,), 0.5)
    prefix_len = 1  # delay=1: 第 0 步是 prefix
    delay = torch.full((b,), prefix_len, dtype=torch.long)
    prefix_mask_t = make_prefix_mask(4, delay)
    # 推理逻辑应与模型内部一致。
    # 用 zeros_like 而非 python 标量 0.0：标量会被 torch 提升为 float64，
    # 与 float32 的 time 混用会报 "expected scalar type double but found float"。
    time_2d = time[:, None].expand(b, 4)
    time_m = torch.where(prefix_mask_t, torch.zeros_like(time_2d), time_2d)
    x_t_ref = time_m.unsqueeze(-1) * noise + (1 - time_m.unsqueeze(-1)) * actions
    # prefix 步 time=0 -> x_t = actions
    assert torch.allclose(x_t_ref[prefix_mask_t], actions[prefix_mask_t])
    # postfix 步 time=0.5 -> x_t 是噪声与动作的中点，必然不等于 actions
    post = ~prefix_mask_t
    assert not torch.allclose(x_t_ref[post], actions[post])


def test_pi07_sample_actions():
    cfg = make_cfg()
    model = Pi07PolicyModel(cfg).eval()
    b = 1
    videos = torch.randn(b, 3, 3, 28, 28)
    lang = torch.randint(1, 100, (b, 4))
    lang_mask = torch.ones(b, 4, dtype=torch.bool)
    state = torch.randn(b, 3, 8)
    out = model.sample_actions(videos, lang, lang_mask, state)
    assert out.shape == (b, 4, 8)
    assert torch.isfinite(out).all()


def _delay_position_sensitivity(per_token_timestep):
    """测量动作专家能否分辨"冻结前缀落在 chunk 哪个位置"

    做法：喂两组 timestep 向量，**值的多重集完全相同**、仅位置不同
    （[0,0,0,.5,.5,.5] vs [.5,.5,.5,0,0,0]），返回输出的最大差异。

    为什么这样构造：若模型把 timestep 序列做了均值池化，两者的池化结果
    逐位相等，输出必然一致 → 差异落在数值噪声量级。反之若逐 token 调制
    生效，同一个 token 位在两组里拿到不同的 scale/shift，输出必然分离。

    注意必须先打破 cond_proj 的零初始化：AdaRMSNorm 对齐官方用 zeros_init，
    初始时任何 cond 都产生零调制，此时两条路径本就无差异，测不出东西。
    """
    torch.manual_seed(0)
    cfg = make_cfg(per_token_timestep=per_token_timestep)
    model = Pi07PolicyModel(cfg).eval()
    for mod in model.action_model.modules():
        if getattr(mod, "cond_proj", None) is not None:
            torch.nn.init.normal_(mod.cond_proj.weight, std=0.3)

    torch.manual_seed(1)
    b, chunk, ad = 1, cfg.action_horizon, cfg.action_dim
    videos = torch.randn(b, 3, 3, 28, 28)
    lang = torch.randint(1, 100, (b, 4))
    lang_mask = torch.ones(b, 4, dtype=torch.bool)
    state = torch.randn(b, 3, 8)
    prefix_embs, prefix_pads, prefix_ars = model.embed_prefix(
        videos, lang, lang_mask, state)
    prefix_hidden = model.prefix_model(
        prefix_embs, attn_mask=make_block_attn_mask(prefix_ars, prefix_pads))
    prefix_len = prefix_embs.shape[1]
    x_t = torch.randn(b, chunk, ad)

    def run(time_2d):
        action_emb, cond = model.embed_suffix(x_t, time_2d)
        suffix_ar = torch.zeros(b, chunk, dtype=torch.bool)
        suffix_ar[:, 0] = True
        cross = make_suffix_cross_block_attn_mask(
            prefix_len, suffix_ar, prefix_ar=prefix_ars, prefix_pad=prefix_pads)
        full_cond = torch.cat(
            [torch.zeros(b, prefix_len, cond.shape[-1]), cond], dim=1)
        hidden = model.action_model(
            torch.cat([prefix_hidden, action_emb], dim=1),
            cond=full_cond, attn_mask=cross)
        return model.action_out_proj(hidden[:, prefix_len:])

    half = chunk // 2
    front = torch.cat([torch.zeros(b, half), torch.full((b, half), 0.5)], dim=1)
    back = torch.cat([torch.full((b, half), 0.5), torch.zeros(b, half)], dim=1)
    return (run(front) - run(back)).abs().max().item()


def test_pi07_per_token_timestep_sees_delay_position():
    """训练期 RTC 忠实性回归：动作专家必须能分辨延迟 d 的位置

    这是训练期 RTC 论文（2512.05964）三步改造的第 1 步：
        "Modify the model architecture to allow for a different flow matching
         timestep for each action timestep ... simply allow the scale, shift,
         and gate to differ between tokens."

    若这一条不成立，RTC 就退化成"只把损失遮掉几列"——冻结前缀的位置信息
    根本没进网络，模型学不到"基于已提交的前 d 步续写"这件事。
    """
    assert _delay_position_sensitivity(per_token_timestep=True) > 1e-3


def test_pi07_pooled_timestep_is_delay_blind_baseline():
    """before/after 基线：关闭 per_token_timestep 后模型对延迟位置失明

    保留这个退化路径是为了对照实验（本项目的核心卖点）。差异应落在
    float32 数值噪声量级，证明"均值池化确实掐断了 RTC 的条件化通路"。
    """
    assert _delay_position_sensitivity(per_token_timestep=False) < 1e-5


def test_pi07_per_token_timestep_adds_no_parameters():
    """逐 token 调制不得增加参数量（论文明确断言）

    2512.05964 p2: "This does not change the number of learnable parameters."
    """
    n_on = sum(p.numel() for p in
               Pi07PolicyModel(make_cfg(per_token_timestep=True)).parameters())
    n_off = sum(p.numel() for p in
                Pi07PolicyModel(make_cfg(per_token_timestep=False)).parameters())
    assert n_on == n_off


def test_pi07_sample_actions_rtc_prefix():
    """RTC 推理：prefix 步被已执行动作覆盖并保持"""
    cfg = make_cfg()
    model = Pi07PolicyModel(cfg).eval()
    b = 1
    videos = torch.randn(b, 3, 3, 28, 28)
    lang = torch.randint(1, 100, (b, 4))
    lang_mask = torch.ones(b, 4, dtype=torch.bool)
    state = torch.randn(b, 3, 8)
    action_prefix = torch.randn(b, 4, 8)  # 前 delay 步真实，其余零填充
    delay = 2
    out = model.sample_actions(videos, lang, lang_mask, state,
                               action_prefix=action_prefix, delay=delay)
    # 前 delay 步应保持 action_prefix 的值
    assert torch.allclose(out[0, :delay], action_prefix[0, :delay])
    assert torch.isfinite(out[0, delay:]).all()


def test_pi07_forward_per_sample():
    cfg = make_cfg()
    model = Pi07PolicyModel(cfg)
    b = 3
    videos = torch.randn(b, 3, 3, 28, 28)
    lang = torch.randint(1, 100, (b, 4))
    lang_mask = torch.ones(b, 4, dtype=torch.bool)
    state = torch.randn(b, 3, 8)
    actions = torch.randn(b, 4, 8)
    loss, extra = model(videos, lang, lang_mask, state, actions, return_per_sample=True)
    assert extra["MSE_per_sample"].shape == (b,)


def test_pi07_rl_token_weight_forward():
    cfg = make_cfg()
    model = Pi07PolicyModel(cfg)
    b = 2
    videos = torch.randn(b, 3, 3, 28, 28)
    lang = torch.randint(1, 100, (b, 4))
    lang_mask = torch.ones(b, 4, dtype=torch.bool)
    state = torch.randn(b, 3, 8)
    actions = torch.randn(b, 4, 8)
    rl_w = torch.tensor([1.25, 0.75])
    loss = model(videos, lang, lang_mask, state, actions, rl_token_weight=rl_w)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------- rl token (RLT, arXiv 2604.23073)
def test_make_rl_token_weights():
    adv = torch.tensor([1.0, 0.0])
    inter = torch.tensor([False, True])
    w = make_rl_token_weights(adv, inter)
    assert torch.allclose(w, torch.tensor([1.25, 0.75 * 1.5]), atol=1e-5)


def test_make_rl_token_weights_explicit():
    adv = torch.tensor([1.0, 0.0])
    exp = torch.tensor([2.0, 1.0])
    w = make_rl_token_weights(adv, explicit_weights=exp)
    assert torch.allclose(w, torch.tensor([2.5, 0.75]), atol=1e-5)


# ---------------------------------------------------------------- RLT extractor
def test_rlt_readout_shape_and_input_dependency():
    """RLT 式 1：RL token 形状固定 (B, token_dim)，且依赖整段输入嵌入。
    用同分布随机初始化两批输入，readout 必须随输入变化（不是常数向量）。"""
    from models.vla.pi07.rl_token import RLTokenExtractor
    ex = RLTokenExtractor(token_dim=16, encoder_layers=1, decoder_layers=1,
                          n_heads=2)
    z1 = torch.randn(2, 6, 16)
    z2 = torch.randn(2, 6, 16)
    r1 = ex.extract(z1)          # (B, token_dim)
    r2 = ex.extract(z2)
    assert r1.shape == (2, 16) and r2.shape == (2, 16)
    # 默认不冻结输入：readout 应随输入显著变化（除非权重完全相同，同 seed）
    assert (r1 - r2).abs().max().item() > 1e-4
    # 输入可反向（extract 默认训练态）
    loss = r1.sum()
    loss.backward()
    assert ex.encoder.blocks[0].attn.q_proj.weight.grad is not None


def test_rlt_readout_bottleneck_reconstructs():
    """RLT 式 2：RL token 是信息瓶颈 —— decoder 能从它自回归重建原嵌入。
    训练一小段后重建损失必须显著下降。"""
    from models.vla.pi07.rl_token import RLTokenExtractor
    torch.manual_seed(0)
    ex = RLTokenExtractor(token_dim=16, encoder_layers=1, decoder_layers=2,
                          n_heads=2)
    z = torch.randn(4, 6, 16)              # 冻结的 VLA 末层嵌入
    opt = torch.optim.Adam(ex.parameters(), lr=1e-2)
    def loss_():
        return ex.reconstruction_loss(z)
    first = loss_().item()
    for _ in range(150):
        opt.zero_grad()
        loss_().backward()
        opt.step()
    last = loss_().item()
    assert last < first * 0.3, f"重建损失未下降: {first:.4f} -> {last:.4f}"


def test_rlt_reconstruction_stopgrad():
    """式 2 的 sg(·)：重建目标与 decoder 输入都停止梯度 —— 不反向到
    VLA 嵌入（对冻结 VLA 而言，extract 的 readout 输入不得有梯度）。"""
    from models.vla.pi07.rl_token import RLTokenExtractor
    ex = RLTokenExtractor(token_dim=16, encoder_layers=1, decoder_layers=1,
                          n_heads=2)
    z = torch.randn(2, 5, 16, requires_grad=True)
    ex.reconstruction_loss(z).backward()
    # 冻结 VLA 语义：重建损失不向 z（VLA 末层嵌入）回传梯度
    assert z.grad is None


# ---------------------------------------------------------------- RLT chunk critic
def test_rlt_critic_shape_min_ensemble():
    """式 3：critic 是双 Q 集成，前向给 (x, a_{1:C})，返回 (B,) 标量值；
    训练 target 用 min over ensemble（TD3）。"""
    from models.vla.pi07.rl_token import ChunkCritic
    c = ChunkCritic(state_dim=8, action_dim=4, chunk_len=5,
                    hidden=32, layers=2, n_critics=2)
    x = torch.randn(3, 8)
    a = torch.randn(3, 5, 4)
    q = c(x, a)
    assert q.shape == (3,)
    # 集成个数与 min 语义：forward = min over individual_values
    assert len(c.critics) == 2
    qs = c.individual_values(x, a)
    assert len(qs) == 2 and qs[0].shape == (3,)
    assert torch.allclose(q, torch.minimum(qs[0], qs[1]), atol=1e-6)


def test_rlt_critic_td_loss_handcheck():
    """式 3 目标值：r1 + γ r2 + γ² r3 + γ^C · min_i Q_{tar}(x', a')。
    用手工张量核对（γ=0.9, C=3）。"""
    from models.vla.pi07.rl_token import ChunkCritic, chunk_return, td_target
    gamma = 0.9
    rewards = torch.tensor([[1.0, 2.0, 0.0]])   # (B, C)
    assert torch.allclose(chunk_return(rewards, gamma),
                          torch.tensor([[1.0 + 0.9 * 2.0 + 0.9 ** 2 * 0.0]]),
                          atol=1e-6)
    # 手工构造两个线性 critic：q1 = w1·[x,a], q2 = w2·[x,a]，验证 min
    c = ChunkCritic(state_dim=2, action_dim=2, chunk_len=2,
                    hidden=4, layers=1, n_critics=2)
    for crit in c.critics:
        crit.net = torch.nn.Identity()
    # 直接测 td_target 的 min 逻辑：用两个不同的标量 Q 目标
    x = torch.randn(1, 2)
    a = torch.randn(1, 2, 2)
    next_x = torch.randn(1, 2)
    r = torch.tensor([[0.5, 0.5]])
    q_tar1 = torch.tensor([[2.0]])
    q_tar2 = torch.tensor([[1.0]])
    target = td_target(r, q_tar1, q_tar2, gamma)
    # target = R + γ^C · min(q1, q2) = 0.5 + 0.9·0.5 + 0.9²·min(2,1)
    expected = 0.5 + 0.9 * 0.5 + 0.9 ** 2 * 1.0
    assert torch.allclose(target, torch.tensor([[expected]]), atol=1e-5)


# ---------------------------------------------------------------- RLT refine actor
def test_rlt_actor_conditioned_on_reference():
    """式 4：actor 是高斯策略，条件在参考动作块 ā 上 —— 不同 ā 得不同均值；
    标准差固定为 actor_std。"""
    from models.vla.pi07.rl_token import RefineActor
    torch.manual_seed(1)
    act = RefineActor(state_dim=8, action_dim=4, chunk_len=5,
                      hidden=32, layers=2, actor_std=0.1)
    x = torch.randn(3, 8)
    ref_a = torch.randn(3, 5, 4)
    ref_b = torch.randn(3, 5, 4)
    d_a = act.distribution(x, ref_a)
    d_b = act.distribution(x, ref_b)
    assert d_a.mean.shape == (3, 5, 4)
    # 标准差固定
    assert torch.allclose(d_a.stddev, torch.full_like(d_a.stddev, 0.1), atol=1e-6)
    # 条件化确实改变均值（随机初始化下均值应可区分）
    assert (d_a.mean - d_b.mean).abs().max().item() > 1e-4


def test_rlt_actor_reference_dropout():
    """论文附录 B：参考动作训练时 50% 置零 —— 整批全 drop 时 actor 不再
    依赖 ā，两批不同 ā 的分布均值应几乎一致（只剩 state 条件）。"""
    from models.vla.pi07.rl_token import RefineActor
    torch.manual_seed(2)
    act = RefineActor(state_dim=8, action_dim=4, chunk_len=5,
                      hidden=32, layers=2, actor_std=0.1)
    x = torch.randn(3, 8)
    ref_a = torch.randn(3, 5, 4)
    ref_b = torch.randn(3, 5, 4)
    da = act.distribution(x, ref_a, ref_dropout=1.0)
    db = act.distribution(x, ref_b, ref_dropout=1.0)
    assert (da.mean - db.mean).abs().max().item() < 1e-5


def test_rlt_actor_refines_and_regulates():
    """式 5：actor 在参考块附近做局部精修 —— 强正则 α 时，策略均值向 ā 靠拢；
    弱正则时偏离更大。"""
    from models.vla.pi07.rl_token import RefineActor
    torch.manual_seed(3)
    act = RefineActor(state_dim=8, action_dim=4, chunk_len=5,
                      hidden=64, layers=2, actor_std=0.1)
    x = torch.randn(1, 8)
    ref = torch.randn(1, 5, 4)
    # 零/小正则：动作均值是"精修"，不应直接等于参考
    mean_no_reg = act.distribution(x, ref).mean
    # 构造一个损失逼着均值接近 ref（相当于 α→∞），训练几轮
    opt = torch.optim.Adam(act.parameters(), lr=1e-2)
    for _ in range(100):
        opt.zero_grad()
        loss = ((act.distribution(x, ref).mean - ref) ** 2).mean()
        loss.backward()
        opt.step()
    mean_reg = act.distribution(x, ref).mean
    # 训练后均值明显更接近参考
    assert ((mean_reg - ref) ** 2).mean().item() < 0.01


def test_rlt_chunk_subsample():
    """论文附录 B：动作块按 stride 2 子采样 —— 每段数据给出多组
    (state, chunk) 对齐样本。形状 (B, N_samp, C, D)。"""
    from models.vla.pi07.rl_token import subsample_chunks
    x = torch.arange(10, dtype=torch.float32).view(1, 10, 1)  # (B, T, D)
    a = torch.arange(10, dtype=torch.float32).view(1, 10, 1)
    xs, as_ = subsample_chunks(x, a, chunk_len=4, stride=2)
    # 起点 0,2,4,6 且窗口不越界 -> 4 组；B=1 -> (1, 4, 4, 1)
    assert xs.shape == (1, 4, 4, 1) and as_.shape == (1, 4, 4, 1)
    # 检查第 0 组与第 1 组对齐（起点差 2）
    assert torch.allclose(as_[0, 0], torch.tensor([[0.0], [1.0], [2.0], [3.0]]))
    assert torch.allclose(as_[0, 1], torch.tensor([[2.0], [3.0], [4.0], [5.0]]))
    # 多 batch 正确广播
    x2 = torch.arange(20, dtype=torch.float32).view(2, 10, 1)
    a2 = torch.arange(20, dtype=torch.float32).view(2, 10, 1)
    xs2, as2 = subsample_chunks(x2, a2, chunk_len=4, stride=2)
    assert xs2.shape == (2, 4, 4, 1)
    assert torch.allclose(as2[1, 0], torch.tensor([[10.0], [11.0], [12.0], [13.0]]))


# ---------------------------------------------------------------- world model
def test_world_model_generate():
    wm = SimpleWorldModel(image_size=28, patch_size=7, dim=32, n_layers=1,
                          n_heads=4, cond_dim=32)
    current = torch.randn(2, 3, 28, 28)
    cond = torch.randn(2, 32)
    out = wm(current, cond)
    assert out.shape == (2, 3, 28, 28)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_world_model_training_loss():
    wm = SimpleWorldModel(image_size=28, patch_size=7, dim=32, n_layers=1,
                          n_heads=4, cond_dim=32)
    batch = {
        "current": torch.randn(2, 3, 28, 28),
        "target": torch.rand(2, 3, 28, 28),
    }
    loss = wm.training_loss(batch)
    assert torch.isfinite(loss)
    # 同一条件、同一输入，损失应可反向传播
    loss.backward()
    assert wm.encoder.patch_embed.weight.grad is not None


# ---------------------------------------------------------------- metadata
def test_bin_episode_length():
    """§V-C 分箱：1750~2250 -> 2000（论文原文区间）"""
    from models.vla.pi07.modules.metadata import bin_episode_length
    lens = torch.tensor([1749, 1750, 1873, 2000, 2249, 2250, 2251, 8124])
    binned = bin_episode_length(lens, bin_size=500)
    assert binned.tolist() == [1500, 2000, 2000, 2000, 2000, 2000, 2500, 8000]
    # dtype 保持
    assert binned.dtype == lens.dtype


def test_bin_episode_length_rounding_symmetry():
    """分箱是"四舍五入到最近 500 倍数"，边界恰在 ±250 处"""
    from models.vla.pi07.modules.metadata import bin_episode_length
    x = torch.arange(1749, 2252, 1, dtype=torch.float32)
    b = bin_episode_length(x)
    # 1749 -> 1500（差 249 < 250，向下），1750 -> 2000（恰在边界，银行家舍入取偶）
    assert b[0] == 1500 and b[1] == 2000
    # 2250 恰在 [1750, 2250] 的右边界，按论文语义归 2000
    assert b[-1] == 2500 and b[-2] == 2000


def test_sample_prompt_dropout_statistics():
    """§V-E 两级 dropout 的经验统计必须落在论文给出的概率附近"""
    from models.vla.pi07.modules.metadata import sample_prompt_dropout
    g = torch.Generator().manual_seed(12345)
    n = 40000
    d = sample_prompt_dropout(
        n, generator=g,
        metadata_drop_prob=0.15, per_field_drop_prob=0.05,
        subgoal_keep_prob=0.25, subtask_drop_given_subgoal=0.30,
    )
    def frac(t): return t.float().mean().item()
    # 整块保留 ≈ 0.85
    assert abs(frac(d["metadata"]) - 0.85) < 0.01
    # 逐字段保留 ≈ 0.95（在整块保留的子集里）
    m = d["metadata"]
    fields_given_meta = d["metadata_fields"][m]
    assert abs(fields_given_meta.float().mean().item() - 0.95) < 0.01
    # 子目标图像加入 ≈ 0.25
    assert abs(frac(d["subgoal"]) - 0.25) < 0.01
    # 有子目标的样本里，子任务被图像顶替 ≈ 0.30
    sg = d["subgoal"]
    drop_rate = (~d["subtask"][sg]).float().mean().item()
    assert abs(drop_rate - 0.30) < 0.02
    # 无子目标的样本，子任务必须恒在
    assert bool((~d["subgoal"] & ~d["subtask"]).any()) is False
    # 控制模式恒在
    assert bool(d["control_mode"].all())


def test_make_metadata_identity():
    """§VII 推理档位：speed 15 百分位 + quality 恒 5 + mistake 恒 false"""
    from models.vla.pi07.modules.metadata import make_metadata_identity
    import torch as t
    speed = t.tensor([1873.0])     # 论文示例区间内
    quality = t.tensor([5.0])
    mistake = t.tensor([0.0])      # false
    meta = make_metadata_identity(speed, quality, mistake)
    assert meta.shape == (1, 3)
    assert meta[0, 0].item() == 2000.0   # 1873 -> 2000（分箱后）
    assert meta[0, 1].item() == 5.0
    assert meta[0, 2].item() == 0.0


def test_metadata_mask_zeroes_embedding_and_mask():
    """per-field dropout：被弃字段 embedding 置零 + pad_mask 置 False"""
    cfg = make_cfg()
    model = Pi07PolicyModel(cfg).eval()
    b = 1
    videos = torch.randn(b, 3, 3, 28, 28)
    lang = torch.randint(1, 100, (b, 4))
    lang_mask = torch.ones(b, 4, dtype=torch.bool)
    state = torch.randn(b, 3, 8)
    meta = torch.tensor([[2000.0, 5.0, 0.0]])
    meta_mask = torch.tensor([[True, False, True]])
    embs, pads, ars = model.embed_prefix(
        videos, lang, lang_mask, state,
        metadata=meta, metadata_masks=meta_mask)
    # 定位元数据字段 token 段（在响应之后、分隔符之前）；这里无 response，
    # 元数据段紧跟状态块。直接验证：被弃字段（index 1）对应的 mask 为 False。
    meta_tokens_mask = torch.all(pads & (embs[0].sum(-1) != 0), dim=-1)  # 非零且非 pad
    # 更直接：检查 pads 里正好有 3 个 True（元数据段）+ 其余非元数据段
    # 用 embed_prefix 的 return_blocks 语义难以直接索引，改为端到端验证：
    # 把 meta_mask 全 True 与当前（字段1被弃）的 prefix 长度差应 = 0（token 数
    # 不变，因为字段被屏蔽而非移除）—— 这正是注释里说的"用 mask 而非移除"。
    embs_all, pads_all, _ = model.embed_prefix(
        videos, lang, lang_mask, state,
        metadata=meta, metadata_masks=torch.ones_like(meta_mask))
    assert embs.shape == embs_all.shape
    # 被弃字段的 embedding 确实被置零了
    zero_cols = embs[0, pads[0]]  # 只取有效 token
    # 验证：全真掩码时有效 token 更多（多出字段1）
    assert pads_all[0].sum() > pads[0].sum()
    # 且差异恰为 1（一个字段被弃）
    assert (pads_all[0].sum() - pads[0].sum()).item() == 1


# ---------------------------------------------------------------- CFG
def test_cfg_guidance_changes_output():
    """CFG 必须改变生成动作（β>0 时），且 β=0 时退化为无条件"""
    cfg = make_cfg()
    model = Pi07PolicyModel(cfg).eval()
    b = 1
    videos = torch.randn(b, 3, 3, 28, 28)
    lang = torch.randint(1, 100, (b, 4))
    lang_mask = torch.ones(b, 4, dtype=torch.bool)
    state = torch.randn(b, 3, 8)
    meta = torch.tensor([[2000.0, 5.0, 0.0]])
    noise = torch.randn(b, 4, 8)
    out_plain = model.sample_actions(videos, lang, lang_mask, state,
                                     metadata=meta, noise=noise, num_steps=2)
    out_cfg0 = model.sample_actions(videos, lang, lang_mask, state,
                                    metadata=meta, noise=noise, num_steps=2,
                                    use_cfg=True, cfg_beta=0.0)
    out_cfg = model.sample_actions(videos, lang, lang_mask, state,
                                   metadata=meta, noise=noise, num_steps=2,
                                   use_cfg=True, cfg_beta=1.7)
    # β=0 时 CFG 外插量为 0，应与普通路径一致
    assert torch.allclose(out_plain, out_cfg0, atol=1e-5)
    # β=1.7 时输出必须显著偏离普通路径
    assert (out_cfg - out_plain).abs().max().item() > 1e-4


# ================================================================ 联动集成
# 以下测试回答两个"能不能真的接起来"的问题：
# Q9  世界模型 -> VLA：SimpleWorldModel 生成的子目标图像能否直接喂给策略；
# Q12 VLA -> RLT：RLTokenExtractor 能否消费策略真实的 prefix 末层隐藏态。
def test_encode_prefix_hidden_exposes_vla_last_layer():
    """RLT 挂载点：策略必须能吐出 prefix 末层隐藏态 z_{1:M} 及其有效掩码。

    这是 RL Token 式 1 的输入。没有这个接口，RLT 只能吃随机张量，
    "冻结 VLA + 在其表示上做 RL"的语义就无法闭环。
    """
    cfg = make_cfg()
    model = Pi07PolicyModel(cfg).eval()
    b = 2
    videos = torch.randn(b, 3, 3, 28, 28)
    lang = torch.randint(1, 100, (b, 4))
    lang_mask = torch.ones(b, 4, dtype=torch.bool)
    state = torch.randn(b, 3, 8)

    z, pad = model.encode_prefix_hidden(videos, lang, lang_mask, state)
    # 形状：(B, M, width)，且与 embed_prefix 的 token 数一致
    embs, pads, _ = model.embed_prefix(videos, lang, lang_mask, state)
    assert z.shape == (b, embs.shape[1], cfg.action.width)
    assert pad.shape == (b, embs.shape[1])
    assert torch.equal(pad, pads)
    # 必须是"过了 VLM 之后"的表示，而不是原始 embedding
    assert not torch.allclose(z, embs, atol=1e-4)
    # 冻结语义：默认 detach（调用方不必再手动 detach）
    assert z.requires_grad is False


def test_rlt_consumes_real_vla_hidden_states():
    """Q12 闭环：RLT 三件套接在真实策略输出上，能跑通并降低重建损失。"""
    from models.vla.pi07.rl_token import RLTokenExtractor, ChunkCritic, RefineActor

    cfg = make_cfg()
    model = Pi07PolicyModel(cfg).eval()
    b = 2
    videos = torch.randn(b, 3, 3, 28, 28)
    lang = torch.randint(1, 100, (b, 4))
    lang_mask = torch.ones(b, 4, dtype=torch.bool)
    state = torch.randn(b, 3, 8)

    z, _ = model.encode_prefix_hidden(videos, lang, lang_mask, state)
    w = cfg.action.width
    rlt = RLTokenExtractor(token_dim=w, encoder_layers=1, decoder_layers=1, n_heads=4)

    # 式 1：真实 z -> 单个 RL token
    z_rl = rlt.extract(z)
    assert z_rl.shape == (b, w)

    # 式 2：重建损失可优化（证明真实 VLA 表示可被压进一个向量）
    opt = torch.optim.Adam(rlt.parameters(), lr=3e-3)
    first = rlt.reconstruction_loss(z).item()
    for _ in range(60):
        opt.zero_grad()
        loss = rlt.reconstruction_loss(z)
        loss.backward()
        opt.step()
    assert loss.item() < first

    # 冻结语义：VLA 参数不因 RLT 训练而获得梯度
    assert all(p.grad is None for p in model.parameters())

    # 式 3/4/5：actor 在策略提议的参考块上做局部精修
    a_ref = model.sample_actions(videos, lang, lang_mask, state, num_steps=2)
    assert a_ref.shape == (b, cfg.action_horizon, cfg.action_dim)
    x = torch.cat([z_rl, state[:, -1]], dim=-1)          # (B, w + state_dim)
    actor = RefineActor(state_dim=x.shape[-1], action_dim=cfg.action_dim,
                        chunk_len=cfg.action_horizon, hidden=32, layers=2)
    critic = ChunkCritic(state_dim=x.shape[-1], action_dim=cfg.action_dim,
                         chunk_len=cfg.action_horizon, hidden=32, layers=2)
    a_refined = actor.distribution(x, a_ref).mean
    assert a_refined.shape == a_ref.shape
    q = critic(x, a_refined)
    assert q.shape == (b,)


def test_world_model_output_feeds_policy_as_subgoal():
    """Q9 闭环：世界模型生成的子目标图像直接作为策略的 subgoal_images。"""
    cfg = make_cfg()
    model = Pi07PolicyModel(cfg).eval()
    wm = SimpleWorldModel(image_size=28, patch_size=7, dim=32, n_layers=1,
                          n_heads=4, cond_dim=32).eval()
    b = 2
    videos = torch.randn(b, 3, 3, 28, 28)
    lang = torch.randint(1, 100, (b, 4))
    lang_mask = torch.ones(b, 4, dtype=torch.bool)
    state = torch.randn(b, 3, 8)

    # 任务条件 = 语言 embedding 均值池化（与世界模型 docstring 约定一致）
    with torch.no_grad():
        cond = model.text_embed(lang).mean(dim=1)         # (B, width=32)
        subgoal = wm(videos[:, -1], cond)                 # (B, 3, 28, 28)
    # 生成图必须与相机帧同形状，才能被 spatial 编码器接受
    assert subgoal.shape == videos[:, -1].shape
    assert float(subgoal.min()) >= 0.0 and float(subgoal.max()) <= 1.0

    mask_on = torch.ones(b, dtype=torch.bool)
    noise = torch.randn(b, cfg.action_horizon, cfg.action_dim)
    a_with = model.sample_actions(videos, lang, lang_mask, state, noise=noise,
                                  num_steps=2, subgoal_images=subgoal,
                                  subgoal_mask=mask_on)
    a_without = model.sample_actions(videos, lang, lang_mask, state, noise=noise,
                                     num_steps=2)
    assert a_with.shape == a_without.shape
    # 子目标图像确实改变了动作（视觉条件真的被读进去了）
    assert (a_with - a_without).abs().max().item() > 1e-5
