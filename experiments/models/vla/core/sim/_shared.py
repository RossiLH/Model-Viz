# -*- coding: utf-8 -*-
"""VLA 浠跨湡鎺ュ叆 鈥斺€?鍙鍖?demo 鐨勫叡浜瀯浠躲€?
闆嗕腑鏀?灏忓瀷 CPU 鍙繍琛岄厤缃?涓庢ā鍨?鐜鏋勯€狅紝閬垮厤鍚勭粯鍥捐剼鏈悇鍐欎竴浠姐€?鎵€鏈夌粯鍥捐剼鏈殑鏁版嵁閮芥潵鑷?*鐪熷疄璺戞湰椤圭洰鐨勬ā鍧?*锛堥潪鎵嬪伐缂栭€犳洸绾匡級锛?杩欐牱鍥句笂鐨勬暟瀛楀彲浠ョ洿鎺ュ拰鏂囨。姝ｆ枃瀵圭収銆?
涓?pi05_faithful/_shared.py銆乸i07_faithful/_shared.py 鐨勫垎宸ワ細
閭ｄ袱浠芥湇鍔′簬"鍗曟ā鍨嬫満鍒跺師鐞嗗浘"锛屾湰浠芥湇鍔′簬"浠跨湡闂幆 + 鍙屾ā鍨嬪姣?锛?闇€瑕佸悓鏃舵瀯閫?PI0.5 涓?PI0.7 涓ゅ閰嶇疆鍙婂悇鑷€傞厤鍣紝鏁呯嫭绔嬬淮鎶ゃ€?"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))

from models.vla.core.sim.push_block_env import PushBlockEnv  # noqa: E402
from models.vla.core.sim.vla_agent import VLAPolicyAgent  # noqa: E402
from models.vla.pi05_faithful.config import Pi05FaithfulConfig  # noqa: E402
from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel  # noqa: E402
from models.vla.pi05_faithful.sim_adapter import Pi05SimAdapter  # noqa: E402
from models.vla.pi07.config import (  # noqa: E402
    Pi07Config, Pi07VisionConfig, Pi07ActionConfig,
)
from models.vla.pi07.model import Pi07PolicyModel  # noqa: E402
from models.vla.pi07.sim_adapter import Pi07SimAdapter  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent

#: 浠跨湡鍥惧儚杈归暱銆?2 鑰岄潪 64锛欳PU 涓婂墠鍚戝揩 4 鍊嶏紝鑰?2D 鍚堟垚鍦烘櫙
#: 鍦?32x32 涓嬪潡/鎺ㄦ潌/鐩爣渚濈劧娓呮櫚鍙鲸锛堝彲瑙嗗寲鍥捐兘鐩存帴鐪嬪嚭鏉ワ級銆?IMAGE_SIZE = 32

#: 鍔ㄤ綔鍧楅暱搴?H銆傝鏂?蟺0.7 鐢?50锛岃繖閲屽帇鍒?8 浠ヤ究鍦ㄥ浘涓婁竴鐪肩湅瀹屾暣鍧椼€?ACTION_HORIZON = 8


def setup_fonts() -> str:
    """鍚敤涓枃瀛椾綋锛屽苟淇帀鏁板璐熷彿锛圲+2212锛夌殑缂哄瓧褰㈠憡璀︺€?
    SimHei 娌℃湁 U+2212 瀛楀舰锛岃€屽潗鏍囪酱璐熷埢搴﹁蛋 mathtext 璺緞浼氬埛鍛婅銆?    鎶?mathtext 鍗曠嫭鎸囧洖 dejavusans 鍗冲彲锛堜腑鏂囦粛鐢?SimHei 娓叉煋锛夈€?    """
    import matplotlib.pyplot as plt
    from plot_utils import setup_cjk_font
    name = setup_cjk_font()
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams["mathtext.default"] = "regular"
    return name


def make_env(**kwargs) -> PushBlockEnv:
    """鏋勯€犳帹鍧楃幆澧冿紙榛樿 32x32 鍥俱€?20 姝ヤ笂闄愶級銆?""
    kwargs.setdefault("image_size", IMAGE_SIZE)
    kwargs.setdefault("max_steps", 120)
    return PushBlockEnv(**kwargs)


def make_pi05_config(**over) -> Pi05FaithfulConfig:
    """灏忓瀷 PI0.5 閰嶇疆銆?
    娉ㄦ剰蹇呴』**閫氳繃鏋勯€犲嚱鏁颁紶鍙?*锛歚`num_image_tokens`` 鏄?    ``__post_init__`` 閲屾寜 ``(image_size//patch_size)**2`` 娲剧敓鐨勶紝
    浜嬪悗鏀?``image_size`` 涓嶄細閲嶇畻锛屼細瀵艰嚧浣嶇疆缂栫爜闀垮害涓?patch 鏁颁笉鍖归厤銆?    """
    params = dict(
        paligemma_width=256, paligemma_depth=2,
        action_width=128, action_depth=2,
        head_dim=32,
        paligemma_num_heads=4, paligemma_num_kv_heads=1,
        action_num_heads=4, action_num_kv_heads=1,
        vocab_size=256,
        image_size=IMAGE_SIZE, patch_size=8,
        action_horizon=ACTION_HORIZON, action_dim=32,
        num_inference_steps=4,
    )
    params.update(over)
    return Pi05FaithfulConfig(**params)


def make_pi07_config(**over) -> Pi07Config:
    """灏忓瀷 PI0.7 閰嶇疆锛堜繚鐣欒棰戝巻鍙?/ block-causal / RTC 鍏ㄩ儴鏈哄埗璺緞锛夈€?""
    vision = dict(image_size=IMAGE_SIZE, patch_size=8, dim=32, n_layers=1,
                  n_heads=4, n_obs_steps=3, temporal_layers=1, temporal_heads=2)
    action = dict(width=32, depth=1, n_heads=4, n_kv_heads=1, head_dim=8,
                  mlp_dim=64)
    params = dict(
        action_horizon=ACTION_HORIZON, action_dim=32, state_dim=32,
        num_inference_steps=4, prompt_max_length=16,
    )
    params.update(over)
    return Pi07Config(
        vision=Pi07VisionConfig(**vision),
        action=Pi07ActionConfig(**action),
        **params,
    )


def build_pi05(config=None):
    """杩斿洖 (model, adapter)锛圥I0.5锛宔val 妯″紡锛夈€?""
    config = config if config is not None else make_pi05_config()
    model = Pi05FaithfulPolicyModel(config).eval()
    return model, Pi05SimAdapter(config)


def build_pi07(config=None):
    """杩斿洖 (model, adapter)锛圥I0.7锛宔val 妯″紡锛夈€?""
    config = config if config is not None else make_pi07_config()
    model = Pi07PolicyModel(config).eval()
    return model, Pi07SimAdapter(config)


def build_agent(which: str, n_action_steps=None, rtc_max_delay=0):
    """涓€绔欏紡鏋勯€?agent銆?
    Args:
        which: ``"pi05"`` 鎴?``"pi07"``
        n_action_steps: ``None`` = 鐩村彇妯″紡锛涙鏁存暟 = chunk 閫€瑙嗙晫妯″紡
        rtc_max_delay:  閲嶆帹鐞嗕綆姘翠綅闃堝€?
    Returns:
        (agent, model, adapter)
    """
    model, adapter = build_pi05() if which == "pi05" else build_pi07()
    agent = VLAPolicyAgent(model, adapter, n_action_steps=n_action_steps,
                           rtc_max_delay=rtc_max_delay)
    return agent, model, adapter


# 缁熶竴閰嶈壊锛堜笌椤圭洰鍏朵粬鍙鍖栦繚鎸佷竴鑷达級
C_STD = "#c62828"      # 鏍囧噯鍩虹嚎 / 鍔ｅ娍渚э紙绾級
C_NEW = "#238636"      # 鍒涙柊 / 浼樺娍渚э紙缁匡級
C_AUX = "#1565c0"      # 杈呭姪锛堣摑锛?C_AUX2 = "#6a1b9a"     # 杈呭姪锛堢传锛?C_WARN = "#ef8c00"     # 璀︾ず锛堟锛?C_GREY = "#8b949e"     # 涓€х伆
