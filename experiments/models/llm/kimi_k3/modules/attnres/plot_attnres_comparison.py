# -*- coding: utf-8 -*-
"""AttnRes vs mHC vs 鏍囧噯娈嬪樊 鈥斺€?娣卞害缁磋繛鎺ヤ笁绉嶈寖寮忕殑瀵规瘮鍥俱€?
鑳屾櫙
====
docs/KIMI-K3/AttnRes/Picture/1 閲屾槸涓€浠?ASCII 涓夋爮瀵规瘮銆傛湰鑴氭湰鎶婂畠鍗囩骇涓哄嵃鍒风骇
PNG锛氫笁寮犳灦鏋勭ず鎰忓浘锛堟爣鍑嗘畫宸?/ mHC / AttnRes锛? 涓夊紶閲忓寲灏忓浘锛堟繁搴?鏃堕棿瀵瑰伓鎬?/
姣忓眰鍐呭瓨 I/O / 璁烘枃 Table 4 鍥版儜搴︼級銆?
涓夎€呯殑鍖哄埆鏈川鏄€庢繁搴﹀巻鍙叉€庝箞璇汇€忥紙璁烘枃 搂6.2 鐨勬繁搴?鏃堕棿瀵瑰伓鎬э級锛?  1. 鏍囧噯娈嬪樊  鈥斺€?鍙姞鑷繁鐨勮緭鍑恒€佷笉璇诲巻鍙诧紙绛夋潈鍔犳硶锛宭inear锛夛紱
  2. mHC       鈥斺€?m 鏉″钩琛屾祦 + 瀛︿範 + 鍙岄殢鏈?Sinkhorn)绾︽潫鐨勮浆绉荤煩闃?A
                  锛堟繁搴︾淮 linear attention锛岀姸鎬佸帇缂╄繘 d脳m锛夛紱
  3. AttnRes   鈥斺€?娣卞害缁?softmax 娉ㄦ剰鍔涳細鍧楄〃绀?+ running partial 浣滀负銆庡巻鍙?                  鐩綍銆忥紝鍙涔犱吉鏌ヨ w_l 鎸夊唴瀹硅蒋閫夋嫨锛坕nput-dependent锛夈€?
杩愯:
    python experiments/attnres/plot_attnres_comparison.py
浜у嚭:
    experiments/attnres/AttnRes-vs-mHC-vs-Residual.png
    docs/KIMI-K3/AttnRes/Picture/AttnRes-vs-mHC-vs-Residual.png锛堝悓涓€寮犲壇鏈級

鏁板€兼潵婧愶細K3 鎶ュ憡 搂2.2 / AttnRes 璁烘枃 Table 4锛?6 hidden 灞傦級/ 鍒嗘瀽鏂囨。闄勫綍 A銆?"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# 鑴氭湰鐩綍 experiments/attnres锛屽叾涓?plot_utils.py 鎻愪緵涓枃瀛椾綋閰嶇疆銆?HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from plot_utils import setup_cjk_font  # noqa: E402

setup_cjk_font()

# 椤圭洰鏍?Model-Viz锛涗骇鐗╁悓鏃跺鍒跺埌 docs 鐨?Picture 鐩綍涓?ASCII 鐗堝苟瀛樸€?ROOT = HERE.parents[6]
PICTURE_DIR = ROOT / "docs" / "KIMI-K3" / "AttnRes" / "Picture"
PNG_NAME = "AttnRes-vs-mHC-vs-Residual.png"

# 涓夋柟娉曚富棰樿壊锛欰ttnRes 鏄枃妗ｄ富瑙掞紙绾級锛屾爣鍑嗘畫宸紙钃濓級銆乵HC锛堢惀鐝€锛夊仛瀵圭収銆?C_RES = "#5B7DB1"    # 鏍囧噯娈嬪樊锛氳摑鐏?C_MHC = "#D98E2B"    # mHC锛氱惀鐝€
C_ARN = "#C0392B"    # AttnRes锛氱孩锛堜富瑙掞級
C_ARN_L = "#E8938B"  # AttnRes 娴呯孩锛堝潡鍙樹綋锛?C_GRAY = "#8A94A0"   # 涓€х伆
C_LINE = "#3D4450"   # 娣卞ⅷ钃濓紙杈规/姝ｆ枃锛?

# ---------------------------------------------------------------------------
# 缁樺浘鍘熻
# ---------------------------------------------------------------------------
def _rbox(ax, cx, cy, w, h, text="", *, fc="#FFFFFF", ec=C_LINE, lw=1.5,
          fs=10, weight="normal", textcolor=None, ls="-", zorder=3):
    """鍦嗚鐭╁舰锛氫腑蹇?(cx, cy)銆佸 w 楂?h锛屽彲甯﹀眳涓枃瀛椼€?""
    if textcolor is None:
        textcolor = C_LINE
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle="round,pad=0.4", fc=fc, ec=ec, lw=lw,
                       linestyle=ls, zorder=zorder)
    ax.add_patch(p)
    if text:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                color=textcolor, weight=weight, zorder=zorder + 1)
    return p


def _circle(ax, x, y, r, *, fc="#FFFFFF", ec=C_LINE, lw=1.5, text="", fs=10,
            textcolor=None, zorder=3):
    """鍦嗗舰鑺傜偣锛堝彲甯﹀眳涓枃瀛楋級銆?""
    from matplotlib.patches import Circle
    ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=lw, zorder=zorder))
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                color=textcolor or C_LINE, zorder=zorder + 1)


def _arr(ax, x1, y1, x2, y2, *, color="#5A6472", lw=1.6, ls="-", rad=0.0,
         ms=11, zorder=2):
    """甯︾澶寸嚎娈碉紙rad>0 涓婂嚫 / rad<0 涓嬪嚬锛岀敤 connectionstyle 鐢诲姬锛夈€?""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                linestyle=ls,
                                connectionstyle=f"arc3,rad={rad}",
                                shrinkA=2, shrinkB=2, mutation_scale=ms),
                zorder=zorder)


def _line(ax, x1, y1, x2, y2, *, color="#8A94A0", lw=1.8, ls="-", zorder=2):
    """鏃犵澶寸嚎娈碉紙琛ㄧず娴?杩炵嚎锛夈€?""
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, ls=ls, zorder=zorder)


def _panel(ax):
    """绀烘剰鍥惧瓙鍥撅細0..100 鑷€傚簲鍧愭爣绯?+ 闅愯棌鍒诲害銆?""
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")


def _ptitle(ax, text, color=C_LINE, fs=13):
    """闈㈡澘鏍囬锛堥《閮級銆?""
    ax.text(50, 95.5, text, ha="center", va="center", fontsize=fs,
            weight="bold", color=color)


def _metric_strip(ax, lines):
    """鏋舵瀯闈㈡澘搴曢儴鐨勩€庨噺鍖栨寚鏍囨潯銆忥細鐏板簳鍦嗚妗?+ 涓夎瑕佺偣銆?""
    _rbox(ax, 50, 13, 96, 20, fc="#F0F2F5", ec="#C6CDD6", lw=1.2, zorder=1)
    ax.text(50, 13, "\n".join(lines), ha="center", va="center",
            fontsize=8.6, color="#333B45")


# ---------------------------------------------------------------------------
# 涓変釜鏋舵瀯绀烘剰鍥?# ---------------------------------------------------------------------------
def draw_standard_residual(ax):
    """锛?锛夋爣鍑嗘畫宸細鍙鐩搁偦涓€姝ワ紝鏇存棭灞傚畬鍏ㄤ笉鍙銆?
    甯冨眬锛氬乏渚ф贰鍑虹殑銆庢棫灞傘€忚妭鐐癸紙鎵撁楋級锛屽綋鍓?h_{l-1} 鈫?F_l 鈫?鍔犳硶鍥炴祦 鈫?h_l锛?    搴曢儴缁欏叕寮忎笌銆庡巻鍙蹭笉鍙銆忔敞閲?+ 閲忓寲鎸囨爣鏉°€?    """
    _panel(ax)
    _ptitle(ax, "锛?锛夋爣鍑嗘畫宸?, color=C_RES)

    # 琚贰鍑虹殑鏇存棭灞傦細鍘嗗彶琚涪寮?    for x in (8, 20, 32):
        _line(ax, x - 4, 72, x + 4, 72, color="#C9CFD6", lw=1.4)
        _circle(ax, x, 72, 2.6, fc="#DCE1E8", ec="#AEB6C0", lw=1.2)
        ax.text(x, 80.5, "脳", ha="center", va="center", fontsize=12,
                color="#B7BEC8")
    ax.text(20, 88, "鏇存棭灞傝緭鍑轰笉鍙", ha="center", fontsize=9, color="#9AA3AE")

    # 褰撳墠娈嬪樊娴侊細h_{l-1} -> F_l -> + -> h_l
    _line(ax, 36, 72, 49, 72, color=C_RES, lw=2.4)
    _circle(ax, 49, 72, 4.6, fc=C_RES, ec=C_RES, text=r"$h_{l-1}$", fs=10,
            textcolor="#FFFFFF")
    _arr(ax, 49, 66, 49, 52, color=C_RES, lw=2.0)
    _rbox(ax, 49, 42, 16, 11, text=r"$F_l$", fc="#EBEFF6", ec=C_RES, fs=12,
          weight="bold")
    _arr(ax, 49, 35.5, 65, 72, color=C_RES, lw=2.0, rad=-0.28)
    _line(ax, 53, 72, 65, 72, color=C_RES, lw=2.4)
    _circle(ax, 65, 72, 4.0, fc="#FFFFFF", ec=C_LINE, lw=1.8, text="+", fs=12,
            textcolor=C_RES)
    _line(ax, 69, 72, 83, 72, color=C_RES, lw=2.4)
    _circle(ax, 85, 72, 4.6, fc="#FFFFFF", ec=C_RES, lw=2.0, text=r"$h_l$",
            fs=10, textcolor=C_RES)

    ax.text(58, 20, r"$h_l = h_{l-1} + F_l(h_{l-1})$", ha="center", fontsize=11,
            color=C_RES)
    ax.text(58, 30.5, "鍘嗗彶鍙繚鐣欐渶杩戜竴姝ワ紝\n鍙よ€佸眰鐨勪俊鎭鐩存帴涓㈠純",
            ha="center", va="center", fontsize=9, color="#8A94A0")

    _metric_strip(ax, [
        "鏉冮噸鏉ユ簮锛氬浐瀹?1          input-dependent锛氬惁",
        "鐘舵€侊細1锛堢函寤惰繜绾匡級        I/O 鈮?3d / 灞?,
        "Table 4锛?6 灞傦級锛?.766",
    ])


def draw_mhc(ax):
    """锛?锛塵HC锛歮 鏉″钩琛屾祦锛岀敱銆庡涔?+ 鍙岄殢鏈?Sinkhorn)銆忕煩闃?A 鍦ㄥ悇娴侀棿杞Щ銆?
    甯冨眬锛氬乏鍗?4 鏉℃祦甯﹀悇鑷姸鎬佹敹鏁涜繘 A_l 鈫?F_l 鈫?鎵囧嚭鍥?4 鏉℃祦锛涘簳閮ㄧ粰鏇存柊鍏紡
    涓庛€庣嚎鎬с€佷笌杈撳叆鏃犲叧銆忔敞閲?+ 閲忓寲鎸囨爣鏉°€?    """
    _panel(ax)
    _ptitle(ax, "锛?锛塵HC锛堟祦褰㈢害鏉熻秴杩炴帴锛?, color=C_MHC)

    ys = [82, 70, 58, 46]                      # 4 鏉℃祦鐨?y 鍧愭爣
    for i, y in enumerate(ys):
        _line(ax, 6, y, 42, y, color="#E8B96B", lw=2.2)
        _circle(ax, 26, y, 3.0, fc="#F6DCA8", ec=C_MHC, lw=1.5)
        ax.text(2.6, y, f"娴亄i + 1}", ha="right", va="center", fontsize=9.5,
                color=C_MHC)
    ax.text(26, 89.5, "4 鏉″钩琛屾畫宸祦锛坢=4锛?, ha="center", fontsize=9,
            color="#A0803F")

    # 娴佸叆杞Щ鐭╅樀 A_l
    for y in ys:
        _line(ax, 42, y, 48, y, color="#E8B96B", lw=2.0)
    _arr(ax, 48, ys[0], 52, 70, color=C_MHC, rad=-0.12)
    _arr(ax, 48, ys[1], 52, 64, color=C_MHC)
    _arr(ax, 48, ys[2], 52, 64, color=C_MHC)
    _arr(ax, 48, ys[3], 52, 58, color=C_MHC, rad=0.12)

    # A_l锛氬涔?+ 鍙岄殢鏈?    _rbox(ax, 63, 64, 22, 34, "A_l銆€杞Щ鐭╅樀\n瀛︿範 + 鍙岄殢鏈篭n(Sinkhorn 绾︽潫)",
          fc="#FBF3E2", ec=C_MHC, lw=1.8, fs=8.8)
    # F_l锛氬眰鍙樻崲
    _rbox(ax, 81, 64, 14, 26, r"$F_l$" + "\n灞傚彉鎹?, fc="#EFE6D2", ec=C_MHC,
          lw=1.6, fs=9.5)
    _arr(ax, 74, 64, 73.5, 64, color=C_MHC)    # A -> F

    # 鎵囧嚭鍥?4 鏉℃祦锛堝彸渚х户缁級
    for i, y in enumerate(ys):
        rad = 0.18 if i < 2 else -0.18
        _arr(ax, 88, 64, 92, y, color=C_MHC, rad=rad)
        _line(ax, 92, y, 97, y, color="#E8B96B", lw=2.2)

    ax.text(63, 33,
            r"$H_l = H_{l-1} A_l + f_{l-1}(H_{l-1}\alpha_{l-1})\beta_{l-1}^{\top}$",
            ha="center", fontsize=9, color=C_MHC)
    ax.text(63, 40.5, "鐘舵€?= d脳4锛涙贩鍚堟槸绾挎€у彉鎹紝鏉冮噸涓庤緭鍏ユ棤鍏?,
            ha="center", fontsize=8.8, color="#8A94A0")

    _metric_strip(ax, [
        "鏉冮噸鏉ユ簮锛氬涔?+ 鍙岄殢鏈烘姇褰?   input-dependent锛氬惁",
        "鐘舵€侊細d脳4锛坮ank 鎵╁紶锛?        I/O 鈮?34d / 灞?,
        "Table 4锛?6 灞傦級锛?.747",
    ])


def draw_attnres(ax):
    """锛?锛堿ttnRes锛圔lock锛夛細娣卞害缁?softmax 娉ㄦ剰鍔?鈥斺€?鍧楄〃绀?+ running partial
    浣滀负銆庡巻鍙茬洰褰曘€忥紝浼煡璇?w_l 鎸夊唴瀹硅蒋閫夋嫨锛坕nput-dependent锛夈€?
    甯冨眬锛氬乏渚ф簮鍒楄〃 鈫?RMSNorm 鈫?softmax 鍔犳潈姹傚拰 鈫?h_l锛泈_l 浠庝笅鏂规敞鍏ワ紝
    鍐欐槑 伪 鍏紡 + 閲忓寲鎸囨爣鏉°€?    """
    _panel(ax)
    _ptitle(ax, "锛?锛堿ttnRes锛圔lock 娣卞害娉ㄦ剰鍔涳級", color=C_ARN)

    # 宸︿晶锛氭繁搴﹀巻鍙?= 婧愬垪琛?    srcs = [
        (82.5, C_ARN, "b_0 = embedding锛堟案杩滃彲璇伙級", "#FFFFFF"),
        (72.5, C_ARN_L, "b_1銆€鍧?1 琛ㄧず", "#5B3340"),
        (62.5, "#EFC4C0", "b_2銆€鍧?2 琛ㄧず", "#5B3340"),
    ]
    for y, fc, lab, tc in srcs:
        _rbox(ax, 16, y, 30, 7.2, text=lab, fc=fc, ec="#D8A09B" if fc != C_ARN
              else C_ARN, lw=1.4, fs=8.6, textcolor=tc)
    ax.text(16, 52, r"$\vdots$", ha="center", va="center", fontsize=16,
            color="#B08982")
    _rbox(ax, 16, 41, 30, 7.2, "partial锛堝綋鍓嶅潡杩愯鍜岋級", fc="#FFFFFF",
          ec="#B08982", lw=1.4, fs=8.6, ls="--")
    ax.text(30, 90, "娣卞害鍘嗗彶 = N 鍧楄〃绀?+ 杩愯鍜?, ha="center", fontsize=9,
            color="#9A5A52")

    # 婧?-> RMSNorm
    for y in (82.5, 72.5, 62.5, 41):
        ey = min(max(y, 52), 72)
        _arr(ax, 32, y, 47, ey, color="#C9B0AC", lw=1.4)
    _rbox(ax, 59, 62, 22, 32, "RMSNorm\n锛堥€愭簮褰掍竴鍖栵級", fc="#F7EFEE",
          ec="#B08982", lw=1.6, fs=8.8)

    # 褰掍竴鍖栧悗鐨勬簮 -> softmax 鍔犳潈姹傚拰锛堟敞鎰忓姏鏍稿績锛?    _arr(ax, 70, 62, 76, 62, color=C_ARN, lw=2.0)
    _rbox(ax, 82, 62, 22, 17, r"$\mathrm{softmax}(\cdot)$" + "\n" +
          r"$\sum_i \alpha_i\,v_i$", fc="#FBE3E0", ec=C_ARN,
          lw=1.8, fs=9)

    # 浼煡璇?w_l 娉ㄥ叆
    _arr(ax, 82, 28, 82, 53, color=C_ARN, lw=1.8)
    ax.text(82, 24, "w_l銆€鍙涔犱吉鏌ヨ", ha="center", fontsize=8.8, color=C_ARN)

    # 杈撳嚭 h_l
    _arr(ax, 92, 62, 95, 62, color=C_ARN, lw=2.0)
    _rbox(ax, 97.5, 62, 8, 9, text=r"$h_l$", fc="#FFFFFF", ec=C_ARN, lw=1.8,
          fs=9.5)

    ax.text(82, 90,
            r"$\alpha_i = \mathrm{softmax}(w_l \cdot \mathrm{RMSNorm}(v_i))$",
            ha="center", fontsize=9, color=C_ARN)
    ax.text(58, 32, "鎸夊唴瀹硅蒋閫夋嫨锛歩nput-dependent", ha="center", fontsize=9,
            color="#8A4A42")

    _metric_strip(ax, [
        "鏉冮噸鏉ユ簮锛氬涔?+ 杈撳叆鐩稿叧 softmax    input-dependent锛氭槸",
        "鐘舵€侊細N+1 鍧楋紙鍧楁暟鍥哄畾锛屽唴瀛樻湁鐣岋級    I/O 鈮?5.5d / 灞?,
        "Table 4锛?6 灞傦級锛?.737锛團ull锛? 1.746锛圔lock锛?,
    ])


# ---------------------------------------------------------------------------
# 涓変釜閲忓寲灏忓浘锛堝簳閮ㄨ锛?# ---------------------------------------------------------------------------
def draw_duality(ax):
    """锛坉锛夋繁搴?鏃堕棿瀵瑰伓鎬э細涓夋柟娉曠殑銆庤娉曘€忔部 linear 鈫?softmax 杩涘寲銆?""
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax.text(50, 93, "銆庢繁搴?鏃堕棿瀵瑰伓鎬с€忥紙璁烘枃 搂6.2锛夆€斺€?娣卞害鍘嗗彶鎬庝箞璇伙紝涓庡簭鍒楀悓鏋?,
            ha="center", fontsize=13, weight="bold", color=C_LINE)

    # 涓夋杩涘寲鏉?    stages = [("绛夋潈鍔犳硶", C_RES, "鍙湅鏈€杩戜竴姝?),
              ("鐭╅樀娴佸姩", C_MHC, "linear attention\n鐘舵€佸帇缂╄繘 d脳m"),
              ("鍐呭瀵诲潃", C_ARN, "softmax attention\n鎸夊唴瀹硅蒋閫夋嫨")]
    xs = [22, 50, 78]
    for (lab, col, sub), x in zip(stages, xs):
        _rbox(ax, x, 64, 26, 12, text=lab, fc="#F7F8FA", ec=col, lw=1.8, fs=11,
              weight="bold", textcolor=col)
        ax.text(x, 55, sub, ha="center", va="center", fontsize=8.6,
                color="#5A6472")
    _arr(ax, 35, 64, 45, 64, color=C_GRAY, lw=2.4)
    _arr(ax, 63, 64, 73, 64, color=C_GRAY, lw=2.4)

    # 涓庡簭鍒楀缓妯＄殑绫绘瘮
    ax.text(50, 40, "搴忓垪寤烘ā锛歭inear attention锛圧NN 閫愮姸鎬佸帇缂╋級鈫?softmax attention锛圱ransformer 鍐呭瀵诲潃锛?,
            ha="center", fontsize=10, color="#3D4450")
    ax.text(50, 33, "娣卞害寤烘ā锛氭爣鍑嗘畫宸紙linear锛夆啋 mHC锛坙inear attention锛夆啋 AttnRes锛坰oftmax attention锛?,
            ha="center", fontsize=10, color="#3D4450")

    # 缁撹妗?    _rbox(ax, 50, 16, 94, 15, "", fc="#FDF3F1", ec=C_ARN, lw=1.4)
    ax.text(50, 16,
            "input-dependent 鐨勫唴瀹瑰鍧€鏄?AttnRes 鐩稿 mHC 鐨勫叧閿鐩?鈥斺€?"
            "Table 4锛欰ttnRes 1.737 / 1.746 < mHC 1.747銆?,
            ha="center", va="center", fontsize=9.6, color="#7A2A22")


def draw_io(ax):
    """锛坋锛夋瘡灞傚唴瀛?I/O锛堢浉瀵?1脳d锛夛細mHC 鏈€楂橈紝AttnRes 鏈€浣庛€?""
    methods = ["鏍囧噯娈嬪樊", "mHC", "AttnRes(Block)"]
    io = [3.0, 34.0, 5.5]
    cols = [C_RES, C_MHC, C_ARN]
    bars = ax.bar(methods, io, color=cols, width=0.55)
    for b, v in zip(bars, io):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.7, f"{v:g}路d",
                ha="center", fontsize=11, weight="bold", color=b.get_facecolor())
    ax.set_ylim(0, 40)
    ax.set_title("姣忓眰鍐呭瓨 I/O锛堢浉瀵?1脳d锛?, fontsize=12, weight="bold")
    ax.set_ylabel("鐩稿鍗曚綅锛埫梔锛?)
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.02, 0.04, "mHC(m=4) 鈮?34d锛汚ttnRes(Block) 鈮?5.5d锛屼綆绾?6 鍊?,
            transform=ax.transAxes, fontsize=9, color="#5A6472")


def draw_table4(ax):
    """锛坒锛夎鏂?Table 4 鍥版儜搴︼紙16 hidden 灞傦紝瓒婁綆瓒婂ソ锛夛細AttnRes 鏈€浼樸€?""
    labels = ["鏍囧噯娈嬪樊", "mHC", "AttnRes\n(Full)", "AttnRes\n(Block)"]
    vals = [1.766, 1.747, 1.737, 1.746]
    cols = [C_RES, C_MHC, C_ARN, C_ARN_L]
    bars = ax.bar(labels, vals, color=cols, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                ha="center", fontsize=10, weight="bold", color=b.get_facecolor())
    ax.set_ylim(1.72, 1.78)
    ax.set_title("16 灞傝瑷€妯″瀷鍥版儜搴︼紙璁烘枃 Table 4锛岃秺浣庤秺濂斤級", fontsize=12,
                 weight="bold")
    ax.set_ylabel("鍥版儜搴︼紙perplexity锛?)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(1.737, color=C_ARN, ls="--", lw=1.0, alpha=0.5)
    ax.text(0.02, 0.9, "鏈€浣?= AttnRes Full 1.737", transform=ax.transAxes,
            fontsize=9, color="#7A2A22")


def fig_attnres_comparison() -> tuple[plt.Figure, str]:
    """鐢熸垚 AttnRes vs mHC vs 鏍囧噯娈嬪樊 2x3 瀵规瘮鍥惧苟杩斿洖 (fig, text)銆?
    鐢ㄤ簬 Gradio demo 鐩存帴璋冪敤锛岄伩鍏嶇淮鎶や袱浠戒唬鐮併€?
    Returns:
        (fig, text): matplotlib Figure 鍜岀粨璁烘枃鏈?    """
    fig, axes = plt.subplots(
        2, 3, figsize=(18, 11.5),
        gridspec_kw=dict(height_ratios=[1.08, 0.92], hspace=0.34, wspace=0.26),
    )

    fig.suptitle("娣卞害缁磋繛鎺ョ殑涓夌鑼冨紡锛氭爣鍑嗘畫宸?路 mHC 路 AttnRes",
                 fontsize=16, weight="bold", y=0.985)
    fig.text(0.5, 0.955,
             "鍚屼竴浠姐€庢繁搴﹀巻鍙层€忥紝涓夌璇绘硶 鈥斺€?浠庣瓑鏉冨姞娉曪紝鍒扮姸鎬佸帇缂╋紝鍐嶅埌鎸夊唴瀹瑰鍧€锛坕nput-dependent锛?,
             ha="center", fontsize=11.5, color="#5A6472")

    draw_standard_residual(axes[0][0])
    draw_mhc(axes[0][1])
    draw_attnres(axes[0][2])

    draw_duality(axes[1][0])
    draw_io(axes[1][1])
    draw_table4(axes[1][2])

    fig.text(0.5, 0.012,
             "鏁板€兼潵婧愶細AttnRes 璁烘枃 Table 4锛?6 hidden 灞傦級锛沵HC 璁烘枃 arXiv:2512.24880锛?
             "K3 鎶ュ憡 搂2.2锛圗qs. 8-10锛夈€傛湰椤圭洰宸插疄鐜?AttnRes锛坰rc/models/llm/kimi_k3/modules/attn_res.py锛夛紝"
             "mHC 浠呬綔瀵圭収銆佹湭瀹炵幇銆?,
             ha="center", fontsize=9, color="#8A94A0")

    text = (
        "娣卞害缁磋繛鎺ョ殑涓夌鑼冨紡瀵规瘮锛歕n"
        "1. 鏍囧噯娈嬪樊锛氬彧鍔犺嚜宸辩殑杈撳嚭锛屼笉璇诲巻鍙诧紙绛夋潈鍔犳硶锛宭inear锛塡n"
        "2. mHC锛歮 鏉″钩琛屾祦 + 瀛︿範 + 鍙岄殢鏈?Sinkhorn)杞Щ鐭╅樀锛坙inear attention锛塡n"
        "3. AttnRes锛氭繁搴︾淮 softmax 娉ㄦ剰鍔涳紝鍧楄〃绀?+ running partial锛屼吉鏌ヨ鎸夊唴瀹硅蒋閫夋嫨\n\n"
        "閲忓寲鎸囨爣锛歕n"
        "- 姣忓眰鍐呭瓨 I/O锛氭爣鍑嗘畫宸?3d锛宮HC 34d锛孉ttnRes(Block) 5.5d\n"
        "- 16 灞傚洶鎯戝害锛氭爣鍑嗘畫宸?1.766锛宮HC 1.747锛孉ttnRes(Full) 1.737锛孉ttnRes(Block) 1.746"
    )
    return fig, text


def main() -> None:
    """甯冨眬 2 琛?3 鍒楋細涓婅涓轰笁寮犳灦鏋勭ず鎰忓浘锛屼笅琛屼负涓夊紶閲忓寲灏忓浘銆?""
    fig, text = fig_attnres_comparison()

    for out_dir in (HERE, PICTURE_DIR):
        out_dir.mkdir(parents=True, exist_ok=True)
        png = out_dir / PNG_NAME
        fig.savefig(png, dpi=150, facecolor="white")
        print("saved", png)
    plt.close(fig)
    print(text)


if __name__ == "__main__":
    main()
