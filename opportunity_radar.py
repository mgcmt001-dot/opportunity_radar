import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
import requests
import talib
import warnings

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

# =========================
# ⚙️ 全局配置
# =========================

WATCHLIST = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT",
    "ADA-USDT", "DOGE-USDT", "LINK-USDT", "AVAX-USDT",
    "SUI-USDT", "APT-USDT", "OP-USDT", "ARB-USDT"
]

TF_MAIN = "4h"   # 战术周期
TF_TREND = "1d"  # 战略周期

SCORE_THRESHOLD = 25        # 单周期判定强弱的阈值
MAX_LIMIT = 800
CORR_LOOKBACK = 90          # 相关性滚动窗口（根K线）
PROB_HORIZON = 6            # 未来 6 根 4h ≈ 24 小时

# =========================
# 🛠️ 工具 & 数据获取
# =========================

def tf_to_okx_bar(tf: str) -> str:
    if tf.endswith("m"): return tf
    if tf.endswith("h"): return tf[:-1] + "H"
    if tf.endswith("d"): return tf[:-1] + "D"
    return tf

@st.cache_data(ttl=300)
def fetch_ohlcv(inst_id: str, tf: str, limit: int = 500):
    """从 OKX 获取 K 线数据"""
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": inst_id, "bar": tf_to_okx_bar(tf), "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=5)
        if r.status_code != 200:
            return None
        js = r.json()
        if js.get("code") != "0" or not js.get("data"):
            return None
        cols = ["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"]
        df = pd.DataFrame(js["data"], columns=cols)
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        for c in ["open", "high", "low", "close", "vol"]:
            df[c] = df[c].astype(float)
        df = df.set_index("ts").sort_index()
        return df
    except Exception:
        return None

# =========================
# 🧠 因子 & 概率引擎
# =========================

def calc_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算：
    - 趋势分 trend_score
    - 综合分 comp_score
    - 风险调整动量 smart_ret
    - 波动率 volatility
    - RSI / ADX
    """
    if df is None or len(df) < 100:
        return None

    c = df["close"].values
    h = df["high"].values
    l = df["low"].values

    rsi = talib.RSI(c, 14)
    adx = talib.ADX(h, l, c, 14)
    atr = talib.ATR(h, l, c, 14)

    macd, sig, hist = talib.MACD(c, 12, 26, 9)
    u, m, d = talib.BBANDS(c, 20, 2, 2)
    # bb_pos = (df["close"] - d) / (u - d)  # 这版暂时不用

    ema_fast = talib.EMA(c, 20)
    ema_slow = talib.EMA(c, 50)
    ema_slope = (pd.Series(ema_fast) - pd.Series(ema_slow)) / pd.Series(ema_slow)

    # 趋势分：EMA斜率 + MACD强度 + ADX
    trend_raw = np.tanh(ema_slope.fillna(0) * 50) * 0.5
    macd_std = pd.Series(hist).rolling(50).std()
    trend_raw += np.tanh(pd.Series(hist).fillna(0) / (macd_std + 1e-8)) * 0.3
    trend_raw += ((pd.Series(adx).fillna(0) - 20).clip(0, None) / 50) * 0.2
    trend_score = (trend_raw * 100).clip(-100, 100)

    # 风险调整动量：类似“短期夏普”
    ret = df["close"].pct_change()
    vol = ret.rolling(20).std()
    smart_ret = ret.rolling(20).mean() / (vol + 1e-8)

    # RSI 反转分
    rev_score = (50 - pd.Series(rsi)) * 2  # RSI<30 -> +40

    comp_score = 0.6 * trend_score + 0.2 * rev_score + 0.2 * (smart_ret * 100).clip(-50, 50)

    res = pd.DataFrame(index=df.index)
    res["close"] = df["close"]
    res["trend_score"] = trend_score
    res["comp_score"] = comp_score
    res["smart_ret"] = smart_ret
    res["volatility"] = vol
    res["adx"] = adx
    res["rsi"] = rsi
    res["atr"] = atr
    return res


def check_resonance(score_4h: float, score_1d: float):
    """双周期共振标签 + 共振权重"""
    if np.isnan(score_4h) or np.isnan(score_1d):
        return "数据不足", 1.0

    if score_4h > SCORE_THRESHOLD and score_1d > SCORE_THRESHOLD:
        return "多头共振", 2.0
    if score_4h < -SCORE_THRESHOLD and score_1d < -SCORE_THRESHOLD:
        return "空头共振", 2.0
    if abs(score_4h) > SCORE_THRESHOLD and score_4h * score_1d < 0:
        return "逆势/背离", 0.5
    return "无共振", 1.0


def calc_prob_stats(df: pd.DataFrame, factors: pd.DataFrame,
                    horizon: int = 6,
                    window: float = 10.0,
                    min_sim: int = 30,
                    min_total: int = 80):
    """
    更严谨的经验概率估计：
    - 优先使用 当前得分 ±window 内的历史样本；
    - 如果相似样本 < min_sim，则退回全部样本；
    - 永远不会无脑给 0.5，而是给出真实历史比例；
    - 返回：
        win_rate: 上涨概率
        exp_ret: 期望收益
        n_samples: 实际使用的样本数
        used_similar: 是否用“相似得分”子样本
        edge_z: 胜率相对 0.5 的 Z 值（显著性粗略指标）
    """
    if df is None or factors is None:
        return np.nan, np.nan, 0, False, 0.0

    if len(df) <= horizon + 5:
        return np.nan, np.nan, 0, False, 0.0

    if "comp_score" not in factors.columns:
        return np.nan, np.nan, 0, False, 0.0

    closes = df["close"]
    scores = factors["comp_score"]

    fwd_ret = closes.shift(-horizon) / closes - 1
    hist_scores = scores.iloc[:-horizon]
    fwd_ret = fwd_ret.iloc[:-horizon]

    mask_valid = hist_scores.notna() & fwd_ret.notna()
    hist_scores = hist_scores[mask_valid]
    fwd_ret = fwd_ret[mask_valid]

    if len(fwd_ret) == 0:
        return np.nan, np.nan, 0, False, 0.0

    # 总样本极少：直接用整体
    if len(fwd_ret) < min_total:
        samples = fwd_ret
        win_rate = (samples > 0).mean()
        exp_ret = samples.mean()
        n = len(samples)
        edge_z = 0.0 if n == 0 else (win_rate - 0.5) / np.sqrt(0.25 / n)
        return float(win_rate), float(exp_ret), int(n), False, float(edge_z)

    curr_score = scores.iloc[-1]
    if pd.isna(curr_score):
        samples = fwd_ret
        win_rate = (samples > 0).mean()
        exp_ret = samples.mean()
        n = len(samples)
        edge_z = 0.0 if n == 0 else (win_rate - 0.5) / np.sqrt(0.25 / n)
        return float(win_rate), float(exp_ret), int(n), False, float(edge_z)

    # 先用 ±window 的相似得分区间
    sim_mask = hist_scores.between(curr_score - window, curr_score + window)
    sim_count = sim_mask.sum()

    if sim_count >= min_sim:
        samples = fwd_ret[sim_mask]
        used_similar = True
    elif sim_count >= 10:
        samples = fwd_ret[sim_mask]
        used_similar = True
    else:
        samples = fwd_ret
        used_similar = False

    if len(samples) == 0:
        return np.nan, np.nan, 0, False, 0.0

    win_rate = (samples > 0).mean()
    exp_ret = samples.mean()
    n = len(samples)
    edge_z = 0.0 if n == 0 else (win_rate - 0.5) / np.sqrt(0.25 / n)

    return float(win_rate), float(exp_ret), int(n), used_similar, float(edge_z)

# =========================
# 🖥️ Streamlit 页面
# =========================

st.set_page_config(page_title="Alpha 研究员雷达", layout="wide")

st.title("🔬 Alpha 研究员级机会雷达")
st.caption(f"双周期共振 ({TF_MAIN} + {TF_TREND}) · 风险调整动量 · 经验胜率 · 相关性矩阵")

status_box = st.empty()
status_box.info("正在进行双周期扫描与因子计算...")

market_data = []
close_matrix = {}
btc_regime = "未知"

for symbol in WATCHLIST:
    df_4h = fetch_ohlcv(symbol, TF_MAIN, MAX_LIMIT)
    df_1d = fetch_ohlcv(symbol, TF_TREND, MAX_LIMIT)

    if df_4h is None or df_1d is None:
        continue

    fac_4h = calc_factors(df_4h)
    fac_1d = calc_factors(df_1d)

    if fac_4h is None or fac_1d is None:
        continue

    # 用于相关性：4h 收益序列
    close_matrix[symbol] = df_4h["close"].pct_change().tail(CORR_LOOKBACK)

    last_4h = fac_4h.iloc[-1]
    last_1d = fac_1d.iloc[-1]

    # BTC 市场状态
    if symbol == "BTC-USDT":
        t_score = last_4h["trend_score"]
        vol_now = last_4h["volatility"]
        adx_now = last_4h["adx"]
        vol_q80 = fac_4h["volatility"].quantile(0.8)

        if abs(t_score) > 30 and adx_now > 25:
            btc_regime = "趋势市 (Trending)"
        elif pd.notna(vol_now) and pd.notna(vol_q80) and vol_now > vol_q80:
            btc_regime = "高波震荡 (Volatile)"
        else:
            btc_regime = "低波盘整 (Ranging)"

    # 双周期共振
    res_label, res_weight = check_resonance(
        last_4h["comp_score"], last_1d["comp_score"]
    )

    # 经验概率
    win_rate, exp_ret, n_samples, used_sim, edge_z = calc_prob_stats(
        df_4h, fac_4h, PROB_HORIZON
    )

    # Alpha 排序分：多因子 + 共振 + 胜率 + 显著性
    raw_alpha = (last_4h["comp_score"] + 0.5 * last_1d["comp_score"])

    sig_weight = 1.0
    if n_samples >= 30:
        sig_weight = min(1.5, 0.5 + 0.1 * abs(edge_z))  # 样本多且Z值大 → 放大一点权重

    alpha_score = (raw_alpha * res_weight + (win_rate - 0.5) * 100) * sig_weight

    market_data.append({
        "Symbol": symbol,
        "Price": df_4h["close"].iloc[-1],
        "4h_Score": last_4h["comp_score"],
        "1d_Score": last_1d["comp_score"],
        "Resonance": res_label,
        "Win_Rate": win_rate,
        "Exp_Ret": exp_ret,
        "Smart_Ret": last_4h["smart_ret"],
        "Alpha_Score": alpha_score,
        "Vol": last_4h["volatility"],
        "Prob_N": n_samples,
        "Prob_Mode": "相似分布" if used_sim else "整体分布",
        "Edge_Z": edge_z
    })

if not market_data:
    status_box.error("所有币种数据获取或因子计算失败。")
    st.stop()
else:
    status_box.success(f"已完成 {len(market_data)} 个币种的扫描。")

df_res = pd.DataFrame(market_data).set_index("Symbol")
df_res = df_res.sort_values("Alpha_Score", ascending=False)

# =========================
# 市场状态 & 核心表格
# =========================

st.markdown("---")
col_reg, _ = st.columns([1, 3])

with col_reg:
    color = "#00C805" if "趋势" in btc_regime else "#FF4B4B" if "高波" in btc_regime else "#FFA500"
    st.markdown(
        f"""
        <div style="padding:15px; border-radius:10px; border:1px solid {color}; background:#111;">
            <h3 style="margin:0; color:{color}">{btc_regime}</h3>
            <p style="margin:5px 0 0 0; color:#888; font-size:12px;">以 BTC-USDT 4h 为代表的当前市场状态</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.subheader("📋 智能机会筛选列表（按 Alpha 排序）")

show_df = df_res[[
    "Alpha_Score", "Resonance", "Price",
    "4h_Score", "1d_Score", "Win_Rate", "Exp_Ret",
    "Smart_Ret", "Prob_N", "Edge_Z"
]].copy()

show_df.columns = [
    "Alpha 排序分", "双周期共振", "当前价格",
    "4h 评分", "1d 评分", "历史胜率", "期望收益",
    "风险调整动量", "样本数", "胜率偏离Z值"
]

def color_resonance(val):
    if "多头" in val:
        return 'color: #00C805; font-weight: bold'
    elif "空头" in val:
        return 'color: #FF4B4B; font-weight: bold'
    elif "逆势" in val:
        return 'color: #FFA500; font-weight: bold'
    return 'color: #BBBBBB'

def color_score(val):
    if val > 30:
        return 'color: #00C805'
    elif val < -30:
        return 'color: #FF4B4B'
    return 'color: #DDDDDD'

st.dataframe(
    show_df.style.format({
        "Alpha 排序分": "{:.1f}",
        "当前价格": "{:.4f}",
        "4h 评分": "{:.1f}",
        "1d 评分": "{:.1f}",
        "历史胜率": "{:.1%}",
        "期望收益": "{:.2%}",
        "风险调整动量": "{:.2f}",
        "样本数": "{:.0f}",
        "胜率偏离Z值": "{:.2f}"
    }).map(color_resonance, subset=["双周期共振"])
      .map(color_score, subset=["4h 评分", "1d 评分"]),
    use_container_width=True,
    height=500
)

# =========================
# 深度拆解 & 风控
# =========================

st.markdown("---")
col_deep, col_risk = st.columns([2, 1])

with col_deep:
    st.subheader("🔍 深度透视：信号拆解")

    default_symbol = df_res.index[0]
    sel_symbol = st.selectbox("选择一个币种查看细节", df_res.index.tolist(),
                              index=df_res.index.tolist().index(default_symbol))

    sel_row = df_res.loc[sel_symbol]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        y=["1d 趋势", "4h 综合", "历史胜率偏移"],
        x=[sel_row["1d_Score"],
           sel_row["4h_Score"],
           (sel_row["Win_Rate"] - 0.5) * 200],
        orientation="h",
        marker=dict(
            color=[
                "#00C805" if sel_row["1d_Score"] > 0 else "#FF4B4B",
                "#00C805" if sel_row["4h_Score"] > 0 else "#FF4B4B",
                "#00C805" if sel_row["Win_Rate"] > 0.5 else "#FF4B4B"
            ]
        )
    ))
    fig_bar.update_layout(
        title=f"{sel_symbol} 多维信号拆解",
        xaxis_title="信号强度（左负右正）",
        template="plotly_dark",
        height=320
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # 文本解释
    res_note = "偏强机会" if "共振" in sel_row["Resonance"] else (
        "逆势结构，适合短线" if "逆势" in sel_row["Resonance"] else "无明显共振，信号一般"
    )

    st.info(
        f"**研究员视角解读**：\n\n"
        f"- 当前 Alpha 排序分：**{sel_row['Alpha_Score']:.1f}**（兼顾多因子得分、共振与历史胜率）。\n"
        f"- 双周期状态：**{sel_row['Resonance']}**（{res_note}）。\n"
        f"- 在历史上“当前得分附近”的情境中，未来约 24 小时上涨概率约为：**{sel_row['Win_Rate']:.1%}**，"
        f"期望收益约 **{sel_row['Exp_Ret']:.2%}**。\n"
        f"- 样本数：**{int(sel_row['Prob_N'])}**，胜率相对 50% 的 Z 值约 **{sel_row['Edge_Z']:.2f}**，"
        f"{'在统计上有一定显著性（|Z|>1.96≈95% 置信）' if abs(sel_row['Edge_Z'])>1.96 else '暂不算非常显著，更多作为参考'}。\n"
        f"- 风险调整动量（Smart Ret）：**{sel_row['Smart_Ret']:.2f}**，"
        f"{'说明在单位波动风险下，这段时间上涨质量较高。' if abs(sel_row['Smart_Ret'])>0.5 else '上涨/下跌伴随较大噪音，注意回撤风险。'}"
    )

with col_risk:
    st.subheader("🛡️ 组合相关性热力图")
    st.caption("避免同时重仓高度相关（深色接近 1）的币种。")

    if len(close_matrix) > 1:
        corr_df = pd.DataFrame(close_matrix).corr()
        fig_corr = px.imshow(
            corr_df,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1
        )
        fig_corr.update_layout(
            height=400,
            template="plotly_dark",
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("数据不足，无法计算相关性矩阵。")

st.markdown("---")
st.caption("""
本工具以研究员视角提供多因子、概率与风险分析，不构成任何投资建议。  
历史统计不代表未来结果，加密资产波动剧烈，请严格控制仓位与风险。
""")
