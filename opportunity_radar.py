import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
import requests
import talib
import time
import warnings

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

# =========================
# ⚙️ 全局配置
# =========================

# 观察池
WATCHLIST = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT",
    "ADA-USDT", "DOGE-USDT", "LINK-USDT", "AVAX-USDT",
    "SUI-USDT", "APT-USDT", "OP-USDT", "ARB-USDT"
]

# 双周期配置
TF_MAIN = "4h"  # 战术周期
TF_TREND = "1d" # 战略周期

# 阈值配置
SCORE_THRESHOLD = 25  # 单周期得分阈值
RES_CONFIDENCE = 0.8  # 共振置信度系数

# 回溯配置
MAX_LIMIT = 800
CORR_LOOKBACK = 90    # 计算相关性的周期（根K线）

# 经验概率参数
PROB_HORIZON = 6      # 4h * 6 = 24h


# =========================
# 🛠️ 数据与工具层
# =========================

def tf_to_okx_bar(tf: str) -> str:
    if tf.endswith("m"): return tf
    if tf.endswith("h"): return tf[:-1] + "H"
    if tf.endswith("d"): return tf[:-1] + "D"
    return tf

@st.cache_data(ttl=300)
def fetch_ohlcv(inst_id: str, tf: str, limit: int = 500):
    """获取 OKX K线数据，带简单的重试机制"""
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": inst_id, "bar": tf_to_okx_bar(tf), "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=5)
        if r.status_code == 200:
            js = r.json()
            if js.get("code") == "0" and js.get("data"):
                cols = ["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"]
                df = pd.DataFrame(js["data"], columns=cols)
                df["ts"] = pd.to_datetime(df["ts"], unit="ms")
                for c in ["open", "high", "low", "close", "vol"]:
                    df[c] = df[c].astype(float)
                df = df.set_index("ts").sort_index()
                return df
    except Exception:
        pass
    return None

# =========================
# 🧠 核心量化引擎
# =========================

def calc_factors(df: pd.DataFrame):
    """计算核心因子：趋势、动量、波动"""
    if df is None or len(df) < 100: return None
    
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    
    # 1. 基础指标
    rsi = talib.RSI(c, 14)
    adx = talib.ADX(h, l, c, 14)
    atr = talib.ATR(h, l, c, 14)
    
    # MACD
    macd, sig, hist = talib.MACD(c, 12, 26, 9)
    
    # 布林带位置
    u, m, d = talib.BBANDS(c, 20, 2, 2)
    bb_pos = (df["close"] - d) / (u - d)
    
    # 2. 趋势强度分 (Trend Score)
    # EMA 斜率 + MACD 柱状图强弱 + ADX
    ema_fast = talib.EMA(c, 20)
    ema_slow = talib.EMA(c, 50)
    ema_slope = (pd.Series(ema_fast) - pd.Series(ema_slow)) / pd.Series(ema_slow)
    
    trend_raw = np.tanh(ema_slope.fillna(0) * 50) * 0.5 + \
                np.tanh(pd.Series(hist).fillna(0) / (pd.Series(hist).rolling(50).std() + 1e-8)) * 0.3 + \
                (pd.Series(adx).fillna(0) - 20).clip(0, None) / 50 * 0.2
    
    trend_score = (trend_raw * 100).clip(-100, 100)
    
    # 3. 波动率调整后的收益 (Smart Return)
    # 类似夏普比率的逻辑：涨幅 / 波动率
    ret = df["close"].pct_change()
    vol = ret.rolling(20).std()
    smart_ret = ret.rolling(20).mean() / (vol + 1e-8)
    
    # 4. 综合打分
    # 趋势 (60%) + RSI反转 (20%) + 波动调整动量 (20%)
    rev_score = (50 - rsi) * 2  # RSI < 30 -> +40分
    comp_score = 0.6 * trend_score + 0.2 * rev_score + 0.2 * (smart_ret * 100).clip(-50, 50)
    
    # 组装结果
    res = pd.DataFrame(index=df.index)
    res["close"] = c
    res["trend_score"] = trend_score
    res["comp_score"] = comp_score
    res["smart_ret"] = smart_ret
    res["volatility"] = vol
    res["adx"] = adx
    res["rsi"] = rsi
    
    return res

def check_resonance(score_4h, score_1d):
    """判断双周期共振状态"""
    # 同向且都足够强
    if score_4h > SCORE_THRESHOLD and score_1d > SCORE_THRESHOLD:
        return "多头共振", 2.0  # 强力加分
    elif score_4h < -SCORE_THRESHOLD and score_1d < -SCORE_THRESHOLD:
        return "空头共振", 2.0
    # 4h 强，1d 弱/反向 -> 背离
    elif abs(score_4h) > SCORE_THRESHOLD and score_4h * score_1d < 0:
        return "逆势/背离", 0.5 # 降权
    # 其他
    else:
        return "无共振", 1.0

def calc_prob_stats(df, factors, horizon=6,
                    window=10,   # 相似得分窗口 ±window
                    min_sim=30,  # 相似样本数 >= 这个值优先用相似样本
                    min_total=80 # 总历史样本太少时，直接用整体
                    ):
    """
    更严谨的经验概率计算：
    - 优先用“当前得分附近”的相似样本；
    - 相似样本太少 -> 用所有历史样本；
    - 从不再直接返回 0.5；
    - 额外返回：
        - n_samples：实际使用的样本数
        - used_similar：是否使用了相似得分样本
        - edge_z：胜率相对 50% 的 Z 值（简单统计显著性指标）
    """
    if df is None or factors is None:
        return np.nan, np.nan, 0, False, 0.0

    if len(df) <= horizon + 5:
        return np.nan, np.nan, 0, False, 0.0

    if "comp_score" not in factors.columns:
        return np.nan, np.nan, 0, False, 0.0

    closes = df["close"]
    scores = factors["comp_score"]

    # 未来 horizon 根的收益
    fwd_ret = closes.shift(-horizon) / closes - 1

    # 为了做配对，把最后 horizon 根去掉
    hist_scores = scores.iloc[:-horizon]
    fwd_ret = fwd_ret.iloc[:-horizon]

    mask_valid = hist_scores.notna() & fwd_ret.notna()
    hist_scores = hist_scores[mask_valid]
    fwd_ret = fwd_ret[mask_valid]

    if len(fwd_ret) == 0:
        return np.nan, np.nan, 0, False, 0.0

    # 总样本太少：直接用整体分布
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

    # 先用 ±window 范围内的相似样本
    sim_mask = hist_scores.between(curr_score - window, curr_score + window)
    sim_count = sim_mask.sum()

    if sim_count >= min_sim:
        samples = fwd_ret[sim_mask]
        used_similar = True
    elif sim_count >= 10:
        # 样本不是很多，但也可以看一眼
        samples = fwd_ret[sim_mask]
        used_similar = True
    else:
        # 相似样本过少，退回整体历史分布
        samples = fwd_ret
        used_similar = False

    if len(samples) == 0:
        return np.nan, np.nan, 0, False, 0.0

    win_rate = (samples > 0).mean()
    exp_ret = samples.mean()
    n = len(samples)

    # 简单统计显著性：Z 值（|Z|>1.96 ~ 95% 置信）
    edge_z = 0.0 if n == 0 else (win_rate - 0.5) / np.sqrt(0.25 / n)

    return float(win_rate), float(exp_ret), int(n), used_similar, float(edge_z)

# =========================
# 🖥️ Streamlit 页面逻辑
# =========================

st.set_page_config(page_title="Alpha 研究员雷达", layout="wide")

st.title("🔬 Alpha 研究员级机会雷达")
st.caption(f"双周期共振 ({TF_MAIN}+{TF_TREND}) · 风险调整动量 · 组合相关性矩阵")

# 1. 数据并行获取与处理
status_box = st.empty()
status_box.info("正在进行全市场双周期数据扫描与因子计算...")

market_data = []
close_matrix = {} # 用于计算相关性

btc_regime = "未知"

for symbol in WATCHLIST:
    # 获取双周期数据
    df_4h = fetch_ohlcv(symbol, TF_MAIN, MAX_LIMIT)
    df_1d = fetch_ohlcv(symbol, TF_TREND, MAX_LIMIT)
    
    if df_4h is None or df_1d is None: continue
    
    # 计算因子
    fac_4h = calc_factors(df_4h)
    fac_1d = calc_factors(df_1d)
    
    if fac_4h is None or fac_1d is None: continue
    
    # 记录用于计算相关性的序列 (对齐到4h)
    close_matrix[symbol] = df_4h["close"].pct_change().tail(CORR_LOOKBACK)
    
    # 提取关键值
    last_4h = fac_4h.iloc[-1]
    last_1d = fac_1d.iloc[-1]
    
    # BTC Regime 判断 (仅一次)
    if symbol == "BTC-USDT":
        t_score = last_4h["trend_score"]
        v_score = last_4h["volatility"]
        if abs(t_score) > 30 and last_4h["adx"] > 25:
            btc_regime = "趋势市 (Trending)"
        elif last_4h["volatility"] > fac_4h["volatility"].quantile(0.8):
            btc_regime = "高波震荡 (Volatile)"
        else:
            btc_regime = "低波盘整 (Ranging)"

    # 共振判断
    res_label, res_weight = check_resonance(last_4h["comp_score"], last_1d["comp_score"])
    
      # 经验概率（带样本数 & 显著性）
    win_rate, exp_ret, n_samples, used_similar, edge_z = calc_prob_stats(
        df_4h, fac_4h, PROB_HORIZON
    )

    # 核心：Alpha 排序分（加入显著性权重）
    raw_alpha = (last_4h["comp_score"] + last_1d["comp_score"] * 0.5)

    # 统计显著性权重：样本多且 Z 值绝对值大 -> 给予更高权重，最多放大到 1.5 倍
    sig_weight = 1.0
    if n_samples >= 30:
        sig_weight = min(1.5, 0.5 + 0.1 * abs(edge_z))  # Z 每增加 1，多给 0.1，最多 1.5

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
        "Prob_Mode": "相似分布" if used_similar else "整体分布",
        "Edge_Z": edge_z
    })

status_box.success("全市场扫描完成。")

# 2. 市场概览 (Regime)
st.markdown("---")
col_reg, col_best = st.columns([1, 3])

with col_reg:
    color = "#00C805" if "趋势" in btc_regime else "#FF4B4B" if "高波" in btc_regime else "#FFA500"
    st.markdown(f"""
    <div style="padding:15px; border-radius:10px; border:1px solid {color}; background:#111;">
        <h3 style="margin:0; color:{color}">{btc_regime}</h3>
        <p style="margin:5px 0 0 0; color:#888; font-size:12px;">BTC 4h 市场状态</p>
    </div>
    """, unsafe_allow_html=True)

# 3. 核心雷达表 (DataFrame)
df_res = pd.DataFrame(market_data).set_index("Symbol")
df_res = df_res.sort_values("Alpha_Score", ascending=False)

# 美化表格显示
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

# 样式映射
def color_resonance(val):
    color = "#888"
    if "多头" in val: color = "#00C805"
    elif "空头" in val: color = "#FF4B4B"
    elif "背离" in val: color = "#FFA500"
    return f'color: {color}; font-weight: bold'

def color_score(val):
    color = "#888"
    if val > 30: color = "#00C805"
    elif val < -30: color = "#FF4B4B"
    return f'color: {color}'

st.subheader("📋 智能机会筛选列表")
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

# 4. 深度分析与风控
st.markdown("---")
col_deep, col_risk = st.columns([2, 1])

with col_deep:
    st.subheader("🔍 深度透视：Top 1 机会")
    top_symbol = df_res.index[0]
    samples = sel_row["Prob_N"]
    edge_z = sel_row["Edge_Z"]

    st.write("---")
    st.markdown("**统计视角补充说明：**")
    st.write(f"- 本次经验概率估计共使用历史样本：**{int(samples)}** 个；")
    st.write(f"- 胜率相对 50% 的 Z 值约为：**{edge_z:.2f}**，"
             "一般认为 |Z| > 1.96 对应约 95% 的统计显著性；"
             "样本越多且 Z 值越大，说明这个优势越“可靠”。")
    
    # 选择器
    sel_symbol = st.selectbox("选择币种查看详情", df_res.index, index=0)
    
    sel_row = df_res.loc[sel_symbol]
    
    # 绘制共振图
    # 这里我们不做简单的 K 线，而是做一个 '信号强度' 对比图
    fig_gauge = go.Figure()
    
    fig_gauge.add_trace(go.Bar(
        y=["1d 趋势", "4h 战术", "历史胜率(偏移)"],
        x=[sel_row["1d_Score"], sel_row["4h_Score"], (sel_row["Win_Rate"]-0.5)*200],
        orientation='h',
        marker=dict(
            color=list(map(lambda x: '#00C805' if x>0 else '#FF4B4B', 
                           [sel_row["1d_Score"], sel_row["4h_Score"], sel_row["Win_Rate"]-0.5]))
        )
    ))
    
    fig_gauge.update_layout(
        title=f"{sel_symbol} 信号多维拆解",
        xaxis_title="信号强度 (左负右正)",
        template="plotly_dark",
        height=300
    )
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # 文字解读
    res_note = "✅ 极佳机会" if "共振" in sel_row["Resonance"] else "⚠️ 存在分歧/背离"
    st.info(f"""
    **研究员解读**：
    该币种当前 Alpha 得分为 **{sel_row['Alpha_Score']:.1f}**。
    双周期状态为 **{sel_row['Resonance']}** ({res_note})。
    在类似当前评分的历史情境下，未来 24h 上涨概率为 **{sel_row['Win_Rate']:.1%}**。
    风险调整后的动量因子（Smart Return）为 **{sel_row['Smart_Ret']:.2f}**，
    { "波动率较低，上涨质量高" if abs(sel_row['Smart_Ret']) > 0.5 else "波动率较高，注意风险" }。
    """)

with col_risk:
    st.subheader("🛡️ 组合风控：相关性热力图")
    st.caption("避免同时持有颜色过深（相关性高）的币种")
    
    # 计算相关性矩阵
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
        st.warning("数据不足，无法计算相关性矩阵")

st.markdown("---")
st.caption("Alpha 研究员雷达 v2.0 | 基于双周期共振与波动率调整模型")


