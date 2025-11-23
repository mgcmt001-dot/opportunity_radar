import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import requests
import talib
import warnings

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

# =========================
# 配置
# =========================

# 观察篮子（你可以按喜好增减）
WATCHLIST = [
    "BTC-USDT",
    "ETH-USDT",
    "SOL-USDT",
    "XRP-USDT",
    "ADA-USDT",
    "DOGE-USDT",
    "LINK-USDT",
    "AVAX-USDT",
    "TON-USDT",
    "OP-USDT",
    "ARB-USDT"
]

MAIN_TF = "4h"

MAX_LIMIT = 800
MIN_BARS_FOR_FACTORS = 80

# 阈值：稳妥偏保守
LONG_THRESHOLD = 30
SHORT_THRESHOLD = -30

PERIOD_RET_LOOKBACK = 18     # 4h * 18 ≈ 3 天
MONTH_WINDOW_DAYS = 30

# 经验概率：未来 horizon 根 K 线（4h * 6 ≈ 1 天）
PROB_HORIZON_BARS = 6


# =========================
# 工具 & 数据获取
# =========================

def tf_to_okx_bar(tf: str) -> str:
    if tf.endswith("m"):
        return tf
    if tf.endswith("h"):
        return tf[:-1] + "H"
    if tf.endswith("d"):
        return tf[:-1] + "D"
    return tf


@st.cache_data(ttl=180)
def fetch_okx_klines(inst_id: str, tf: str, limit: int = 500):
    url = "https://www.okx.com/api/v5/market/candles"
    params = {
        "instId": inst_id,
        "bar": tf_to_okx_bar(tf),
        "limit": limit
    }
    try:
        r = requests.get(url, params=params, timeout=10)
    except Exception as e:
        st.error(f"{inst_id} 请求 OKX 失败：{e}")
        return None

    if r.status_code != 200:
        st.error(f"{inst_id} OKX HTTP 错误：{r.status_code}")
        return None

    js = r.json()
    if js.get("code") != "0":
        st.error(f"{inst_id} OKX API 错误：{js.get('msg')}")
        return None

    data = js.get("data", [])
    if not data:
        st.warning(f"{inst_id} OKX 返回空数据")
        return None

    cols = [
        "ts", "open", "high", "low",
        "close", "volume", "volCcy",
        "volCcyQuote", "confirm"
    ]
    df = pd.DataFrame(data, columns=cols)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")

    float_cols = ["open", "high", "low", "close", "volume"]
    for c in float_cols:
        df[c] = df[c].astype(float)

    df.set_index("ts", inplace=True)
    df.sort_index(inplace=True)
    return df


# =========================
# 因子 & 统计模块
# =========================

def compute_factor_series(df: pd.DataFrame) -> pd.DataFrame:
    """和你之前那套类似：趋势 + 反转 + 波动 + 综合评分"""
    if df is None or len(df) < MIN_BARS_FOR_FACTORS:
        return pd.DataFrame(index=df.index if df is not None else None)

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    rsi = talib.RSI(close, timeperiod=14)
    adx = talib.ADX(high, low, close, timeperiod=14)
    ema_fast = talib.EMA(close, timeperiod=20)
    ema_slow = talib.EMA(close, timeperiod=50)
    macd, macd_signal, macd_hist = talib.MACD(
        close, fastperiod=12, slowperiod=26, signalperiod=9
    )
    atr = talib.ATR(high, low, close, timeperiod=14)
    bb_upper, bb_mid, bb_lower = talib.BBANDS(
        close, timeperiod=20, nbdevup=2, nbdevdn=2
    )

    ret = pd.Series(close, index=df.index).pct_change()
    vol20 = ret.rolling(20).std()

    fac = pd.DataFrame(index=df.index)
    fac["rsi"] = rsi
    fac["adx"] = adx
    fac["ema_fast"] = ema_fast
    fac["ema_slow"] = ema_slow
    fac["macd"] = macd
    fac["macd_signal"] = macd_signal
    fac["macd_hist"] = macd_hist
    fac["atr"] = atr
    fac["bb_upper"] = bb_upper
    fac["bb_mid"] = bb_mid
    fac["bb_lower"] = bb_lower
    fac["volatility"] = vol20

    fac["ema_slope"] = (fac["ema_fast"] - fac["ema_slow"]) / fac["ema_slow"]
    fac["bb_position"] = (df["close"] - fac["bb_lower"]) / (fac["bb_upper"] - fac["bb_lower"])
    fac["bb_position"] = fac["bb_position"].clip(0, 1)

    trend_raw = np.zeros(len(df))
    trend_raw += np.tanh(fac["ema_slope"].fillna(0) * 50)

    macd_std = fac["macd_hist"].rolling(50).std()
    macd_norm = fac["macd_hist"] / (macd_std + 1e-8)
    trend_raw += np.tanh(macd_norm.fillna(0))

    adx_comp = (fac["adx"] - 20) / 25
    adx_comp[fac["adx"] < 20] = 0
    trend_raw += adx_comp.fillna(0)

    fac["trend_score"] = (trend_raw * 20).clip(-50, 50)

    reversal_raw = np.zeros(len(df))
    reversal_raw += (50 - fac["rsi"]) / 25.0
    reversal_raw += (0.5 - fac["bb_position"]) * 2.0
    fac["reversal_score"] = (reversal_raw * 20).clip(-50, 50)

    base_vol = fac["volatility"].rolling(100).median()
    vol_ratio = fac["volatility"] / (base_vol + 1e-8)
    fac["volatility_score"] = ((vol_ratio - 1.0) * 30).clip(-50, 50)

    comp = (
        0.5 * fac["trend_score"] +
        0.3 * fac["reversal_score"] +
        0.2 * np.sign(fac["trend_score"]) * fac["volatility_score"].abs()
    )
    fac["composite_score"] = comp.clip(-100, 100)

    return fac


def compute_forward_prob_stats(df: pd.DataFrame, fac: pd.DataFrame,
                               horizon: int, score_window: float = 10.0,
                               min_samples: int = 40):
    """
    在当前得分附近（±score_window）找历史样本，
    统计未来 horizon 根的经验上涨概率 / 期望收益。
    """
    if df is None or fac is None:
        return None
    if len(df) <= horizon + 5:
        return None
    if "composite_score" not in fac.columns:
        return None

    scores = fac["composite_score"]
    closes = df["close"]

    fwd_ret = (closes.shift(-horizon) / closes - 1).iloc[:-horizon]
    scores_hist = scores.iloc[:-horizon]

    mask = scores_hist.notna() & fwd_ret.notna()
    scores_hist = scores_hist[mask]
    fwd_ret = fwd_ret[mask]

    if len(fwd_ret) < min_samples:
        return None

    score_now = scores.iloc[-1]
    if pd.isna(score_now):
        return None

    similar = scores_hist.between(score_now - score_window, score_now + score_window)
    if similar.sum() >= min_samples:
        rets = fwd_ret[similar]
    else:
        rets = fwd_ret

    if len(rets) == 0:
        return None

    prob_up = (rets > 0).mean()
    exp_ret = rets.mean()
    worst_10 = rets.quantile(0.1)
    best_10 = rets.quantile(0.9)

    return {
        "prob_up": float(prob_up),
        "exp_ret": float(exp_ret),
        "worst_10": float(worst_10),
        "best_10": float(best_10),
        "n_samples": int(len(rets))
    }


def analyze_symbol(inst_id: str, df: pd.DataFrame):
    """对单个币种做统计分析，返回一行 dict"""
    fac = compute_factor_series(df)
    if fac is None or fac.empty:
        return None

    last = fac.iloc[-1]
    price = float(df["close"].iloc[-1])

    # 近 N 根涨跌幅
    if len(df) > PERIOD_RET_LOOKBACK:
        period_ret = df["close"].iloc[-1] / df["close"].iloc[-PERIOD_RET_LOOKBACK] - 1
    else:
        period_ret = np.nan

    # 本月高低点百分位（近 30 天）
    if len(df) > 20:
        cutoff = df.index[-1] - timedelta(days=MONTH_WINDOW_DAYS)
        df_win = df[df.index >= cutoff]
        if len(df_win) < 10:
            df_win = df
        hi = df_win["high"].max()
        lo = df_win["low"].min()
        last_close = df_win["close"].iloc[-1]
        if hi > lo:
            month_pct = (last_close - lo) / (hi - lo)
        else:
            month_pct = np.nan
    else:
        month_pct = np.nan

    # 历史评分分位数
    hist_scores = fac["composite_score"].dropna()
    if len(hist_scores) >= 60:
        score = float(last["composite_score"])
        score_pct = (hist_scores < score).mean()
    else:
        score = float(last["composite_score"])
        score_pct = np.nan

    # 经验概率
    prob_stats = compute_forward_prob_stats(df, fac, horizon=PROB_HORIZON_BARS)
    if prob_stats is not None:
        prob_up = prob_stats["prob_up"]
        exp_ret = prob_stats["exp_ret"]
        n_samples = prob_stats["n_samples"]
    else:
        prob_up = np.nan
        exp_ret = np.nan
        n_samples = 0

    row = {
        "symbol": inst_id,
        "price": price,
        "trend_score": float(last["trend_score"]),
        "reversal_score": float(last["reversal_score"]),
        "volatility_score": float(last["volatility_score"]),
        "composite_score": score,
        "rsi": float(last["rsi"]),
        "adx": float(last["adx"]),
        "atr": float(last["atr"]) if not np.isnan(last["atr"]) else np.nan,
        "bb_position": float(last["bb_position"]),
        "period_return": period_ret,
        "month_percentile": month_pct,
        "score_percentile": score_pct,
        "prob_up": prob_up,
        "exp_ret": exp_ret,
        "prob_n": n_samples
    }
    return row


def classify_opportunity(row, long_thr=LONG_THRESHOLD, short_thr=SHORT_THRESHOLD):
    """基于一堆指标，给出“机会类型”标签和一句解释"""

    score = row["composite_score"]
    trend = row["trend_score"]
    rev = row["reversal_score"]
    month_pct = row["month_percentile"]
    period_ret = row["period_return"]
    prob_up = row["prob_up"]
    exp_ret = row["exp_ret"]
    score_pct = row["score_percentile"]

    label = "中性观察"
    note = "当前信号较为中性，可耐心等待更明确的机会。"

    # 为了避免 NaN 把逻辑短路，做一些默认
    if pd.isna(prob_up):
        prob_up = 0.5
    if pd.isna(exp_ret):
        exp_ret = 0.0

    # 趋势多头候选
    if (
        pd.notna(score) and pd.notna(month_pct)
        and score >= long_thr
        and prob_up >= 0.55
        and 0.25 <= month_pct <= 0.9
    ):
        label = "趋势多头候选"
        note = "综合评分偏多，历史相似状态下上涨概率较高，且价格不在极端高位，适合考虑顺势做多或加仓。"

    # 趋势空头候选
    elif (
        pd.notna(score) and pd.notna(month_pct)
        and score <= short_thr
        and prob_up <= 0.45
        and 0.1 <= month_pct <= 0.85
    ):
        label = "趋势空头候选"
        note = "综合评分偏空，历史相似状态下上涨概率偏低，若支持做空，可考虑顺势布局空单或减少现货敞口。"

    # 超跌反弹博弈
    elif (
        pd.notna(month_pct) and pd.notna(period_ret)
        and month_pct < 0.25
        and period_ret < -0.08
        and rev > 0
    ):
        label = "超跌反弹博弈"
        note = "价格接近本月低位且近期跌幅较大，反转因子转向有利，多头可以考虑小仓位左侧博弈反弹。"

    # 高位风险
    elif (
        pd.notna(month_pct) and month_pct > 0.9
        and pd.notna(period_ret) and period_ret > 0.08
    ):
        label = "高位风险·谨慎"
        note = "价格接近本月高位且近期涨幅明显，继续追高的风险加大，更适合考虑分批止盈或减仓。"

    # 震荡市观望
    elif (
        abs(score) < 10
        and 0.45 <= prob_up <= 0.55
        and (month_pct is not np.nan and 0.3 <= month_pct <= 0.7)
    ):
        label = "震荡市观望"
        note = "综合评分靠近中性，历史统计上涨概率接近五五开，且价格位于区间中部，更适合观望或区间短线。"

    # 机会评分：用于排序（不是绝对意义）
    base = {
        "趋势多头候选": 2.0,
        "趋势空头候选": 1.8,
        "超跌反弹博弈": 1.5,
        "高位风险·谨慎": 0.8,
        "震荡市观望": 0.7,
        "中性观察": 0.5
    }.get(label, 0.5)

    op_score = base + (prob_up - 0.5) * 2 + score / 100.0

    # 轻微考虑极端分位（分位太高/太低减一点）
    if pd.notna(score_pct):
        if score_pct > 0.95 or score_pct < 0.05:
            op_score -= 0.3

    return label, note, float(op_score)


def detect_regime_from_btc(df: pd.DataFrame):
    """用 BTC 4h 因子，判断整个市场大环境 Regime"""
    fac = compute_factor_series(df)
    if fac is None or fac.empty:
        return "未知", "BTC 数据不足，无法判断当前市场状态。"

    last = fac.iloc[-1]
    trend = last.get("trend_score", np.nan)
    adx = last.get("adx", np.nan)
    vol_score = last.get("volatility_score", np.nan)

    if pd.isna(trend) or pd.isna(adx) or pd.isna(vol_score):
        return "未知", "关键因子缺失，暂时无法判断市场状态。"

    at = abs(trend)
    av = abs(vol_score)

    if at > 20 and adx > 25:
        return "趋势市", "BTC 4h 呈现明显趋势行情，顺势类信号通常更具统计优势，逆势博弈要格外注意控制风险。"
    elif at < 10 and adx < 18 and av < 10:
        return "低波震荡市", "BTC 4h 趋势不强、波动有限，更像箱体震荡，区间交易和均值回归更适配，追涨杀跌效率偏低。"
    elif av > 20:
        return "高波动混乱市", "BTC 4h 波动率显著放大，行情方向噪声大，建议降低杠杆和单笔仓位，耐心等待结构更清晰。"
    else:
        return "过渡阶段", "BTC 4h 介于趋势与震荡之间，处于酝酿新方向的阶段，信号可靠度有限，可以适当减仓观望。"


# =========================
# Streamlit 页面
# =========================

st.set_page_config(
    page_title="📊 多币种加密机会雷达（4h）",
    layout="wide"
)

st.title("📊 多币种加密机会雷达 · 4h 周期")
st.caption("多因子评分 + 分位数视角 + 经验概率 + 机会分类 · 仅做决策辅助，不构成投资建议")

st.sidebar.header("说明")
st.sidebar.write("本页面一次性扫描一篮子主流币，基于 4 小时 K 线给出：")
st.sidebar.markdown(
"""
- 趋势 / 反转 / 波动因子综合评分  
- 近几天涨跌 & 本月高低点百分位  
- 当前得分在历史中的分位数  
- 在类似状态下，未来约 1 天上涨的经验概率 & 期望收益  
- 基于以上信息的“机会类型”标签与排序
"""
)
st.sidebar.caption("所有结果基于历史统计，不保证未来表现。")

# 数据获取
status = st.empty()
status.info("正在从 OKX 批量获取 4h 行情数据……")

data_map = {}
for inst in WATCHLIST:
    df = fetch_okx_klines(inst, MAIN_TF, limit=MAX_LIMIT)
    if df is None or df.empty:
        continue
    data_map[inst] = df

if not data_map:
    status.error("所有币种数据获取失败，请稍后重试。")
    st.stop()

status.success(f"已成功获取 {len(data_map)} 个币种的 4h 行情数据。")

# BTC Regime
btc_df = data_map.get("BTC-USDT")
if btc_df is not None:
    regime_label, regime_comment = detect_regime_from_btc(btc_df)
else:
    regime_label, regime_comment = "未知", "未能获取 BTC-USDT 数据，无法判断整体 Regime。"

st.subheader("📌 整体市场大环境（以 BTC 4h 为代表）")
reg_color = "#16c784" if regime_label == "趋势市" else "#ea3943" if regime_label == "高波动混乱市" else "#f0ad4e"

st.markdown(
    f"""
    <div style="border-radius:8px; border:1px solid {reg_color}; padding:10px; background-color:#050505;">
        <span style="color:{reg_color}; font-weight:bold;">当前 4h 市场状态：{regime_label}</span><br>
        <span style="color:#dddddd; font-size:12px;">{regime_comment}</span>
    </div>
    """,
    unsafe_allow_html=True
)

# 分析每个币种
rows = []
for inst, df in data_map.items():
    row = analyze_symbol(inst, df)
    if row is None:
        continue
    label, note, op_score = classify_opportunity(row)
    row["label"] = label
    row["label_note"] = note
    row["opportunity_score"] = op_score
    rows.append(row)

if not rows:
    st.warning("有效因子数据不足，暂时无法生成机会雷达。")
    st.stop()

df_symbols = pd.DataFrame(rows).set_index("symbol")
df_symbols.sort_values("opportunity_score", ascending=False, inplace=True)

st.markdown("---")
st.subheader("🎯 多币种机会一览（按综合机会评分排序）")

# 展示表格
show_cols = [
    "opportunity_score",
    "label",
    "price",
    "composite_score",
    "trend_score",
    "reversal_score",
    "volatility_score",
    "period_return",
    "month_percentile",
    "score_percentile",
    "prob_up",
    "exp_ret"
]

display = df_symbols[show_cols].copy()

display = display.rename(columns={
    "opportunity_score": "机会评分（排序用）",
    "label": "机会类型",
    "price": "价格",
    "composite_score": "综合评分",
    "trend_score": "趋势分",
    "reversal_score": "反转分",
    "volatility_score": "波动分",
    "period_return": f"近{PERIOD_RET_LOOKBACK}根涨跌幅",
    "month_percentile": "本月高低点百分位",
    "score_percentile": "当前评分历史分位",
    "prob_up": f"未来{PROB_HORIZON_BARS}根上涨概率",
    "exp_ret": f"未来{PROB_HORIZON_BARS}根期望收益"
})

fmt_dict = {
    "机会评分（排序用）": "{:.2f}",
    "价格": "{:.4f}",
    "综合评分": "{:.1f}",
    "趋势分": "{:.1f}",
    "反转分": "{:.1f}",
    "波动分": "{:.1f}",
    f"近{PERIOD_RET_LOOKBACK}根涨跌幅": "{:.2%}",
    "本月高低点百分位": "{:.1%}",
    "当前评分历史分位": "{:.1%}",
    f"未来{PROB_HORIZON_BARS}根上涨概率": "{:.1%}",
    f"未来{PROB_HORIZON_BARS}根期望收益": "{:.2%}"
}

st.dataframe(
    display.style.format(fmt_dict, na_rep="—"),
    use_container_width=True
)

# 详情查看
st.markdown("---")
st.subheader("🔍 单币详细结构与信号解读")

default_symbol = df_symbols.index[0]
sel_symbol = st.selectbox(
    "选择一个币种查看详细 4h 结构：",
    df_symbols.index.tolist(),
    index=df_symbols.index.tolist().index(default_symbol)
)

sel_df = data_map[sel_symbol]
sel_fac = compute_factor_series(sel_df)
sel_row = df_symbols.loc[sel_symbol]

col_a, col_b = st.columns([2, 1])

with col_a:
    st.markdown(f"#### {sel_symbol} · 4h K 线与 EMA 结构")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=sel_df.index,
        open=sel_df["open"],
        high=sel_df["high"],
        low=sel_df["low"],
        close=sel_df["close"],
        name="4h K 线",
        increasing_line_color="green",
        decreasing_line_color="red"
    ))

    if not sel_fac.empty:
        fig.add_trace(go.Scatter(
            x=sel_df.index,
            y=sel_fac["ema_fast"],
            name="EMA 20",
            line=dict(color="deepskyblue", width=1.2)
        ))
        fig.add_trace(go.Scatter(
            x=sel_df.index,
            y=sel_fac["ema_slow"],
            name="EMA 50",
            line=dict(color="orange", width=1.2)
        ))

    fig.update_layout(
        height=500,
        template="plotly_dark",
        xaxis_title="时间",
        yaxis_title="价格 (USDT)"
    )
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.markdown(f"#### {sel_symbol} 机会解读")

    label = sel_row["label"]
    note = sel_row["label_note"]
    score = sel_row["composite_score"]
    prob_up = sel_row["prob_up"]
    exp_ret = sel_row["exp_ret"]
    month_pct = sel_row["month_percentile"]
    score_pct = sel_row["score_percentile"]
    period_ret = sel_row["period_return"]

    color = "#16c784" if "多头" in label or "反弹" in label else "#ea3943" if "空头" in label or "风险" in label else "#f0ad4e"

    st.markdown(
        f"""
        <div style="border-radius:8px; border:1px solid {color}; padding:10px; background-color:#050505;">
            <div style="color:{color}; font-weight:bold; font-size:16px; margin-bottom:6px;">
                机会类型：{label}
            </div>
            <div style="color:#dddddd; font-size:13px; margin-bottom:6px;">
                {note}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("---")
    st.markdown("**关键数字一览：**")
    st.write(f"- 综合评分：{score:.1f}")
    if pd.notna(period_ret):
        st.write(f"- 最近 {PERIOD_RET_LOOKBACK} 根累计涨跌：{period_ret:.2%}")
    if pd.notna(month_pct):
        st.write(f"- 当前在本月高低点区间的百分位：{month_pct:.1%}")
    if pd.notna(score_pct):
        st.write(f"- 当前评分在历史中的分位：{score_pct:.1%}")
    if pd.notna(prob_up) and pd.notna(exp_ret):
        st.write(f"- 在历史相似得分下，未来 {PROB_HORIZON_BARS} 根上涨概率约：{prob_up:.1%}")
        st.write(f"- 对应期望收益约：{exp_ret:.2%}")

st.markdown("---")
st.caption("""
本页面仅基于历史行情和统计方法做机会筛选和风险提示，不构成任何投资建议。  
加密货币波动性极高，请结合自身风险承受能力，谨慎决策。
""")
