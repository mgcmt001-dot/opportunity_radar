import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import time
import talib
from okx import Market
import warnings
warnings.filterwarnings('ignore')

# ========================
# CONFIG
# ========================
STABLECOINS = ['USDT', 'USDC', 'DAI']
PAIRS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT', 'DOGE/USDT']
TIMEFRAMES = ['15m', '1h', '4h', '1d']
N_SIGNALS = 100  # 回测最近N次信号
N_DAYS_BACK = 90  # 模拟过去90天
MAX_CAPITAL = 10000  # 用户资金（美元）
ATR_MULTIPLIER = 2.5  # 止损倍数
KELLY_FRACTION = 0.5  # 减半凯利（保守）

# ========================
# DATA FETCHERS
# ========================

@st.cache_data(ttl=300)  # 缓存5分钟
def fetch_okx_klines(pair, timeframe, limit=200):
    market = Market()
    data = market.get_candlesticks(instId=pair, bar=timeframe, limit=limit)
    if data['code'] != '0':
        st.error(f"OKX API Error: {data['msg']}")
        return None
    df = pd.DataFrame(data['data'], columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirm'
    ])
    df = df.astype({
        'open': float, 'high': float, 'low': float, 'close': float, 'volume': float
    })
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df.sort_index()

@st.cache_data(ttl=600)
def fetch_greed_fear_index():
    url = "https://api.alternative.me/fng/"
    response = requests.get(url, timeout=10)
    data = response.json()['data'][0]
    return {
        'value': int(data['value']),
        'classification': data['value_classification'],
        'timestamp': pd.to_datetime(int(data['timestamp']), unit='s')
    }

# ========================
# FACTOR CALCULATIONS
# ========================

def calculate_factors(df):
    """计算多因子评分：趋势、反转、波动率"""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values

    # 趋势因子：ADX + EMA斜率
    adx = talib.ADX(high, low, close, timeperiod=14)[-1]
    ema20 = talib.EMA(close, timeperiod=20)[-1]
    ema50 = talib.EMA(close, timeperiod=50)[-1]
    ema_slope = (ema20 - ema50) / ema50  # EMA20/50斜率
    trend_score = (adx / 25) * np.sign(ema_slope) if adx > 25 else 0

    # 反转因子：RSI + Bollinger Band位置
    rsi = talib.RSI(close, timeperiod=14)[-1]
    bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    bb_upper, bb_middle, bb_lower = bb_upper[-1], bb_middle[-1], bb_lower[-1]
    bb_position = (close[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
    reversal_score = -2 * (rsi - 50) / 100  # RSI偏离中心得分
    reversal_score += (0.5 - bb_position) * 0.5  # 超买超卖位置惩罚

    # 波动率因子：ATR + 波动率扩张
    atr = talib.ATR(high, low, close, timeperiod=14)[-1]
    volatility = np.std(close[-20:]) / np.mean(close[-20:])
    volatility_score = (volatility - 0.01) * 50  # 基准1%，每高出0.01=+50分

    # 综合评分：趋势（40%）+ 反转（30%）+ 波动率（30%）
    composite_score = (
        trend_score * 0.4 +
        reversal_score * 0.3 +
        volatility_score * 0.3
    )
    composite_score = np.clip(composite_score, -100, 100)

    return {
        'trend_score': trend_score,
        'reversal_score': reversal_score,
        'volatility_score': volatility_score,
        'composite_score': composite_score,
        'adx': adx,
        'rsi': rsi,
        'atr': atr,
        'volatility': volatility,
        'bb_position': bb_position,
        'ema20': ema20,
        'ema50': ema50
    }

# ========================
# SIGNAL GENERATION & BACKTEST
# ========================

class SignalHistory:
    def __init__(self):
        self.signals = []  # 存储 [timestamp, score, direction, pnl, entry, exit, stop_loss]

    def add_signal(self, timestamp, score, direction, entry, exit, stop_loss, pnl):
        self.signals.append({
            'timestamp': timestamp,
            'score': score,
            'direction': direction,
            'entry': entry,
            'exit': exit,
            'stop_loss': stop_loss,
            'pnl': pnl
        })
        if len(self.signals) > N_SIGNALS:
            self.signals.pop(0)

    def get_stats(self):
        if not self.signals:
            return {}
        df = pd.DataFrame(self.signals)
        win_rate = (df['pnl'] > 0).mean() * 100
        avg_return = df['pnl'].mean()
        sharpe = df['pnl'].mean() / (df['pnl'].std() + 1e-8)
        max_drawdown = (df['pnl'].cumsum().cummax() - df['pnl'].cumsum()).max()
        return {
            'win_rate': win_rate,
            'avg_pnl': avg_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'total_signals': len(df),
            'long_signals': (df['direction'] == 'long').sum(),
            'short_signals': (df['direction'] == 'short').sum()
        }

# 全局信号历史
signal_history = SignalHistory()

def generate_signal(pair, df, timeframe, capital=MAX_CAPITAL):
    """生成交易信号 + 止盈止损 + 仓位建议"""
    factors = calculate_factors(df)
    score = factors['composite_score']
    atr = factors['atr']
    close = df['close'].iloc[-1]

    # 信号逻辑：基于综合得分
    direction = None
    if score > 20:
        direction = 'long'
    elif score < -20:
        direction = 'short'

    if not direction:
        return None, None, None, None, factors

    # 止损：ATR倍数
    stop_loss = close - ATR_MULTIPLIER * atr if direction == 'long' else close + ATR_MULTIPLIER * atr
    take_profit = close + 2 * ATR_MULTIPLIER * atr if direction == 'long' else close - 2 * ATR_MULTIPLIER * atr

    # 仓位计算：凯利公式 + 波动率调整
    win_prob = 0.55  # 基准胜率（历史均值）
    win_loss_ratio = 2.0  # 盈亏比 2:1
    kelly = win_prob - (1 - win_prob) / win_loss_ratio
    kelly_fraction = kelly * KELLY_FRACTION
    risk_per_trade = capital * kelly_fraction  # 风险资金
    dollar_risk = abs(close - stop_loss)  # 每单位风险
    position_size = risk_per_trade / dollar_risk  # 币数

    # OKX合约：BTC/USDT 1张 = 0.001 BTC，我们按币数计算
    if 'BTC' in pair:
        position_size = round(position_size / 0.001) * 0.001  # 以0.001张为单位
    elif 'ETH' in pair:
        position_size = round(position_size / 0.01) * 0.01  # 以0.01张为单位
    else:
        position_size = round(position_size)

    # 记录信号
    signal_history.add_signal(
        timestamp=df.index[-1],
        score=score,
        direction=direction,
        entry=close,
        exit=take_profit,
        stop_loss=stop_loss,
        pnl=0  # 暂时为0，回测时填充
    )

    return direction, position_size, stop_loss, take_profit, factors

# ========================
# BACKTEST ENGINE
# ========================

def backtest_strategy(pair, days=N_DAYS_BACK):
    """模拟过去N天的机械交易"""
    df_daily = fetch_okx_klines(pair, '1d', limit=days + 50)
    if df_daily is None:
        return None

    capital = MAX_CAPITAL
    equity_curve = [capital]
    positions = []  # 存储每笔交易
    last_signal_time = None

    for i in range(50, len(df_daily)):
        df_slice = df_daily.iloc[:i+1]
        direction, size, sl, tp, factors = generate_signal(pair, df_slice, '1d', capital)

        if direction and (last_signal_time is None or df_slice.index[-1] > last_signal_time + timedelta(days=1)):
            # 模拟开仓
            entry = df_slice['close'].iloc[-1]
            next_close = df_daily['close'].iloc[i+1] if i+1 < len(df_daily) else entry
            pnl = (next_close - entry) * size if direction == 'long' else (entry - next_close) * size
            capital += pnl
            equity_curve.append(capital)

            # 记录真实PnL
            signal_history.signals[-1]['pnl'] = pnl
            last_signal_time = df_slice.index[-1]
        else:
            equity_curve.append(equity_curve[-1])

    return pd.Series(equity_curve, index=df_daily.index[50:])

# ========================
# STREAMLIT APP
# ========================

st.set_page_config(page_title="📈 华尔街级加密量化分析助手", layout="wide")
st.title("📈 华尔街级加密量化分析助手 —— 多周期因子模型 + 自动仓位系统")
st.caption("无需代理 · 实时OKX数据 · 机械回测 · 风险控制 · 情绪辅助")

# 侧边栏配置
st.sidebar.header("🔧 配置")
selected_pair = st.sidebar.selectbox("选择交易对", PAIRS, index=0)
capital_input = st.sidebar.number_input("您的资金规模 (USD)", min_value=100, max_value=1000000, value=MAX_CAPITAL, step=1000)
ATR_MULTIPLIER = st.sidebar.slider("止损倍数 (ATR)", 1.0, 5.0, 2.5, 0.1)
KELLY_FRACTION = st.sidebar.slider("凯利比例（保守）", 0.1, 1.0, 0.5, 0.1)
MAX_CAPITAL = capital_input

# 获取数据
st.info(f"正在获取 {selected_pair} 的实时数据...")

# 多周期K线
dfs = {}
for tf in TIMEFRAMES:
    dfs[tf] = fetch_okx_klines(selected_pair, tf)

if any(df is None for df in dfs.values()):
    st.error("❌ 数据获取失败，请检查网络或OKX API状态。")
    st.stop()

# 获取贪婪恐惧指数
gf_data = fetch_greed_fear_index()

# ========================
# 主面板：多周期K线 + 指标
# ========================

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(f"📊 {selected_pair} 多周期K线图（15m → 1d）")
    fig = go.Figure()

    # 主图：1D K线
    df = dfs['1d']
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='1D K线',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))

    # 添加EMA20/50
    factors = calculate_factors(df)
    fig.add_trace(go.Scatter(x=df.index, y=df['close'].ewm(span=20).mean(), name='EMA20', line=dict(color='blue', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['close'].ewm(span=50).mean(), name='EMA50', line=dict(color='orange', width=1)))

    # 添加ATR通道（波动率带）
    atr = factors['atr']
    upper_band = df['close'] + 2 * atr
    lower_band = df['close'] - 2 * atr
    fig.add_trace(go.Scatter(x=df.index, y=upper_band, name='ATR上轨', line=dict(color='gray', dash='dot'), opacity=0.5))
    fig.add_trace(go.Scatter(x=df.index, y=lower_band, name='ATR下轨', line=dict(color='gray', dash='dot'), opacity=0.5))

    # 添加贪婪恐惧指数（次坐标轴）
    fig.add_trace(go.Scatter(
        x=[df.index[-1]], y=[df['close'].iloc[-1] * 1.05],
        mode='text',
        text=[f"📈 恐惧/贪婪: {gf_data['value']} ({gf_data['classification']})"],
        textposition="top center",
        textfont=dict(color="purple", size=14),
        showlegend=False,
        yaxis="y2"
    ))

    fig.update_layout(
        title=f"{selected_pair} - 多周期因子分析",
        yaxis_title="价格 (USDT)",
        xaxis_title="时间",
        yaxis2=dict(
            title="情绪指数",
            overlaying="y",
            side="right",
            showgrid=False,
            range=[df['close'].min() * 0.98, df['close'].max() * 1.02]
        ),
        height=600,
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🎯 信号与仓位建议")
    direction, size, stop_loss, take_profit, factors = generate_signal(selected_pair, dfs['1d'], '1d', MAX_CAPITAL)

    if direction:
        st.success(f"✅ **当前信号：{direction.upper()}**")
        st.metric("综合评分", f"{factors['composite_score']:.1f}", delta=f"{factors['composite_score'] - 0:.1f}")
        st.metric("建议仓位", f"{size:.6f} {selected_pair.split('/')[0]}", delta=f"${size * dfs['1d']['close'].iloc[-1]:.2f}")
        st.metric("止损价", f"${stop_loss:.2f}", delta=f"{stop_loss - dfs['1d']['close'].iloc[-1]:.2f}")
        st.metric("止盈价", f"${take_profit:.2f}", delta=f"{take_profit - dfs['1d']['close'].iloc[-1]:.2f}")
    else:
        st.warning("⚠️ 无明确信号：市场震荡中，建议观望")

    # 风格剖面雷达图
    st.subheader("🌀 多因子风格剖面")
    categories = ['趋势因子', '反转因子', '波动率因子']
    values = [factors['trend_score'], factors['reversal_score'], factors['volatility_score']]
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='当前风格',
        line=dict(color='cyan')
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[-50, 50]),
            angularaxis=dict(rotation=90)
        ),
        title="多因子风格雷达图",
        height=300,
        template="plotly_dark"
    )
    st.plotly_chart(fig_radar)

# ========================
# 回测与统计面板
# ========================

st.subheader("📊 历史信号回测分析")

col1, col2, col3 = st.columns(3)

# 1. 最近N次信号盈亏分布
stats = signal_history.get_stats()
if stats:
    df_hist = pd.DataFrame(signal_history.signals)
    if len(df_hist) > 1:
        fig_hist = px.histogram(df_hist, x='pnl', nbins=20, title="最近100次信号盈亏分布", color_discrete_sequence=['#00FF99'])
        fig_hist.add_vline(x=df_hist['pnl'].mean(), line_dash="dash", line_color="red", annotation_text="平均盈亏")
        fig_hist.add_vline(x=0, line_dash="dot", line_color="white")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col1:
        st.metric("胜率", f"{stats['win_rate']:.1f}%")
    with col2:
        st.metric("平均盈亏", f"${stats['avg_pnl']:.2f}")
    with col3:
        st.metric("夏普比率", f"{stats['sharpe']:.2f}")

# 2. 模拟净值曲线
st.subheader("📈 机械执行回测：过去90天净值曲线")
with st.spinner("正在模拟过去90天的机械交易..."):
    equity_series = backtest_strategy(selected_pair, N_DAYS_BACK)

if equity_series is not None:
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(x=equity_series.index, y=equity_series, mode='lines', name='模拟净值', line=dict(color='gold', width=3)))
    fig_equity.add_trace(go.Scatter(x=[equity_series.index[0], equity_series.index[-1]], y=[MAX_CAPITAL, MAX_CAPITAL], mode='lines', name='初始资金', line=dict(color='gray', dash='dash')))
    fig_equity.update_layout(
        title=f"{selected_pair} 过去90天机械交易净值曲线",
        yaxis_title="账户价值 (USD)",
        xaxis_title="时间",
        height=400,
        template="plotly_dark"
    )
    st.plotly_chart(fig_equity)

    max_dd = (equity_series.cummax() - equity_series).max()
    final_value = equity_series.iloc[-1]
    roi = (final_value - MAX_CAPITAL) / MAX_CAPITAL * 100
    st.success(f"🎯 回测结果：最终净值 ${final_value:.2f} | 总收益 {roi:+.1f}% | 最大回撤 {max_dd:.1f}%")

# ========================
# 情绪辅助面板
# ========================

st.subheader("🧠 市场情绪辅助：贪婪与恐惧指数")
col1, col2 = st.columns([1, 2])

with col1:
    color = "green" if gf_data['value'] > 70 else "red" if gf_data['value'] < 30 else "yellow"
    st.markdown(f"""
    <div style="text-align:center; padding:20px; background-color:{color}20; border-radius:10px; border:1px solid {color}">
        <h3 style="color:{color}">{gf_data['value']}</h3>
        <p style="color:white; margin:0">{gf_data['classification']}</p>
        <small style="color:lightgray">{gf_data['timestamp'].strftime('%Y-%m-%d %H:%M')}</small>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    > **贪婪与恐惧解读**：
    > - **0–24**：极度恐惧 → 潜在买入机会  
    > - **25–49**：恐惧 → 谨慎观察  
    > - **50**：中性  
    > - **51–74**：贪婪 → 警惕回调  
    > - **75–100**：极度贪婪 → 考虑减仓  
    >  
    > **策略建议**：当综合评分 > +30 且 指数 > 70 → 警惕顶部；当评分 < -30 且 指数 < 20 → 强力买入信号增强
    """)

# ========================
# 底部说明
# ========================

st.markdown("---")
st.caption("""
💡 **本系统为量化分析助手，非投资建议**。  
所有信号基于历史统计与因子模型，市场存在极端波动风险。  
请始终使用止损，勿重仓。  
© 2025 华尔街量化实验室 · 代码开源 · 可部署于 GitHub
""")