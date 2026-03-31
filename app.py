import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from scraper import fetch_stock_daily, fetch_stock_name
from preprocessor import build_chronos_input, build_forecast_index
from predictor import predict_close

st.set_page_config(
    page_title="Chronos-2 股票预测系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _init_session():
    if "history_df" not in st.session_state:
        st.session_state.history_df = None
    if "pred_df" not in st.session_state:
        st.session_state.pred_df = None
    if "stock_name" not in st.session_state:
        st.session_state.stock_name = ""
    if "symbol" not in st.session_state:
        st.session_state.symbol = ""


_init_session()


# ── Sidebar ──────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ 参数配置")

symbol_input = st.sidebar.text_input(
    "股票代码",
    value="600519",
    help="输入纯数字代码即可，自动识别市场（6开头=沪市sh，其他=深市sz）",
)

days_input = st.sidebar.slider(
    "历史数据天数",
    min_value=60,
    max_value=800,
    value=365,
    step=10,
)

pred_len = st.sidebar.slider(
    "预测天数",
    min_value=1,
    max_value=120,
    value=30,
    step=1,
)

run_btn = st.sidebar.button("🚀 开始预测", type="primary", width="stretch")


# ── Main ─────────────────────────────────────────────────────────────────
st.title("📈 Chronos-2 股票预测系统")

with st.expander("📖 项目说明", expanded=True):
    st.markdown(
        """
### 项目简介

本项目是一个基于 **Amazon Chronos-2** 时间序列基础模型的股票价格预测工具。

- **核心模型**: [Amazon Chronos-2](https://huggingface.co/amazon/chronos-2) — 120M 参数的 T5 编码器架构时间序列基础模型
- **输入维度**: 日线 K 线数据 OCHLV（Open / Close / High / Low / Volume）
- **输出维度**: 未来 Close（收盘价）预测值，含分位数置信区间
- **数据源**: 新浪财经 HTTP API（A 股日线行情）
- **前端框架**: [Streamlit](https://streamlit.io)
- **可视化**: [Plotly](https://plotly.com) 交互式 K 线图

### 模型能力

| 特性 | 说明 |
|------|------|
| 上下文长度 | 最长 8192 时间步 |
| 最大预测长度 | 最长 1024 时间步 |
| 协变量支持 | 支持 Open / High / Low / Volume 作为历史协变量 |
| 概率预测 | 21 个分位数水平 (0.01 ~ 0.99) |
| 零样本预测 | 无需微调即可直接推理 |

### 免责声明

> ⚠️ 本工具仅供学习和研究使用，**不构成任何投资建议**。股市有风险，投资需谨慎。
> 模型预测结果受多种因素影响，不保证准确性。

### 作者

- GitHub: [AK60000](https://github.com/AK60000)
        """
    )

st.markdown("---")

# ── Run prediction ───────────────────────────────────────────────────────
if run_btn:
    with st.spinner(f"正在获取 {symbol_input} 的历史数据..."):
        try:
            raw_df = fetch_stock_daily(symbol_input, days=days_input)
            stock_name = fetch_stock_name(symbol_input)
            st.session_state.stock_name = stock_name
            st.session_state.symbol = symbol_input
        except Exception as e:
            st.error(f"数据获取失败: {e}")
            st.stop()

    if raw_df.empty:
        st.warning("未获取到有效数据，请检查股票代码是否正确。")
        st.stop()

    st.session_state.history_df = raw_df

    with st.spinner("正在运行 Chronos-2 模型预测..."):
        try:
            close_arr, past_covs = build_chronos_input(raw_df)
            pred_tensor = predict_close(
                close_arr,
                past_covariates=past_covs,
                prediction_length=pred_len,
            )

            last_date = raw_df["date"].max()
            future_dates = build_forecast_index(last_date, pred_len)

            model_quantiles = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
            pred_records = []
            for i, dt in enumerate(future_dates):
                row = {"date": dt}
                for q in [0.1, 0.5, 0.9]:
                    q_idx = model_quantiles.index(q)
                    row[f"pred_{q}"] = pred_tensor[0, q_idx, i].item()
                pred_records.append(row)

            pred_df = pd.DataFrame(pred_records)
            st.session_state.pred_df = pred_df

        except Exception as e:
            st.error(f"模型预测失败: {e}")
            st.stop()


# ── Display results ──────────────────────────────────────────────────────
hist = st.session_state.history_df
pred = st.session_state.pred_df

if hist is not None and not hist.empty:
    name = st.session_state.stock_name or st.session_state.symbol
    st.subheader(f"📊 {name} ({st.session_state.symbol})")

    col_a, col_b, col_c, col_d = st.columns(4)
    latest = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) > 1 else latest
    chg = latest["close"] - prev["close"]
    chg_pct = (chg / prev["close"]) * 100 if prev["close"] != 0 else 0

    col_a.metric("最新收盘价", f"{latest['close']:.2f}", f"{chg:+.2f} ({chg_pct:+.2f}%)")
    col_b.metric("最高价", f"{latest['high']:.2f}")
    col_c.metric("最低价", f"{latest['low']:.2f}")
    col_d.metric("成交量", f"{latest['volume']:,.0f}")

    st.markdown("---")

    hist_dates = hist["date"].dt.strftime("%Y-%m-%d").tolist()
    hist_idx = list(range(len(hist)))

    # ── Historical K-line ────────────────────────────────────────────
    st.subheader("🕯️ 历史 K 线图")

    fig_hist = go.Figure()

    fig_hist.add_trace(
        go.Candlestick(
            x=hist_idx,
            open=hist["open"],
            high=hist["high"],
            low=hist["low"],
            close=hist["close"],
            name="K线",
            increasing_line_color="#ef4444",
            decreasing_line_color="#22c55e",
            increasing_fillcolor="#ef4444",
            decreasing_fillcolor="#22c55e",
            line=dict(width=0.5),
            hovertemplate="<b>%{customdata}</b><br>开: %{open:.2f}<br>高: %{high:.2f}<br>低: %{low:.2f}<br>收: %{close:.2f}<extra></extra>",
            customdata=hist_dates,
        )
    )

    fig_hist.add_trace(
        go.Scatter(
            x=hist_idx,
            y=hist["close"].rolling(5).mean(),
            name="MA5",
            line=dict(color="#f59e0b", width=1.5),
        )
    )
    fig_hist.add_trace(
        go.Scatter(
            x=hist_idx,
            y=hist["close"].rolling(20).mean(),
            name="MA20",
            line=dict(color="#3b82f6", width=1.5),
        )
    )

    fig_hist.update_layout(
        xaxis=dict(
            title="日期",
            tickvals=hist_idx[::max(1, len(hist_idx) // 10)],
            ticktext=hist_dates[::max(1, len(hist_idx) // 10)],
            tickangle=-45,
        ),
        yaxis_title="价格",
        height=500,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_hist, width="stretch")

    # ── Volume chart ─────────────────────────────────────────────────
    st.subheader("📊 成交量")
    fig_vol = go.Figure()
    colors = ["#ef4444" if c >= o else "#22c55e" for o, c in zip(hist["open"], hist["close"])]
    fig_vol.add_trace(
        go.Bar(
            x=hist_idx,
            y=hist["volume"],
            marker_color=colors,
            name="成交量",
            hovertemplate="<b>%{customdata}</b><br>成交量: %{y:,.0f}<extra></extra>",
            customdata=hist_dates,
        )
    )
    fig_vol.update_layout(
        xaxis=dict(
            title="日期",
            tickvals=hist_idx[::max(1, len(hist_idx) // 10)],
            ticktext=hist_dates[::max(1, len(hist_idx) // 10)],
            tickangle=-45,
        ),
        yaxis_title="成交量",
        height=250,
        template="plotly_white",
        showlegend=False,
    )
    st.plotly_chart(fig_vol, width="stretch")

    # ── Prediction ───────────────────────────────────────────────────
    if pred is not None and not pred.empty:
        st.markdown("---")
        st.subheader("🔮 未来走势预测")

        pred_date_strs = pred["date"].dt.strftime("%Y-%m-%d").tolist()
        pred_offset = len(hist_idx)
        pred_idx = list(range(pred_offset, pred_offset + len(pred)))
        all_idx = hist_idx + pred_idx
        all_dates = hist_dates + pred_date_strs

        pred_mid = pred["pred_0.5"].values
        pred_low = pred["pred_0.1"].values
        pred_high = pred["pred_0.9"].values

        fig_pred = go.Figure()

        fig_pred.add_trace(
            go.Candlestick(
                x=hist_idx,
                open=hist["open"],
                high=hist["high"],
                low=hist["low"],
                close=hist["close"],
                name="历史K线",
                increasing_line_color="#ef4444",
                decreasing_line_color="#22c55e",
                increasing_fillcolor="#ef4444",
                decreasing_fillcolor="#22c55e",
                line=dict(width=0.5),
                hovertemplate="<b>%{customdata}</b><br>开: %{open:.2f}<br>高: %{high:.2f}<br>低: %{low:.2f}<br>收: %{close:.2f}<extra></extra>",
                customdata=hist_dates,
            )
        )

        fig_pred.add_trace(
            go.Scatter(
                x=pred_idx,
                y=pred_mid,
                name="预测收盘价",
                line=dict(color="#8b5cf6", width=2.5),
                mode="lines+markers",
                marker=dict(size=4),
                hovertemplate="<b>%{customdata}</b><br>预测: %{y:.2f}<extra></extra>",
                customdata=pred_date_strs,
            )
        )

        fig_pred.add_trace(
            go.Scatter(
                x=pred_idx + pred_idx[::-1],
                y=np.concatenate([pred_high, pred_low[::-1]]),
                fill="toself",
                fillcolor="rgba(139,92,246,0.15)",
                line=dict(color="rgba(139,92,246,0)"),
                name="80% 置信区间",
                hoverinfo="skip",
                showlegend=True,
            )
        )

        sep_x = hist_idx[-1]
        fig_pred.add_shape(
            type="line",
            x0=sep_x,
            x1=sep_x,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(dash="dash", color="gray", width=1),
        )
        fig_pred.add_annotation(
            x=sep_x,
            y=1,
            yref="paper",
            text="今天",
            showarrow=False,
            yanchor="bottom",
            font=dict(size=11, color="gray"),
        )

        fig_pred.update_layout(
            xaxis=dict(
                title="日期",
                tickvals=all_idx[::max(1, len(all_idx) // 12)],
                ticktext=all_dates[::max(1, len(all_idx) // 12)],
                tickangle=-45,
            ),
            yaxis_title="价格",
            height=500,
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_pred, width="stretch")

        st.markdown("---")
        st.subheader("📋 预测数据明细")

        pred_display = pred.copy()
        pred_display["date"] = pred_display["date"].dt.strftime("%Y-%m-%d")
        pred_display = pred_display.rename(
            columns={
                "pred_0.1": "下界 (10%)",
                "pred_0.5": "预测值 (50%)",
                "pred_0.9": "上界 (90%)",
            }
        )
        st.dataframe(pred_display, width="stretch", hide_index=True)

        with st.expander("📥 下载预测数据"):
            csv = pred_display.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="下载 CSV",
                data=csv,
                file_name=f"{st.session_state.symbol}_prediction.csv",
                mime="text/csv",
            )

    st.markdown("---")
    st.subheader("📋 历史数据明细")
    hist_display = hist.copy()
    hist_display["date"] = hist_display["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(hist_display, width="stretch", hide_index=True)

else:
    st.info("👈 请在左侧栏输入股票代码并点击 **开始预测** 按钮。")


# ── Footer ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#888;font-size:0.85rem;">'
    'Powered by <a href="https://huggingface.co/amazon/chronos-2" target="_blank">Amazon Chronos-2</a> | '
    'Built with Streamlit & Plotly | '
    '<a href="https://github.com/AK60000" target="_blank">GitHub: AK60000</a>'
    "</div>",
    unsafe_allow_html=True,
)
