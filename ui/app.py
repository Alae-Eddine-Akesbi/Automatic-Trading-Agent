import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

from data_utils import feature_engineering
from environment import TradingEnvironment
from agent import QLearningAgent

def flatten_columns(df):
    """Flattens MultiIndex columns after Yahoo download."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(c) for c in col if c]) for col in df.columns.values]
    return df

def plotly_trade_animation(
    df: pd.DataFrame,
    ohlc: pd.DataFrame,
    *,
    tail: int | None = 200,
    suffix: str = "",
    frame_duration: int = 40,
) -> go.Figure:
    ohlc = flatten_columns(ohlc)
    colors = {"BUY": "lime", "SELL": "red", "HOLD": "gold"}
    n = len(df)

    # Initial traces for the very first frame
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=ohlc.index[:1],
                open=ohlc["Open" + suffix].iloc[:1],
                high=ohlc["High" + suffix].iloc[:1],
                low=ohlc["Low" + suffix].iloc[:1],
                close=ohlc["Close" + suffix].iloc[:1],
                increasing=dict(line=dict(color="lime", width=3), fillcolor="rgba(0,255,0,0.55)"),
                decreasing=dict(line=dict(color="red", width=3), fillcolor="rgba(255,0,0,0.55)"),
                whiskerwidth=0.4,
                opacity=1.0,
                showlegend=False,
                name="OHLC",
            ),
            go.Scatter(x=[], y=[], mode="lines", line=dict(color="white", width=2), hoverinfo="skip", showlegend=False),
            go.Scatter(x=[], y=[], mode="markers", marker=dict(size=24, color="gold", line=dict(width=2, color="white")), hoverinfo="skip", showlegend=False),
        ]
    )

    # --- Animation frames with dynamic annotation for current date ---
    frames = []
    for i in range(n):
        candles = ohlc.iloc[: i + 1]
        lo = 0 if tail is None else max(0, i - tail)
        seg = df.iloc[lo : i + 1]
        row = df.iloc[i]
        current_date_str = row["Date"].strftime("%Y-%m-%d") if hasattr(row["Date"], "strftime") else str(row["Date"])

        # Annotation for the current date (bottom center of the plot)
        annotation = [
            dict(
                x=0.5,
                y=-0.12,
                xref="paper",
                yref="paper",
                showarrow=False,
                text=f"<b>Date: {current_date_str}</b>",
                font=dict(size=18, color="white"),
                bgcolor="rgba(0,0,0,0.5)",
            )
        ]

        frames.append(
            go.Frame(
                data=[
                    go.Candlestick(
                        x=candles.index,
                        open=candles["Open" + suffix],
                        high=candles["High" + suffix],
                        low=candles["Low" + suffix],
                        close=candles["Close" + suffix],
                        increasing=dict(line=dict(color="lime", width=3), fillcolor="rgba(0,255,0,0.55)"),
                        decreasing=dict(line=dict(color="red", width=3), fillcolor="rgba(255,0,0,0.55)"),
                        whiskerwidth=0.4,
                        opacity=1.0,
                        showlegend=False,
                        name="OHLC",
                    ),
                    go.Scatter(
                        x=seg["Date"],
                        y=seg["Close"],
                        mode="lines",
                        line=dict(color="white", width=2),
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    go.Scatter(
                        x=[row["Date"]],
                        y=[row["Close"]],
                        mode="markers",
                        marker=dict(size=24, color=colors.get(row["Action"], "gold"), line=dict(width=2, color="white")),
                        name=row["Action"],
                        showlegend=False,
                    ),
                ],
                traces=[0, 1, 2],
                name=str(i),
                layout=go.Layout(annotations=annotation),
            )
        )

    fig.frames = frames
    # Add annotation for first frame
    first_date_str = df["Date"].iloc[0].strftime("%Y-%m-%d") if hasattr(df["Date"].iloc[0], "strftime") else str(df["Date"].iloc[0])
    fig.update_layout(
        template="plotly_dark",
        height=500,
        margin=dict(t=50, b=40, l=20, r=20),
        xaxis_title="Date",
        yaxis_title="Prix (USD)",
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": frame_duration, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ],
        annotations=[
            dict(
                x=0.5,
                y=-0.12,
                xref="paper",
                yref="paper",
                showarrow=False,
                text=f"<b>Date: {first_date_str}</b>",
                font=dict(size=18, color="white"),
                bgcolor="rgba(0,0,0,0.5)",
            )
        ],
    )
    return fig


st.set_page_config(page_title="Trading Agent RL", layout="wide")
st.title("📈 Agent de Trading Automatique – Q-Learning (Plotly Edition)")

st.sidebar.header("Paramètres")
ticker      = st.sidebar.text_input("Ticker Yahoo Finance :", "AAPL")
start_date  = st.sidebar.date_input("Date début", pd.to_datetime("2020-01-01"))
end_date    = st.sidebar.date_input("Date fin",   pd.to_datetime("2025-01-01"))
n_episodes  = st.sidebar.slider("Nombre d'épisodes", 10, 500, 100)
fast_test   = st.sidebar.checkbox("Mode rapide (300 jours)")
anim_speed_ms = st.sidebar.slider("Animation speed (ms per frame)", min_value=10, max_value=300, value=40, step=10)

ACTION_LABELS = {0: "HOLD", 1: "BUY", 2: "SELL"}

if st.sidebar.button("Lancer l'entraînement"):
    tabs = st.tabs(["📥 Données",
                    "🤖 Entraînement",
                    "📊 Test",
                    "🎞 Animation",
                    "📋 Q-table"])

    # --- 1. Data ---
    with tabs[0]:
        st.info("📥 Chargement des données…")
        data = yf.download(ticker, start=start_date, end=end_date)
        data = flatten_columns(data)
        if f"Adj Close_{ticker}" not in data.columns:
            data[f"Adj Close_{ticker}"] = data[f"Close_{ticker}"]
        st.success(f"{len(data)} lignes chargées.")
        if fast_test:
            data = data.tail(300)
            n_episodes = 10
            st.warning("Mode rapide : dataset réduit à 300 jours, 10 épisodes.")
        data = data.rename(columns={
            f"Open_{ticker}": "Open",
            f"High_{ticker}": "High",
            f"Low_{ticker}": "Low",
            f"Close_{ticker}": "Close",
            f"Adj Close_{ticker}": "Adj Close",
            f"Volume_{ticker}": "Volume"
        })
        data = feature_engineering(data)
        st.write(data.head())

    # --- 2. Training ---
    with tabs[1]:
        st.info("🤖 Entraînement agent Q-Learning…")
        train_data = data.iloc[: int(len(data) * 0.8)].copy()
        env   = TradingEnvironment(train_data)
        agent = QLearningAgent()
        profits      = []
        progress_bar = st.progress(0.)
        for ep in range(n_episodes):
            state = env.reset()
            while True:
                act_idx = agent.choose_action(state)
                next_state, reward, done = env.step(act_idx)
                if next_state is not None:
                    agent.learn(state, act_idx, reward, next_state)
                    state = next_state
                if done:
                    break
            agent.decay_epsilon()
            profits.append(env.total_profit)
            progress_bar.progress((ep + 1) / n_episodes)
        progress_bar.empty()
        st.success("✅ Entraînement terminé")
        st.plotly_chart(
            px.line(y=profits,
                    labels={"x": "Épisode", "y": "Profit total"},
                    template="plotly_dark"),
            use_container_width=True
        )

    # --- 3. Back-test ---
    with tabs[2]:
        st.subheader("📊 Back-test sur données test")
        test_data = data.iloc[int(len(data) * 0.8):].copy()
        test_env  = TradingEnvironment(test_data)
        state         = test_env.reset()
        actions_taken = []
        while True:
            act_idx = int(agent.get_qs(state).argmax())
            actions_taken.append(ACTION_LABELS[act_idx])
            next_state, reward, done = test_env.step(act_idx)
            if next_state is None:
                break
            state = next_state
        profit_total = float(test_env.total_profit)
        returns      = pd.to_numeric(pd.Series(test_env.profits), errors="coerce").dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() else 0
        cumulative   = returns.cumsum()
        max_drawdown = (np.maximum.accumulate(cumulative) - cumulative).max()
        st.info(f"Profit test : {profit_total:,.2f} $")
        st.info(f"Sharpe Ratio : {sharpe:.2f}")
        st.info(f"Max Drawdown : {max_drawdown:,.2f} $")
        prices  = test_data["Adj Close"].values
        buy_idx = [i for i, a in enumerate(actions_taken) if a == "BUY"]
        sell_idx= [i for i, a in enumerate(actions_taken) if a == "SELL"]
        fig_bt = go.Figure()
        fig_bt.add_trace(
            go.Candlestick(
                x=test_data.index,
                open=test_data["Open"], high=test_data["High"],
                low=test_data["Low"], close=test_data["Close"],
                increasing=dict(line=dict(color="lime", width=3),
                                fillcolor="rgba(0,255,0,0.55)"),
                decreasing=dict(line=dict(color="red", width=3),
                                fillcolor="rgba(255,0,0,0.55)"),
                whiskerwidth=0.4,
                opacity=1.0,
                name="OHLC",
            )
        )
        if buy_idx:
            fig_bt.add_trace(go.Scatter(
                x=test_data.index[buy_idx], y=prices[buy_idx],
                mode="markers",
                marker=dict(symbol="triangle-up", size=10,
                            color="lime", line=dict(width=1, color="white")),
                name="BUY"))
        if sell_idx:
            fig_bt.add_trace(go.Scatter(
                x=test_data.index[sell_idx], y=prices[sell_idx],
                mode="markers",
                marker=dict(symbol="triangle-down", size=10,
                            color="red", line=dict(width=1, color="white")),
                name="SELL"))
        fig_bt.update_layout(
            height=600, template="plotly_dark",
            xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_bt, use_container_width=True)

    # --- 4. Animation ---
    with tabs[3]:
        st.subheader("🎞 Animation des décisions de l’agent")
        if not actions_taken:
            st.warning("Aucune action à animer.")
        else:
            n = len(actions_taken)
            anim_df = pd.DataFrame(
                {
                    "Date":   test_data.index[:n],
                    "Close":  test_data["Adj Close"].iloc[:n].values,
                    "Action": actions_taken,
                }
            ).reset_index(drop=True)
            anim_ohlc = test_data.iloc[:n].copy()

            
            fig_anim = plotly_trade_animation(
                anim_df, ohlc=anim_ohlc, tail=500, suffix="", frame_duration=anim_speed_ms
            )
            html_str = fig_anim.to_html(include_plotlyjs='cdn')
            components.html(html_str, height=600)


    # --- 5. Q-table ---
    with tabs[4]:
        st.subheader("📋 Aperçu Q-table (Top 10 états)")
        q_table_df = pd.DataFrame(
            [
                {"State": s, "Hold": v[0], "Buy": v[1], "Sell": v[2]}
                for s, v in list(agent.q_table.items())[:10]
            ]
        )
        st.dataframe(q_table_df, use_container_width=True)
