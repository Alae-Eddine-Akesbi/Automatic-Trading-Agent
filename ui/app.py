import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

from environment import TradingEnvironment
from agent import QLearningAgent
from data_utils import feature_engineering

st.set_page_config(page_title="Trading Agent RL", layout="wide")
st.title("📈 Agent de Trading Automatique - Q-Learning")

st.sidebar.header("Paramètres")
ticker = st.sidebar.text_input("Ticker Yahoo Finance :", "AAPL")
start_date = st.sidebar.date_input("Date début", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("Date fin", pd.to_datetime("2025-01-01"))
n_episodes = st.sidebar.slider("Nombre d'épisodes", 10, 500, 100)
fast_test = st.sidebar.checkbox("300 jours)")

if st.sidebar.button("Lancer l'entraînement"):

    tabs = st.tabs(["📥 Données", "🤖 Entraînement", "📊 Test", "📋 Q-table"])

    with tabs[0]:
        st.info("📥 Chargement des données...")
        data = yf.download(ticker, start=start_date, end=end_date)
        if 'Adj Close' not in data.columns:
            data['Adj Close'] = data['Close']
        st.success(f"{len(data)} lignes chargées.")

        if fast_test:
            data = data.tail(300)
            n_episodes = 10
            st.warning("Mode rapide activé : dataset réduit à 300 jours, 10 épisodes.")

        data = feature_engineering(data)

    with tabs[1]:
        st.info("🤖 Entraînement agent Q-Learning...")
        env = TradingEnvironment(data.iloc[:int(len(data)*0.8)].copy())
        agent = QLearningAgent()
        profits = []
        progress_bar = st.progress(0)

        for episode in range(n_episodes):
            state = env.reset()
            while True:
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)
                if next_state is not None:
                    agent.learn(state, action, reward, next_state)
                    state = next_state
                if done:
                    break
            agent.decay_epsilon()
            profits.append(env.total_profit)
            progress_bar.progress((episode + 1) / n_episodes)

        progress_bar.empty()
        st.success("✅ Entraînement terminé")

        fig, ax = plt.subplots()
        ax.plot(profits)
        ax.set_xlabel("Épisode")
        ax.set_ylabel("Profit total")
        st.pyplot(fig)

    with tabs[2]:
        st.subheader("📊 Backtest sur données test")
        test_data = data.iloc[int(len(data)*0.8):].copy()
        test_env = TradingEnvironment(test_data)
        state = test_env.reset()
        while True:
            action = agent.get_qs(state).argmax()
            next_state, reward, done = test_env.step(action)
            if next_state is None:
                break
            state = next_state

        profit_total = float(test_env.total_profit)
        returns = pd.to_numeric(pd.Series(test_env.profits), errors='coerce').dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        cumulative = returns.cumsum()
        max_drawdown = (np.maximum.accumulate(cumulative) - cumulative).max()

        st.info(f"Profit test : {profit_total:.2f} $")
        st.info(f"Sharpe Ratio : {sharpe:.2f}")
        st.info(f"Max Drawdown : {max_drawdown:.2f} $")

        st.subheader("📉 Signaux Buy / Sell")
        prices = test_data['Adj Close'].values
        buy_signals = [step for step, action, price in test_env.history if action == 'BUY']
        sell_signals = [step for step, action, price in test_env.history if action == 'SELL']

        fig2, ax2 = plt.subplots()
        ax2.plot(prices, label=f"Prix {ticker}")
        ax2.scatter(buy_signals, prices[buy_signals], marker='^', color='green', label="Buy", s=60)
        ax2.scatter(sell_signals, prices[sell_signals], marker='v', color='red', label="Sell", s=60)
        ax2.legend()
        st.pyplot(fig2)

    with tabs[3]:
        st.subheader("📋 Aperçu Q-table (Top 10 états)")
        q_table_df = pd.DataFrame([
            {"State": state, "Hold": values[0], "Buy": values[1], "Sell": values[2]}
            for state, values in list(agent.q_table.items())[:10]
        ])
        st.dataframe(q_table_df)
