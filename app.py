import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import random
import time
from datetime import date
import statsmodels.tsa.stattools as ts

# --- Helper function for chatbot ---
def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

# --- Streamlit Layout ---
st.set_page_config(layout="wide")
st.title("📈 Stock Comparison & Backtesting App")

# --- Sidebar: Chatbot ---
st.sidebar.title("🤖 Chat Assistant")
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat-style UI without background colors
for message in st.session_state.messages:
    is_user = message["role"] == "user"
    align = "right" if is_user else "left"
    emoji = "👤" if is_user else "🤖"
    with st.sidebar:
        st.markdown(f"""
        <div style='padding:10px; margin:5px 0; text-align:{align};'>
            <span style='font-size:20px'>{emoji}</span><br/>
            {message["content"]}
        </div>
        """, unsafe_allow_html=True)

st.sidebar.markdown("---")
chat_prompt = st.sidebar.chat_input("Ask me anything...")
if chat_prompt:
    st.session_state.messages.append({"role": "user", "content": chat_prompt})
    response = "".join(response_generator())
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.experimental_rerun()

# --- Initialize state for pairs trading ---
if "run_pairs_test" not in st.session_state:
    st.session_state.run_pairs_test = False

# --- Main Area: Stock UI and Analysis ---
st.subheader("Compare and Backtest Stocks")

stock1 = st.text_input("Enter first stock ticker (e.g. XOM)", "XOM")
stock2 = st.text_input("Enter second stock ticker (e.g. CVX)", "CVX")
start_date = st.date_input("Start Date", value=date(2023, 1, 1))
end_date = st.date_input("End Date", value=date.today())

if st.button("Run Analysis"):
    st.session_state.run_pairs_test = False
    data1 = yf.download(stock1, start=start_date, end=end_date)
    data2 = yf.download(stock2, start=start_date, end=end_date)
    if data1.empty or data2.empty:
        st.error("One of the stock tickers is invalid or no data available.")
    else:
        st.session_state.data1 = data1
        st.session_state.data2 = data2

if "data1" in st.session_state and "data2" in st.session_state:
    data1 = st.session_state.data1
    data2 = st.session_state.data2

    # Price Chart
    st.subheader(f"📊 {stock1} vs {stock2} - Closing Prices")
    fig1, ax1 = plt.subplots()
    ax1.plot(data1.index, data1['Close'], label=stock1)
    ax1.plot(data2.index, data2['Close'], label=stock2)
    ax1.set_ylabel("Close Price", fontsize=8)
    ax1.set_xlabel("Date", fontsize=8)
    ax1.tick_params(axis='both', labelsize=8)
    ax1.legend(fontsize=8)
    ax1.set_title("Stock Closing Prices", fontsize=10)
    st.pyplot(fig1)

    # Cumulative Returns
    st.subheader("📈 Cumulative Returns")
    cum1 = (data1['Close'] / data1['Close'].iloc[0]) - 1
    cum2 = (data2['Close'] / data2['Close'].iloc[0]) - 1
    fig2, ax2 = plt.subplots()
    ax2.plot(data1.index, cum1, label=f"{stock1} Return")
    ax2.plot(data2.index, cum2, label=f"{stock2} Return")
    ax2.set_ylabel("Cumulative Return", fontsize=8)
    ax2.set_xlabel("Date", fontsize=8)
    ax2.tick_params(axis='both', labelsize=8)
    ax2.legend(fontsize=8)
    ax2.set_title("Cumulative Return Comparison", fontsize=10)
    st.pyplot(fig2)

    # Cointegration Test
    st.subheader("🔗 Cointegration Test")
    joined = pd.concat([data1['Close'], data2['Close']], axis=1).dropna()
    joined.columns = [stock1, stock2]
    coint_score, p_value, _ = ts.coint(joined[stock1], joined[stock2])

    st.write(f"Cointegration test p-value between {stock1} and {stock2}: **{p_value:.4f}**")

    if p_value < 0.05:
        st.success("✅ The two series are cointegrated (reject the null hypothesis).")
    else:
        st.warning("❌ The two series are NOT cointegrated (fail to reject the null hypothesis).")

    # Pairs Trading
    if st.button("Run Pairs Trading Strategy"):
        st.session_state.run_pairs_test = True

    if st.session_state.run_pairs_test:
        combined = yf.download([stock1, stock2], start=start_date, end=end_date)['Close']
        X = sm.add_constant(combined[stock1])
        y = combined[stock2]
        model = sm.OLS(y, X).fit()
        combined['spread'] = y - model.predict(X)

        mean_spread = combined['spread'].mean()
        std_spread = combined['spread'].std()
        upper = mean_spread + std_spread
        lower = mean_spread - std_spread

        fig, ax = plt.subplots()
        ax.plot(combined.index, combined['spread'], label='Spread', color='blue')
        ax.axhline(mean_spread, color='black', linestyle='--', label='Mean')
        ax.axhline(upper, color='red', linestyle='--', label='Upper Threshold')
        ax.axhline(lower, color='green', linestyle='--', label='Lower Threshold')
        ax.legend(fontsize=8)
        ax.set_title(f"Pairs Trading Strategy: Spread between {stock1} and {stock2}", fontsize=10)
        ax.set_xlabel("Date", fontsize=8)
        ax.set_ylabel("Spread", fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
        st.pyplot(fig)

        combined['long'] = combined['spread'] < lower
        combined['short'] = combined['spread'] > upper
        combined['returns_stock1'] = combined[stock1].pct_change()
        combined['returns_stock2'] = combined[stock2].pct_change()

        combined['pnl'] = np.where(combined['long'], combined['returns_stock2'] - combined['returns_stock1'], 0) + \
                          np.where(combined['short'], combined['returns_stock1'] - combined['returns_stock2'], 0)
        combined['cumulative_pnl'] = combined['pnl'].cumsum()

        fig2, ax2 = plt.subplots()
        ax2.plot(combined.index, combined['cumulative_pnl'], label='Cumulative PnL', color='purple')
        ax2.set_xlabel('Date', fontsize=8)
        ax2.set_ylabel('Cumulative PnL', fontsize=8)
        ax2.set_title('Pairs Trading Strategy: Cumulative Profit and Loss', fontsize=10)
        ax2.tick_params(axis='both', labelsize=8)
        ax2.legend(fontsize=8)
        st.pyplot(fig2)

        final_pnl = combined['cumulative_pnl'].iloc[-1]
        initial_capital = 10000
        pnl_amount = final_pnl * initial_capital
        pct_return = pnl_amount / initial_capital * 100

        st.write(f"💰 Strategy Result:")
        st.write(f"- Final PnL (Spread units): {final_pnl:.4f}")
        st.write(f"- Simulated Profit (on $10,000): ${pnl_amount:.2f}")
        st.write(f"- Return: {pct_return:.2f}%")
