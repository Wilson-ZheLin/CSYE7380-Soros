import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
from datetime import date

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

for message in st.session_state.messages:
    with st.sidebar.chat_message(message["role"]):
        st.markdown(message["content"])

st.sidebar.markdown("---")
chat_prompt = st.sidebar.chat_input("Ask me anything...")
if chat_prompt:
    st.session_state.messages.append({"role": "user", "content": chat_prompt})
    with st.sidebar.chat_message("user"):
        st.markdown(chat_prompt)

    with st.sidebar.chat_message("assistant"):
        response = st.write_stream(response_generator())
    st.session_state.messages.append({"role": "assistant", "content": response})

# --- Main Area: Stock UI and Analysis ---
st.subheader("Compare and Backtest Stocks")

stock1 = st.text_input("Enter first stock ticker (e.g. AAPL)", "AAPL")
stock2 = st.text_input("Enter second stock ticker (e.g. MSFT)", "MSFT")
start_date = st.date_input("Start Date", value=date(2023, 1, 1))
end_date = st.date_input("End Date", value=date.today())

if st.button("Run Analysis"):
    data1 = yf.download(stock1, start=start_date, end=end_date)
    data2 = yf.download(stock2, start=start_date, end=end_date)

    if data1.empty or data2.empty:
        st.error("One of the stock tickers is invalid or no data available.")
    else:
        # Closing Prices
        st.subheader(f"📊 {stock1} vs {stock2} - Closing Prices")
        fig1, ax1 = plt.subplots()
        ax1.plot(data1.index, data1['Close'], label=stock1)
        ax1.plot(data2.index, data2['Close'], label=stock2)
        ax1.set_ylabel("Close Price")
        ax1.set_xlabel("Date")
        ax1.legend()
        ax1.set_title("Stock Closing Prices")
        st.pyplot(fig1)

        # Cumulative Returns
        st.subheader("📈 Cumulative Returns")
        cum1 = (data1['Close'] / data1['Close'].iloc[0]) - 1
        cum2 = (data2['Close'] / data2['Close'].iloc[0]) - 1

        fig2, ax2 = plt.subplots()
        ax2.plot(data1.index, cum1, label=f"{stock1} Return")
        ax2.plot(data2.index, cum2, label=f"{stock2} Return")
        ax2.set_ylabel("Cumulative Return")
        ax2.set_xlabel("Date")
        ax2.legend()
        ax2.set_title("Cumulative Return Comparison")
        st.pyplot(fig2)

        # Correlation Analysis
        st.subheader("🔁 Correlation Analysis")
        joined = pd.concat([data1['Close'], data2['Close']], axis=1)
        joined.columns = [stock1, stock2]
        joined = joined.dropna()
        correlation = joined.corr().iloc[0, 1]
        st.write(f"Correlation between {stock1} and {stock2}: **{correlation:.4f}**")

        # Simple Strategy Backtest (e.g. Moving Average Crossover)
        st.subheader("🧪 Simple Backtest: MA Crossover on First Stock")
        data1['SMA50'] = data1['Close'].rolling(50).mean()
        data1['SMA200'] = data1['Close'].rolling(200).mean()

        fig3, ax3 = plt.subplots()
        ax3.plot(data1.index, data1['Close'], label='Close')
        ax3.plot(data1.index, data1['SMA50'], label='SMA50')
        ax3.plot(data1.index, data1['SMA200'], label='SMA200')
        ax3.set_title(f"{stock1} - MA Crossover Strategy")
        ax3.legend()
        st.pyplot(fig3)

        st.write(f"**{stock1} last close**: ${data1['Close'].iloc[-1].item():.2f}")
        st.write(f"**{stock2} last close**: ${data2['Close'].iloc[-1].item():.2f}")
