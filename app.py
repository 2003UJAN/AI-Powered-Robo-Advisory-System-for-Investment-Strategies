import streamlit as st
import os
from train import load_trained_model, recommend_portfolio
from portfolio_optimizer import optimize_portfolio

# Load trained model
MODEL_PATH = "models/rl_trading_model.zip"  # Update path if needed
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}. Please train the model first.")
else:
    model = load_trained_model()

# Streamlit UI
st.set_page_config(page_title="AI-Powered Robo-Advisory System", layout="wide")

# Center the stock selection UI
st.markdown(
    """
    <style>
        .block-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }
        .stTextInput, .stNumberInput, .stButton {
            width: 50%;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 AI-Powered Robo-Advisory System")

# Stock selection input
st.subheader("Stock Selection")
stocks = st.text_input("Enter stock symbols (comma-separated)", "AAPL,GOOGL,MSFT,AMZN,TSLA")
investment = st.number_input("Investment Amount ($)", min_value=1000, value=10000, step=1000)

if st.button("Optimize Portfolio"):
    stock_list = stocks.split(",")
    recommended_portfolio = recommend_portfolio(model, stock_list, investment)
    optimized_portfolio = optimize_portfolio(recommended_portfolio)

    st.subheader("📊 Optimized Portfolio Allocation")
    st.write(optimized_portfolio)
