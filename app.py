import streamlit as st
import os
from portfolio_optimizer import optimize_portfolio
from train import load_trained_model, recommend_portfolio

# Load the trained RL model
model = load_trained_model()

# Streamlit UI Layout
st.set_page_config(page_title="AI-Powered Robo-Advisory System", layout="wide")

# Centered layout using columns
st.markdown("<h1 style='text-align: center;'>AI-Powered Robo-Advisory System</h1>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.subheader("Stock Selection")
    stock_symbols = st.text_input("Enter stock symbols (comma-separated)", "AAPL,GOOGL,MSFT,AMZN,TSLA")
    investment_amount = st.number_input("Investment Amount ($)", min_value=1000, value=10000, step=1000)

    if st.button("Optimize Portfolio"):
        stock_list = [s.strip().upper() for s in stock_symbols.split(",")]
        optimized_portfolio = recommend_portfolio(model, stock_list, investment_amount)
        st.write("### Recommended Portfolio:")
        st.write(optimized_portfolio)
