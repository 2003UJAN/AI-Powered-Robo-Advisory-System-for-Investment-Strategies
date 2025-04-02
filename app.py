import streamlit as st
import pandas as pd
import yfinance as yf
from portfolio_optimizer import optimize_portfolio
from train import load_trained_model, recommend_portfolio

st.title("AI-Powered Robo-Advisory System")

st.sidebar.header("Stock Selection")
stocks = st.sidebar.text_input("Enter stock symbols (comma-separated)", "AAPL,GOOGL,MSFT,AMZN,TSLA")
investment_amount = st.sidebar.number_input("Investment Amount ($)", min_value=1000, value=10000, step=500)

if st.sidebar.button("Optimize Portfolio"):
    stock_list = [s.strip().upper() for s in stocks.split(",")]
    stock_data = {ticker: yf.download(ticker, period="1y")["Adj Close"] for ticker in stock_list}
    
    portfolio_weights = optimize_portfolio(pd.DataFrame(stock_data))
    model = load_trained_model()
    ai_recommendation = recommend_portfolio(model, portfolio_weights)
    
    st.subheader("Optimized Portfolio Allocation")
    st.write(portfolio_weights)
    
    st.subheader("AI-Recommended Adjustments")
    st.write(ai_recommendation)
