# app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from train_model import train_rl_model
from portfolio_optimization import markowitz_portfolio

# Function to fetch stock data from Yahoo Finance
def get_stock_data(tickers, start_date, end_date):
    stock_data = yf.download(tickers, start=start_date, end=end_date)
    return stock_data['Adj Close']

# Main Streamlit App
def main():
    st.title('AI-Powered Robo-Advisory System for Investment Strategies')

    # Input for stock tickers and date range
    tickers = st.text_input('Enter Stock Tickers (comma separated)', 'AAPL,GOOGL,MSFT')
    start_date = st.date_input('Start Date', pd.to_datetime('2015-01-01'))
    end_date = st.date_input('End Date', pd.to_datetime('2021-01-01'))

    tickers = tickers.split(',')

    # Fetch stock data
    stock_data = get_stock_data(tickers, start_date, end_date)

    st.write(f"Displaying data for {', '.join(tickers)}:")
    st.dataframe(stock_data.tail())

    # Markowitz Portfolio Optimization
    st.subheader('Markowitz Portfolio Optimization')
    returns = stock_data.pct_change().dropna()
    optimal_weights = markowitz_portfolio(returns)
    st.write(f"Optimal Portfolio Weights: {dict(zip(tickers, optimal_weights))}")

    # Plot the portfolio
    fig, ax = plt.subplots()
    ax.pie(optimal_weights, labels=tickers, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    # Reinforcement Learning for Portfolio Optimization
    st.subheader('Reinforcement Learning Agent')
    model = train_rl_model(stock_data)
    
    # Use RL agent to get portfolio allocation
    observation = model.env.reset()
    for _ in range(10):
        action, _ = model.predict(observation)
        observation, reward, done, info = model.env.step(action)
        if done:
            break

    st.write(f"Reinforcement Learning Portfolio Allocation: {dict(zip(tickers, action))}")

if __name__ == '__main__':
    main()
