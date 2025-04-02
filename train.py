import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import yfinance as yf

def fetch_stock_data(symbols):
    stock_data = {ticker: yf.download(ticker, period="2y")["Adj Close"] for ticker in symbols}
    return pd.DataFrame(stock_data)

def train_model(stock_data):
    env = StockTradingEnv(stock_data)  # Define a stock trading gym environment
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    model.save("rl_trading_model")
    return model

def load_trained_model():
    return PPO.load("rl_trading_model")

def recommend_portfolio(model, current_allocation):
    action, _ = model.predict(np.array(current_allocation).reshape(1, -1))
    return {stock: round(weight, 2) for stock, weight in zip(current_allocation.keys(), action)}

if __name__ == "__main__":
    stock_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    stock_data = fetch_stock_data(stock_symbols)
    train_model(stock_data)
