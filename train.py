import os
import gym
import numpy as np
from stable_baselines3 import PPO

# Define or load environment (use OpenAI Gym or custom stock trading env)
class StockTradingEnv(gym.Env):
    def __init__(self, stock_symbols):
        self.stock_symbols = stock_symbols
        self.state = np.random.rand(len(stock_symbols))  # Dummy state

    def reset(self):
        self.state = np.random.rand(len(self.stock_symbols))
        return self.state

    def step(self, action):
        reward = np.random.rand()  # Placeholder
        done = False
        return self.state, reward, done, {}

def train_rl_model(stock_symbols):
    env = StockTradingEnv(stock_symbols)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("models/rl_trading_model")

def load_trained_model():
    model_path = os.path.abspath("models/rl_trading_model.zip")
    return PPO.load(model_path)

def recommend_portfolio(model, stock_symbols, investment_amount):
    action = model.predict(np.random.rand(len(stock_symbols)))[0]
    weights = np.exp(action) / np.sum(np.exp(action))  # Normalize to sum to 1
    allocations = (weights * investment_amount).round(2)
    return dict(zip(stock_symbols, allocations))
