# train_model.py

import gym
from stable_baselines3 import A2C
import pandas as pd

# Reinforcement Learning Environment for Portfolio Allocation
class PortfolioEnv(gym.Env):
    def __init__(self, data):
        super(PortfolioEnv, self).__init__()
        self.data = data
        self.n_assets = data.shape[1]
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.n_assets,), dtype=np.float32)
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self.data.iloc[self.current_step].values

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True
            reward = 0
        else:
            done = False
            reward = self.calculate_reward(action)
        obs = self.data.iloc[self.current_step].values
        return obs, reward, done, {}

    def calculate_reward(self, action):
        portfolio_return = np.dot(self.data.pct_change().iloc[self.current_step], action)
        return portfolio_return

def train_rl_model(stock_data):
    # Initialize RL environment
    env = PortfolioEnv(stock_data)
    
    # Train a reinforcement learning agent (A2C)
    model = A2C('MlpPolicy', env, verbose=0)
    model.learn(total_timesteps=5000)
    
    return model
