# portfolio_optimization.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Markowitz Portfolio Optimization
def markowitz_portfolio(returns):
    cov_matrix = returns.cov()
    mean_returns = returns.mean()

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def objective_function(weights):
        return -np.dot(weights, mean_returns) / portfolio_volatility(weights)

    n_assets = len(mean_returns)
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for asset in range(n_assets))
    initial_weights = n_assets * [1. / n_assets]

    opt_result = minimize(objective_function, initial_weights, bounds=bounds, constraints=constraints)
    return opt_result.x
