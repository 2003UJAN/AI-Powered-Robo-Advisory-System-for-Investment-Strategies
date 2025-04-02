import numpy as np
import pandas as pd
from scipy.optimize import minimize

def optimize_portfolio(stock_data):
    returns = stock_data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))

    num_assets = len(mean_returns)
    init_guess = num_assets * [1. / num_assets]
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    result = minimize(portfolio_variance, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return dict(zip(stock_data.columns, result.x))

