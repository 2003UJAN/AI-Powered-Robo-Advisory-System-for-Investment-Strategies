import numpy as np
import pandas as pd
from scipy.optimize import minimize
from data_loader import fetch_stock_data

def optimize_portfolio(stock_symbols, investment_amount):
    data = fetch_stock_data(stock_symbols)

    # Compute returns & covariance matrix
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Define objective function for minimizing risk
    def portfolio_volatility(weights):
        return np.sqrt(weights.T @ cov_matrix @ weights)

    # Constraints: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in stock_symbols]

    # Optimize weights
    init_guess = [1 / len(stock_symbols)] * len(stock_symbols)
    result = minimize(portfolio_volatility, init_guess, bounds=bounds, constraints=constraints)

    # Allocate investment
    allocation = (result.x * investment_amount).round(2)
    return dict(zip(stock_symbols, allocation))
