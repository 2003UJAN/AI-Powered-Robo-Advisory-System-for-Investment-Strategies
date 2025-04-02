import yfinance as yf
import pandas as pd

def get_stock_data(tickers, period="1y"):
    stock_data = {ticker: yf.download(ticker, period=period)["Adj Close"] for ticker in tickers}
    return pd.DataFrame(stock_data)
