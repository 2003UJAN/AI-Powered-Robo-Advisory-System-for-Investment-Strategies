import yfinance as yf
import pandas as pd

def fetch_stock_data(stock_symbols, start_date="2020-01-01", end_date="2024-01-01"):
    stock_data = {}
    
    for symbol in stock_symbols:
        try:
            stock = yf.download(symbol, start=start_date, end=end_date)
            stock_data[symbol] = stock["Adj Close"]
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")

    return pd.DataFrame(stock_data)
