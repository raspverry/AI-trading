import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set date range
end_date = datetime.now()
start_date = end_date - timedelta(days=365*5)  # 5 years of data

# Download stock price data
symbol = "AAPL"  # Replace with your stock symbol
stock_data = yf.download(symbol, start=start_date, end=end_date)
stock_data.to_csv('stock_price_data.csv')

# Simulate news data
dates = pd.date_range(start=start_date, end=end_date)
headlines = [f"News about {symbol} on {date.strftime('%Y-%m-%d')}" for date in dates]
sentiment_scores = np.random.uniform(-1, 1, len(dates))  # Random sentiment scores
news_data = pd.DataFrame({'Date': dates, 'Headline': headlines, 'Sentiment': sentiment_scores})
news_data.to_csv('stock_news_data.csv', index=False)

# Simulate macroeconomic data
np.random.seed(42)  # For reproducibility
macro_data = pd.DataFrame({
    'Date': dates,
    'GDP_Growth': np.random.normal(2, 0.5, len(dates)),
    'Inflation_Rate': np.random.normal(2, 0.3, len(dates)),
    'Unemployment_Rate': np.random.normal(5, 0.5, len(dates))
})
macro_data.set_index('Date', inplace=True)
macro_data.to_csv('macro_economic_data.csv')

# Simulate order book data
order_book_data = pd.DataFrame({
    'Date': stock_data.index,
    'Best_Bid': stock_data['Close'] - np.random.uniform(0, 0.1, len(stock_data)),
    'Best_Ask': stock_data['Close'] + np.random.uniform(0, 0.1, len(stock_data)),
    'Bid_Volume': np.random.randint(1000, 10000, len(stock_data)),
    'Ask_Volume': np.random.randint(1000, 10000, len(stock_data))
})
order_book_data.set_index('Date', inplace=True)
order_book_data.to_csv('order_book_data.csv')

print("Data files created successfully.")
