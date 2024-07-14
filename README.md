# Advanced AI Trading System

## Currently WIP

## Overview

This repository contains an advanced AI-based trading system that integrates various machine learning models, financial technical indicators, and sentiment analysis to simulate trading strategies. The system preprocesses financial data, trains models, makes predictions, and executes trades to simulate trading performance over historical data.

## Key Features

1. **Data Preprocessing:**
    - **Technical Indicators:** Computes SMA, RSI, MACD, Bollinger Bands, OBV, ATR, Stochastic Oscillator, ADX, and Williams %R.
    - **Sentiment Analysis:** Analyzes news headlines using NLTK's SentimentIntensityAnalyzer.
    - **Order Book Data:** Calculates Bid-Ask Spread and Order Imbalance.
    - **Combining Data:** Merges price data, sentiment scores, macroeconomic data, and order book data into a single dataset.

2. **Model Training:**
    - Uses RandomForestRegressor, GradientBoostingRegressor, and MLPRegressor for prediction.
    - Performs grid search to find the best parameters for each model.
    - Trains an ARIMA model for time series forecasting.

3. **Prediction and Trading:**
    - Predicts the next day's return using the trained models.
    - Calculates the Expected Shortfall for risk management.
    - Dynamically sizes positions based on prediction confidence and market volatility.
    - Executes buy/sell trades based on predictions and risk assessments.

4. **Performance Evaluation:**
    - Simulates trading over historical data and calculates portfolio values.
    - Computes performance metrics like total return, annualized volatility, Sharpe Ratio, Sortino Ratio, and Maximum Drawdown.
    - Plots cumulative returns and Monte Carlo simulation results.
    - Performs walk-forward optimization to validate model performance.

5. **SHAP Explanation:**
    - Uses SHAP to explain the predictions of the RandomForestRegressor model.

## Execution Flow

1. **Initialization:**
    - Sets up initial capital and initializes models and sentiment analyzer.

2. **Data Preprocessing:**
    - Processes input CSV files for stock prices, news data, macroeconomic data, and order book data.
    - Computes technical indicators and combines all data into a single dataframe.

3. **Model Training:**
    - Trains machine learning models using historical data.
    - Logs the best parameters and model performance metrics.

4. **Simulation:**
    - Runs a trading simulation using the trained models.
    - Executes trades based on predicted returns and updates portfolio value.
    - Logs each step of the simulation.

5. **Performance Metrics:**
    - Calculates and logs key performance metrics.
    - Plots cumulative returns and performs Monte Carlo simulation.

6. **Walk-Forward Optimization:**
    - Performs walk-forward optimization to validate model performance over rolling windows.

7. **Prediction Explanation:**
    - Generates SHAP plots to explain model predictions.

## Usage

To run the backtest, execute the script with the required CSV files:
```bash
python trading_system.py
```

dataset.py will generate csv file. (AAPL is set as default for testing)

Ensure the following CSV files are in the same directory:
- `stock_price_data.csv`
- `stock_news_data.csv`
- `macro_economic_data.csv`
- `order_book_data.csv`

### Example CSV File Formats

An example CSV file format for `stock_price_data.csv`:
```csv
Date,Open,High,Low,Close,Volume
2024-01-01,100,105,99,104,1000000
2024-01-02,104,106,102,105,1100000
...
```

For `stock_news_data.csv`:
```csv
Date,Headline
2024-01-01,Stock market opens higher
2024-01-02,Tech stocks surge amid new product launches
...
```

### Logging

The script logs detailed information about the preprocessing, training, prediction, and trading processes. Check the logs for detailed insights into the model's performance and trading decisions.

## Requirements

- Python 3.x
- Required libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `nltk`, `statsmodels`, `shap`, `scipy`

Install the required libraries using:
```bash
pip install -r requirements.txt
```

## Performance Report

After running the backtest, the script generates a performance report that includes key metrics such as:

- **Initial Capital:** $100,000
- **Final Portfolio Value:** $XXX.XX
- **Total Return:** XX.XX%
- **Annualized Volatility:** XX.XX%
- **Sharpe Ratio:** X.XX
- **Sortino Ratio:** X.XX
- **Maximum Drawdown:** XX.XX%

Additionally, the script performs walk-forward optimization and Monte Carlo simulations to further validate the model's performance.

## Explanation of Predictions

The system uses SHAP to explain the predictions made by the RandomForestRegressor model, providing insights into the features that drive the model's predictions.

## Backtest Results

The script saves the following plots:
- `cumulative_returns.png`: Plot of cumulative returns over the backtest period.
- `monte_carlo_simulation.png`: Results of the Monte Carlo simulation of returns.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

This README provides a comprehensive guide to understanding, using, and contributing to the Advanced AI Trading System. If you have any questions or need further assistance, please open an issue on the GitHub repository.