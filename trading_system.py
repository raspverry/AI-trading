import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import logging
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm
import shap
import traceback

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedAITradingSystem:
    def __init__(self, initial_capital=100000):
        self.capital = initial_capital
        self.positions = {}
        self.models = {
            'rf': RandomForestRegressor(random_state=42),
            'gb': GradientBoostingRegressor(random_state=42),
            'nn': MLPRegressor(random_state=42),
            'arima': None
        }
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)

    def preprocess_data(self, price_data, news_data, macro_data, order_book_data):
        try:
            # Process price data
            price_data['SMA_10'] = price_data['Close'].rolling(window=10).mean()
            price_data['SMA_30'] = price_data['Close'].rolling(window=30).mean()
            price_data['RSI'] = self.calculate_rsi(price_data['Close'], window=14)
            price_data['MACD'] = self.calculate_macd(price_data['Close'])
            price_data['Bollinger_Upper'], price_data['Bollinger_Lower'] = self.calculate_bollinger_bands(price_data['Close'])
            price_data['OBV'] = self.calculate_obv(price_data)
            price_data['ATR'] = self.calculate_atr(price_data)
            price_data['Stochastic_Oscillator'] = self.calculate_stochastic_oscillator(price_data)
            price_data['ADX'] = self.calculate_adx(price_data)
            price_data['Williams_R'] = self.calculate_williams_r(price_data)
            
            # Add time-based features
            price_data['Day_of_Week'] = price_data.index.dayofweek
            price_data['Month'] = price_data.index.month
            
            # Process news data
            news_data['Sentiment'] = news_data['Headline'].apply(self.get_sentiment)
            daily_sentiment = news_data.groupby('Date')['Sentiment'].mean().reindex(price_data.index, fill_value=0)
            
            # Add order book features
            price_data['Bid-Ask Spread'] = order_book_data['Best_Ask'] - order_book_data['Best_Bid']
            price_data['Order Imbalance'] = (order_book_data['Ask_Volume'] - order_book_data['Bid_Volume']) / (order_book_data['Ask_Volume'] + order_book_data['Bid_Volume'])
            
            # Combine price, sentiment, macro, and order book data
            combined_data = price_data.join(daily_sentiment, how='left').join(macro_data, how='left').fillna(0)
            combined_data['Sentiment_RSI_Interaction'] = combined_data['Sentiment'] * combined_data['RSI']
            
            # Calculate returns for target variable
            combined_data['Target'] = combined_data['Close'].pct_change().shift(-1)
            
            return combined_data.dropna()
        except Exception as e:
            logging.error(f"Error in preprocessing data: {e}")
            raise e

    def calculate_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices, slow=26, fast=12):
        ema_slow = prices.ewm(span=slow, min_periods=slow).mean()
        ema_fast = prices.ewm(span=fast, min_periods=fast).mean()
        return ema_fast - ema_slow

    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band

    def calculate_obv(self, data):
        obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
        return obv

    def calculate_atr(self, data, window=14):
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(window=window).mean()

    def calculate_stochastic_oscillator(self, data, window=14):
        low_min = data['Low'].rolling(window=window).min()
        high_max = data['High'].rolling(window=window).max()
        return 100 * (data['Close'] - low_min) / (high_max - low_min)

    def calculate_adx(self, data, window=14):
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = np.max([high_low, high_close, low_close], axis=0)
        pdm = np.where((data['High'] - data['High'].shift()) > (data['Low'].shift() - data['Low']), data['High'] - data['High'].shift(), 0)
        ndm = np.where((data['Low'].shift() - data['Low']) > (data['High'] - data['High'].shift()), data['Low'].shift() - data['Low'], 0)
        pdm_smooth = pd.Series(pdm).rolling(window=window).mean()
        ndm_smooth = pd.Series(ndm).rolling(window=window).mean()
        tr_smooth = pd.Series(true_range).rolling(window=window).mean()
        pdi = 100 * (pdm_smooth / tr_smooth)
        ndi = 100 * (ndm_smooth / tr_smooth)
        dx = 100 * (np.abs(pdi - ndi) / (pdi + ndi))
        adx = pd.Series(dx).rolling(window=window).mean()
        return adx

    def calculate_williams_r(self, data, window=14):
        high_max = data['High'].rolling(window=window).max()
        low_min = data['Low'].rolling(window=window).min()
        return -100 * (high_max - data['Close']) / (high_max - low_min)

    def get_sentiment(self, headline):
        return self.sentiment_analyzer.polarity_scores(headline)['compound']

    def train_models(self, data):
        try:
            features = [col for col in data.columns if col != 'Target']
            X = data[features]
            y = data['Target']
            
            X_scaled = self.scaler.fit_transform(X)
            X_pca = self.pca.fit_transform(X_scaled)
            
            X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
            
            for name, model in self.models.items():
                if name != 'arima':
                    param_grid = {
                        'rf': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]},
                        'gb': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
                        'nn': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['relu', 'tanh']}
                    }
                    if name in param_grid:
                        grid_search = GridSearchCV(model, param_grid[name], cv=5, n_jobs=-1)
                        grid_search.fit(X_train, y_train)
                        
                        self.models[name] = grid_search.best_estimator_
                        logging.info(f"Best parameters for {name}: {grid_search.best_params_}")
                        
                        predictions = self.models[name].predict(X_test)
                        mse = mean_squared_error(y_test, predictions)
                        mae = mean_absolute_error(y_test, predictions)
                        logging.info(f"{name.upper()} Model - MSE: {mse}, MAE: {mae}")
                else:
                    # Train ARIMA model
                    self.models['arima'] = ARIMA(data['Close'], order=(5,1,0))
                    self.models['arima'] = self.models['arima'].fit()
                    # Make a test forecast to ensure it works
                    test_forecast = self.models['arima'].forecast(steps=1)
                    logging.info(f"ARIMA test forecast: {test_forecast}")
                    
        except Exception as e:
            logging.error(f"Error in training models: {e}")
            raise e

    def predict_return(self, current_data):
        if isinstance(current_data, pd.Series):
            current_data = current_data.values.reshape(1, -1)
        elif isinstance(current_data, np.ndarray):
            current_data = current_data.reshape(1, -1)
        
        logging.info(f"Current data shape: {current_data.shape}")
        current_data_scaled = self.scaler.transform(current_data)
        current_data_pca = self.pca.transform(current_data_scaled)
        predictions = []
        for name, model in self.models.items():
            if name != 'arima':
                pred = model.predict(current_data_pca)[0]
            else:
                forecast = model.forecast(steps=1)
                pred = forecast.iloc[0] if isinstance(forecast, pd.Series) else forecast[0]
            predictions.append(pred)
            logging.info(f"{name} prediction: {pred}")
        
        # Average predictions from ML models only
        ml_predictions = [pred for name, pred in zip(self.models.keys(), predictions) if name != 'arima']
        return np.mean(ml_predictions)

    def calculate_expected_shortfall(self, returns, confidence_level=0.95):
        if len(returns) < 2:
            return abs(returns[0])  # Return absolute value if only one return
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return -np.mean([r for r in returns if r <= var])

    def execute_trade(self, symbol, price, prediction, volatility, current_data):
        try:
            expected_shortfall = self.calculate_expected_shortfall([prediction])
            logging.info(f"Expected Shortfall: {expected_shortfall}")
            
            # Dynamic position sizing based on prediction confidence and volatility
            confidence = 1 / volatility if volatility != 0 else 1
            position_size = min(0.02 * confidence, 0.1)  # Max 10% of capital, scaled by confidence
            logging.info(f"Position Size: {position_size}")
            
            if prediction > expected_shortfall:
                shares = int(self.capital * position_size / price) if price != 0 else 0
                cost = shares * price
                if cost <= self.capital and shares > 0:
                    self.positions[symbol] = self.positions.get(symbol, 0) + shares
                    self.capital -= cost
                    logging.info(f"Bought {shares} shares of {symbol} at {price}")
                else:
                    logging.info(f"Not enough capital to buy {shares} shares of {symbol} at {price}")
            elif prediction < -expected_shortfall:
                shares = self.positions.get(symbol, 0)
                if shares > 0:
                    self.positions[symbol] = 0
                    self.capital += shares * price
                    logging.info(f"Sold {shares} shares of {symbol} at {price}")
                else:
                    logging.info(f"No shares of {symbol} to sell")
            else:
                logging.info("No trade executed")

        except Exception as e:
            logging.error(f"Error in executing trade: {str(e)}")
            logging.error(f"Symbol: {symbol}, Price: {price}, Prediction: {prediction}, Volatility: {volatility}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            raise e

    def run_simulation(self, data):
        portfolio_values = []
        for i in range(len(data)):
            try:
                current_data = data.iloc[i].drop('Target')
                prediction = self.predict_return(current_data)
                
                if data['ATR'].iloc[i] == 0:
                    logging.warning(f"ATR is zero at step {i}. Using default volatility of 1.")
                    volatility = 1
                else:
                    volatility = data['ATR'].iloc[i] / data['Close'].iloc[i]
                
                logging.info(f"Step {i}: Prediction = {prediction}, Volatility = {volatility}")
                logging.info(f"ATR: {data['ATR'].iloc[i]}, Close: {data['Close'].iloc[i]}")
                
                self.execute_trade('AAPL', data['Close'].iloc[i], prediction, volatility, current_data)
                
                # Calculate current portfolio value
                portfolio_value = self.capital + self.positions.get('AAPL', 0) * data['Close'].iloc[i]
                portfolio_values.append(portfolio_value)
                logging.info(f"Step {i}: Portfolio Value = {portfolio_value}")
            except Exception as e:
                logging.error(f"Error in simulation step {i}: {str(e)}")
                logging.error(f"Current data: {current_data}")
                logging.error(f"Traceback: {traceback.format_exc()}")
                break
        
        if not portfolio_values:
            logging.error("No portfolio values generated. Simulation failed.")
            return pd.Series()
        
        return pd.Series(portfolio_values, index=data.index[:len(portfolio_values)])

    def explain_prediction(self, data):
        explainer = shap.TreeExplainer(self.models['rf'])
        shap_values = explainer.shap_values(data.drop('Target', axis=1))
        shap.summary_plot(shap_values, data.drop('Target', axis=1), plot_type="bar")

def calculate_max_drawdown(returns):
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns/peak) - 1
    return drawdown.min()

def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate/252
    downside_returns = excess_returns[excess_returns < 0]
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    sortino_ratio = np.mean(excess_returns) / downside_deviation
    return sortino_ratio * np.sqrt(252)

def run_backtest(price_csv_path, news_csv_path, macro_csv_path, order_book_csv_path):
    try:
        price_data = pd.read_csv(price_csv_path, index_col='Date', parse_dates=True)
        news_data = pd.read_csv(news_csv_path, parse_dates=['Date'])
        macro_data = pd.read_csv(macro_csv_path, index_col='Date', parse_dates=True)
        order_book_data = pd.read_csv(order_book_csv_path, index_col='Date', parse_dates=True)
    except FileNotFoundError as e:
        logging.error(f"Error: {e}. Please make sure all required CSV files are in the correct location.")
        return
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    logging.info(f"Price data columns: {price_data.columns}")
    logging.info(f"News data columns: {news_data.columns}")
    logging.info(f"Macro data columns: {macro_data.columns}")
    logging.info(f"Order book data columns: {order_book_data.columns}")

    trading_system = AdvancedAITradingSystem()
    processed_data = trading_system.preprocess_data(price_data, news_data, macro_data, order_book_data)
    
    logging.info(f"Processed data columns: {processed_data.columns}")
    
    trading_system.train_models(processed_data)
    
    # Run simulation
    portfolio_values = trading_system.run_simulation(processed_data)
    
    if len(portfolio_values) == 0:
        logging.error("Simulation failed to generate portfolio values. Exiting.")
        return

    # Calculate performance metrics
    returns = portfolio_values.pct_change().dropna()
    cumulative_returns = (1 + returns).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1
    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
    sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    max_drawdown = calculate_max_drawdown(returns)
    sortino_ratio = calculate_sortino_ratio(returns)

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.plot(cumulative_returns)
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.grid(True)
    plt.savefig('cumulative_returns.png')
    plt.close()

    # Generate performance report
    performance_report = f"""
    Performance Report:
    -------------------
    Initial Capital: $100,000
    Final Portfolio Value: ${portfolio_values.iloc[-1]:.2f}
    Total Return: {total_return:.2%}
    Annualized Volatility: {volatility:.2%}
    Sharpe Ratio: {sharpe_ratio:.2f}
    Sortino Ratio: {sortino_ratio:.2f}
    Maximum Drawdown: {max_drawdown:.2%}
    """

    logging.info(performance_report)

    # Perform walk-forward optimization
    def walk_forward_optimization(data, window_size=252, step_size=20):
        returns = []
        for start in range(0, len(data) - window_size, step_size):
            end = start + window_size
            train_data = data.iloc[start:end]
            test_data = data.iloc[end:end+step_size]
            
            trading_system.train_models(train_data)
            portfolio_values = trading_system.run_simulation(test_data)
            returns.append(portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1)
        
        return pd.Series(returns)

    walk_forward_returns = walk_forward_optimization(processed_data)
    avg_walk_forward_return = walk_forward_returns.mean()
    logging.info(f"Average Walk-Forward Return: {avg_walk_forward_return:.2%}")

    # Monte Carlo simulation
    def monte_carlo_simulation(returns, num_simulations=1000, time_horizon=252):
        simulated_returns = np.random.choice(returns, size=(num_simulations, time_horizon), replace=True)
        simulated_cum_returns = np.cumprod(1 + simulated_returns, axis=1)
        return simulated_cum_returns

    monte_carlo_results = monte_carlo_simulation(returns)
    confidence_interval = np.percentile(monte_carlo_results[:, -1], [5, 95])

    logging.info(f"95% Confidence Interval for 1-year return: [{confidence_interval[0]:.2%}, {confidence_interval[1]:.2%}]")

    # Plot Monte Carlo simulation results
    plt.figure(figsize=(12, 8))
    plt.plot(monte_carlo_results.T, color='blue', alpha=0.1)
    plt.title('Monte Carlo Simulation of Returns')
    plt.xlabel('Trading Days')
    plt.ylabel('Cumulative Returns')
    plt.grid(True)
    plt.savefig('monte_carlo_simulation.png')
    plt.close()

    # Explain predictions
    trading_system.explain_prediction(processed_data)

    logging.info("Backtesting and analysis completed. Results saved to files.")

if __name__ == "__main__":
    run_backtest('stock_price_data.csv', 'stock_news_data.csv', 'macro_economic_data.csv', 'order_book_data.csv')
