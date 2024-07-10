import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os
import matplotlib.pyplot as plt

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import VaRModel, GARCHModel, ARIMAModel
from data import DataLoader, DataPreprocessor

def backtest_var_model(returns, confidence_level=0.95, window=252):
    var_model = VaRModel(confidence_level)
    var_violations = 0
    var_values = []

    for i in range(window, len(returns)):
        historical_returns = returns[i-window:i]
        var = var_model.calculate_var_historical(historical_returns)
        var_values.append(var)
        if returns.iloc[i] < -var:
            var_violations += 1

    violation_rate = var_violations / (len(returns) - window)
    expected_rate = 1 - confidence_level

    return var_values, violation_rate, expected_rate

def backtest_garch_model(returns, window=252):
    garch_model = GARCHModel()
    forecasted_volatility = []
    actual_volatility = []

    for i in range(window, len(returns)):
        historical_returns = returns[i-window:i]
        garch_model.fit(historical_returns)
        forecast = garch_model.forecast(1)
        forecasted_volatility.append(forecast[0])
        actual_volatility.append(returns.iloc[i]**2)

    mse = np.mean((np.array(forecasted_volatility) - np.array(actual_volatility))**2)
    return forecasted_volatility, actual_volatility, mse

def backtest_arima_model(returns, window=252):
    arima_model = ARIMAModel()
    forecasted_returns = []
    actual_returns = returns[window:]

    for i in range(window, len(returns)):
        historical_returns = returns[i-window:i]
        arima_model.fit(historical_returns)
        forecast = arima_model.predict(steps=1)
        forecasted_returns.append(forecast[0])

    mse = np.mean((np.array(forecasted_returns) - np.array(actual_returns))**2)
    return forecasted_returns, actual_returns, mse

def plot_backtest_results(actual, predicted, title):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.title(title)
    plt.legend()
    plt.show()

def main():
    # Load data
    ticker = 'AAPL'
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    data_loader = DataLoader()
    data = data_loader.load_from_yfinance([ticker], start_date, end_date)
    
    # Check the structure of the loaded data
    print("Data structure:")
    print(type(data))
    print(data.head())
    
    # Extract the price series for the ticker
    if isinstance(data, pd.DataFrame):
        price_series = data[ticker]
    else:
        raise ValueError(f"Unexpected data type: {type(data)}")
    
    preprocessor = DataPreprocessor()
    returns = preprocessor.calculate_returns(price_series)
    returns = returns.dropna()

    # Backtest VaR model
    var_values, violation_rate, expected_rate = backtest_var_model(returns)
    print(f"VaR Model - Violation Rate: {violation_rate:.4f}, Expected Rate: {expected_rate:.4f}")

    # Backtest GARCH model
    forecasted_volatility, actual_volatility, garch_mse = backtest_garch_model(returns)
    print(f"GARCH Model - MSE: {garch_mse:.6f}")
    plot_backtest_results(actual_volatility[-252:], forecasted_volatility[-252:], "GARCH Volatility Forecast (Last Year)")

    # Backtest ARIMA model
    forecasted_returns, actual_returns, arima_mse = backtest_arima_model(returns)
    print(f"ARIMA Model - MSE: {arima_mse:.6f}")
    plot_backtest_results(actual_returns[-252:], forecasted_returns[-252:], "ARIMA Returns Forecast (Last Year)")

if __name__ == "__main__":
    main()
    