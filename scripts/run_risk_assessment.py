import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    VaRModel, CVaRModel, VolatilityModel, BetaModel, SharpeRatioModel,
    RandomForestModel, GradientBoostingModel, ARIMAModel
)
from data import DataLoader, DataPreprocessor, FeatureEngineer

def run_risk_assessment(tickers, start_date, end_date, risk_free_rate=0.02):
    # Load data
    data_loader = DataLoader()
    data = data_loader.load_from_yfinance(tickers + ['^GSPC'], start_date, end_date)
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    returns = preprocessor.calculate_returns(data)
    returns = preprocessor.handle_missing_values(returns)
    
    # Feature engineering
    engineer = FeatureEngineer()
    all_features = {}
    
    for ticker in tickers:
        ticker_data = data[ticker]
        
        print(f"Structure of data for {ticker}:")
        print(ticker_data.head())
        print(type(ticker_data))
        
        # Convert Series to DataFrame
        ticker_df = pd.DataFrame({
            'Close': ticker_data,
            'Open': ticker_data,
            'High': ticker_data,
            'Low': ticker_data,
            'Volume': pd.Series(1, index=ticker_data.index)
        })
        
        try:
            ticker_features = engineer.add_technical_indicators(ticker_df)
            ticker_features = engineer.add_rolling_statistics(ticker_features, [5, 10, 20])
            # Add ticker prefix to column names
            ticker_features.columns = [f"{ticker}_{col}" for col in ticker_features.columns]
            all_features[ticker] = ticker_features
            print(f"Features for {ticker}:")
            print(ticker_features.head())
            print(ticker_features.columns)
        except Exception as e:
            print(f"Error in feature engineering for {ticker}: {str(e)}")
            continue
    
    # Combine all features
    features = pd.concat(all_features.values(), axis=1)
    print("Final features:")
    print(features.head())
    print(features.columns)
    
    if features.empty:
        print("No features were generated. Cannot proceed with model training.")
        return None, None
    
    # Risk metrics calculation
    risk_metrics = {}
    for ticker in tickers:
        vol_model = VolatilityModel()
        vol_model.set_returns(returns[ticker])
        volatility = vol_model.calculate_historical_volatility().iloc[-1]  # Get the last value
        
        risk_metrics[ticker] = {
            'VaR': VaRModel().calculate_var(returns[ticker]),
            'CVaR': CVaRModel().calculate_cvar(returns[ticker]),
            'Volatility': volatility,
            'Beta': BetaModel().calculate_beta(returns[ticker], returns['^GSPC']),
            'Sharpe Ratio': SharpeRatioModel().calculate_sharpe_ratio(returns[ticker], risk_free_rate)
        }
    
    # Machine learning models
    predictions = {}
    for ticker in tickers:
        X = features[[col for col in features.columns if col.startswith(ticker)]]
        y = returns[ticker]
        
        if X.empty:
            print(f"No features for {ticker}. Skipping model training.")
            continue
        
        X_train, X_test = X[:-30], X[-30:]
        y_train, y_test = y[:-30], y[-30:]
        
        rf_model = RandomForestModel()
        gb_model = GradientBoostingModel()
        arima_model = ARIMAModel()
        
        rf_model.train(X_train, y_train)
        gb_model.train(X_train, y_train)
        arima_model.fit(y_train)
        
        predictions[ticker] = {
            'RandomForest': rf_model.predict(X_test),
            'GradientBoosting': gb_model.predict(X_test),
            'ARIMA': arima_model.predict(steps=len(X_test))
        }
    
    return risk_metrics, predictions

def main():
    tickers = ["TSLA"]
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    risk_metrics, predictions = run_risk_assessment(tickers, start_date, end_date)
    
    print("Risk Metrics:")
    print(pd.DataFrame(risk_metrics).T)
    
    print("\nPredictions (last 5 days):")
    if predictions:
        for ticker in tickers:
            if ticker in predictions:
                print(f"\n{ticker}:")
                print(pd.DataFrame({model: preds[-5:] for model, preds in predictions[ticker].items()}))
            else:
                print(f"\n{ticker}: No predictions available")
    else:
        print("No predictions were generated.")

if __name__ == "__main__":
    main()