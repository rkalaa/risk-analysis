import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from typing import Tuple, Dict
from sklearn.metrics import mean_squared_error

class ARIMAModel:
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        self.order = order
        self.model = None
        self.results = None

    def fit(self, data: pd.Series):
        """
        Fit the ARIMA model to the data.
        
        :param data: Time series data
        """
        self.model = ARIMA(data, order=self.order)
        self.results = self.model.fit()

    def predict(self, steps: int) -> pd.Series:
        """
        Make predictions using the fitted ARIMA model.
        
        :param steps: Number of steps to forecast
        :return: Series of forecasted values
        """
        if self.results is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        forecast = self.results.forecast(steps=steps)
        return forecast

    def get_residuals(self) -> pd.Series:
        """
        Get the residuals of the fitted model.
        
        :return: Series of residuals
        """
        if self.results is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        return self.results.resid

    def get_aic(self) -> float:
        """
        Get the Akaike Information Criterion (AIC) of the fitted model.
        
        :return: AIC value
        """
        if self.results is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        return self.results.aic

    def get_bic(self) -> float:
        """
        Get the Bayesian Information Criterion (BIC) of the fitted model.
        
        :return: BIC value
        """
        if self.results is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        return self.results.bic

    def test_stationarity(self, data: pd.Series) -> Dict:
        """
        Perform Augmented Dickey-Fuller test for stationarity.
        
        :param data: Time series data
        :return: Dictionary with test results
        """
        result = adfuller(data)
        return {
            'Test Statistic': result[0],
            'p-value': result[1],
            'Critical Values': result[4]
        }

    def grid_search(self, data: pd.Series, p_range: range, d_range: range, q_range: range) -> Dict:
        """
        Perform grid search to find the best ARIMA parameters.
        
        :param data: Time series data
        :param p_range: Range of p values to test
        :param d_range: Range of d values to test
        :param q_range: Range of q values to test
        :return: Dictionary with best parameters and corresponding AIC
        """
        best_aic = np.inf
        best_params = None
        
        for p in p_range:
            for d in d_range:
                for q in q_range:
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        results = model.fit()
                        aic = results.aic
                        if aic < best_aic:
                            best_aic = aic
                            best_params = (p, d, q)
                    except:
                        continue
        
        return {
            'Best Parameters': best_params,
            'AIC': best_aic
        }

    def calculate_mse(self, test_data: pd.Series) -> float:
        """
        Calculate Mean Squared Error for out-of-sample predictions.
        
        :param test_data: Out-of-sample time series data
        :return: MSE value
        """
        if self.results is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        forecast = self.predict(steps=len(test_data))
        mse = mean_squared_error(test_data, forecast)
        return mse