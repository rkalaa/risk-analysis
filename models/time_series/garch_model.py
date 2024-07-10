import numpy as np
import pandas as pd
from arch import arch_model
from typing import Tuple, Dict
from sklearn.metrics import mean_squared_error

class GARCHModel:
    def __init__(self, p: int = 1, q: int = 1, mean: str = 'Constant', vol: str = 'GARCH', dist: str = 'Normal'):
        self.p = p
        self.q = q
        self.mean = mean
        self.vol = vol
        self.dist = dist
        self.model = None
        self.results = None

    def fit(self, data: pd.Series):
        """
        Fit the GARCH model to the data.
        
        :param data: Time series data of returns
        """
        self.model = arch_model(data, p=self.p, q=self.q, mean=self.mean, vol=self.vol, dist=self.dist)
        self.results = self.model.fit(disp='off')

    def predict(self, steps: int) -> Tuple[pd.Series, pd.Series]:
        """
        Make predictions using the fitted GARCH model.
        
        :param steps: Number of steps to forecast
        :return: Tuple of Series (mean forecast, variance forecast)
        """
        if self.results is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        forecast = self.results.forecast(horizon=steps)
        return forecast.mean, forecast.variance

    def get_residuals(self) -> pd.Series:
        """
        Get the residuals of the fitted model.
        
        :return: Series of residuals
        """
        if self.results is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        return self.results.resid

    def get_conditional_volatility(self) -> pd.Series:
        """
        Get the conditional volatility of the fitted model.
        
        :return: Series of conditional volatility
        """
        if self.results is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        return pd.Series(self.results.conditional_volatility, index=self.results.index)

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

    def grid_search(self, data: pd.Series, p_range: range, q_range: range) -> Dict:
        """
        Perform grid search to find the best GARCH parameters.
        
        :param data: Time series data of returns
        :param p_range: Range of p values to test
        :param q_range: Range of q values to test
        :return: Dictionary with best parameters and corresponding AIC
        """
        best_aic = np.inf
        best_params = None
        
        for p in p_range:
            for q in q_range:
                try:
                    model = arch_model(data, p=p, q=q, mean=self.mean, vol=self.vol, dist=self.dist)
                    results = model.fit(disp='off')
                    aic = results.aic
                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, q)
                except:
                    continue
        
        return {
            'Best Parameters': best_params,
            'AIC': best_aic
        }

    def calculate_mse(self, test_data: pd.Series) -> float:
        """
        Calculate Mean Squared Error for out-of-sample volatility predictions.
        
        :param test_data: Out-of-sample time series data of returns
        :return: MSE value
        """
        if self.results is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        forecast = self.predict(steps=len(test_data))[1]  # Use variance forecast
        realized_variance = test_data ** 2
        mse = mean_squared_error(realized_variance, forecast)
        return mse

    def simulate(self, n_periods: int, n_simulations: int) -> np.ndarray:
        """
        Simulate future paths based on the fitted GARCH model.
        
        :param n_periods: Number of periods to simulate
        :param n_simulations: Number of simulation paths
        :return: Array of simulated paths
        """
        if self.results is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        return self.results.simulate(n_periods, n_simulations)