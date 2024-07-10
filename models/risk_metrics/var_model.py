import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, Tuple

class VaRModel:
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level

    def calculate_var_historical(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate Value at Risk using the historical method.
        
        :param returns: Series or array of historical returns
        :return: VaR value
        """
        return np.percentile(returns, 100 * (1 - self.confidence_level))

    def calculate_var_parametric(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate Value at Risk using the parametric method (assuming normal distribution).
        
        :param returns: Series or array of historical returns
        :return: VaR value
        """
        mu = np.mean(returns)
        sigma = np.std(returns)
        return stats.norm.ppf(1 - self.confidence_level, mu, sigma)

    def calculate_var_monte_carlo(self, returns: Union[pd.Series, np.ndarray], num_simulations: int = 10000, time_horizon: int = 1) -> float:
        """
        Calculate Value at Risk using Monte Carlo simulation.
        
        :param returns: Series or array of historical returns
        :param num_simulations: Number of Monte Carlo simulations
        :param time_horizon: Time horizon for VaR calculation (in days)
        :return: VaR value
        """
        mu = np.mean(returns)
        sigma = np.std(returns)
        
        simulated_returns = np.random.normal(mu, sigma, size=(num_simulations, time_horizon)).sum(axis=1)
        return np.percentile(simulated_returns, 100 * (1 - self.confidence_level))

    def calculate_var(self, returns: Union[pd.Series, np.ndarray], method: str = 'historical') -> float:
        """
        Calculate Value at Risk using the specified method.
        
        :param returns: Series or array of historical returns
        :param method: Method to use for VaR calculation ('historical', 'parametric', or 'monte_carlo')
        :return: VaR value
        """
        if method == 'historical':
            return self.calculate_var_historical(returns)
        elif method == 'parametric':
            return self.calculate_var_parametric(returns)
        elif method == 'monte_carlo':
            return self.calculate_var_monte_carlo(returns)
        else:
            raise ValueError("Invalid method. Choose 'historical', 'parametric', or 'monte_carlo'.")

    def calculate_var_contribution(self, returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
        """
        Calculate VaR contribution for each asset in a portfolio.
        
        :param returns: DataFrame of asset returns
        :param weights: Array of asset weights in the portfolio
        :return: Series of VaR contributions
        """
        portfolio_returns = returns.dot(weights)
        var = self.calculate_var(portfolio_returns)
        
        marginal_var = np.dot(returns.cov(), weights) / np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
        var_contribution = weights * marginal_var * var
        
        return pd.Series(var_contribution, index=returns.columns)

    def backtest_var(self, returns: Union[pd.Series, np.ndarray], window: int = 252) -> Tuple[float, int]:
        """
        Backtest the VaR model.
        
        :param returns: Series or array of historical returns
        :param window: Rolling window size for VaR calculation
        :return: Tuple of (VaR violation rate, number of VaR violations)
        """
        var_violations = 0
        total_observations = len(returns) - window

        for i in range(window, len(returns)):
            historical_returns = returns[i-window:i]
            var = self.calculate_var(historical_returns)
            if returns[i] < var:
                var_violations += 1

        violation_rate = var_violations / total_observations
        return violation_rate, var_violations