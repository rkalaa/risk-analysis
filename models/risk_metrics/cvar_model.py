import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, Tuple

class CVaRModel:
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level

    def calculate_cvar_historical(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate Conditional Value at Risk using the historical method.
        
        :param returns: Series or array of historical returns
        :return: CVaR value
        """
        var = np.percentile(returns, 100 * (1 - self.confidence_level))
        return -np.mean(returns[returns <= var])

    def calculate_cvar_parametric(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate Conditional Value at Risk using the parametric method (assuming normal distribution).
        
        :param returns: Series or array of historical returns
        :return: CVaR value
        """
        mu = np.mean(returns)
        sigma = np.std(returns)
        var = stats.norm.ppf(1 - self.confidence_level, mu, sigma)
        return -(mu + sigma * stats.norm.pdf(stats.norm.ppf(1 - self.confidence_level)) / (1 - self.confidence_level))

    def calculate_cvar_monte_carlo(self, returns: Union[pd.Series, np.ndarray], num_simulations: int = 10000, time_horizon: int = 1) -> float:
        """
        Calculate Conditional Value at Risk using Monte Carlo simulation.
        
        :param returns: Series or array of historical returns
        :param num_simulations: Number of Monte Carlo simulations
        :param time_horizon: Time horizon for CVaR calculation (in days)
        :return: CVaR value
        """
        mu = np.mean(returns)
        sigma = np.std(returns)
        
        simulated_returns = np.random.normal(mu, sigma, size=(num_simulations, time_horizon)).sum(axis=1)
        var = np.percentile(simulated_returns, 100 * (1 - self.confidence_level))
        return -np.mean(simulated_returns[simulated_returns <= var])

    def calculate_cvar(self, returns: Union[pd.Series, np.ndarray], method: str = 'historical') -> float:
        """
        Calculate Conditional Value at Risk using the specified method.
        
        :param returns: Series or array of historical returns
        :param method: Method to use for CVaR calculation ('historical', 'parametric', or 'monte_carlo')
        :return: CVaR value
        """
        if method == 'historical':
            return self.calculate_cvar_historical(returns)
        elif method == 'parametric':
            return self.calculate_cvar_parametric(returns)
        elif method == 'monte_carlo':
            return self.calculate_cvar_monte_carlo(returns)
        else:
            raise ValueError("Invalid method. Choose 'historical', 'parametric', or 'monte_carlo'.")

    def calculate_cvar_contribution(self, returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
        """
        Calculate CVaR contribution for each asset in a portfolio.
        
        :param returns: DataFrame of asset returns
        :param weights: Array of asset weights in the portfolio
        :return: Series of CVaR contributions
        """
        portfolio_returns = returns.dot(weights)
        cvar = self.calculate_cvar(portfolio_returns)
        var = np.percentile(portfolio_returns, 100 * (1 - self.confidence_level))
        
        conditional_returns = returns[portfolio_returns <= var]
        cvar_contribution = weights * (conditional_returns.mean() * cvar / (conditional_returns.mean().dot(weights)))
        
        return pd.Series(cvar_contribution, index=returns.columns)

    def backtest_cvar(self, returns: Union[pd.Series, np.ndarray], window: int = 252) -> Tuple[float, float]:
        """
        Backtest the CVaR model.
        
        :param returns: Series or array of historical returns
        :param window: Rolling window size for CVaR calculation
        :return: Tuple of (Average CVaR violation, CVaR violation rate)
        """
        cvar_violations = []
        total_observations = len(returns) - window

        for i in range(window, len(returns)):
            historical_returns = returns[i-window:i]
            cvar = self.calculate_cvar(historical_returns)
            if returns[i] < -cvar:
                cvar_violations.append(abs(returns[i] + cvar))

        avg_cvar_violation = np.mean(cvar_violations) if cvar_violations else 0
        violation_rate = len(cvar_violations) / total_observations
        return avg_cvar_violation, violation_rate