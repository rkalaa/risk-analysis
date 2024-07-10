import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional

class BetaModel:
    def __init__(self):
        self.beta = None
        self.alpha = None
        self.r_squared = None
        self.std_error = None

    def calculate_beta(self, asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Calculate the beta of an asset relative to the market.
        
        :param asset_returns: Series of asset returns
        :param market_returns: Series of market returns
        :return: Beta value
        """
        covariance = np.cov(asset_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        self.beta = covariance / market_variance
        return self.beta

    def calculate_alpha(self, asset_returns: pd.Series, market_returns: pd.Series, risk_free_rate: float = 0) -> float:
        """
        Calculate the alpha of an asset.
        
        :param asset_returns: Series of asset returns
        :param market_returns: Series of market returns
        :param risk_free_rate: Risk-free rate (annualized)
        :return: Alpha value
        """
        if self.beta is None:
            self.calculate_beta(asset_returns, market_returns)
        
        asset_mean_return = asset_returns.mean() * 252  # Annualized
        market_mean_return = market_returns.mean() * 252  # Annualized
        self.alpha = asset_mean_return - (risk_free_rate + self.beta * (market_mean_return - risk_free_rate))
        return self.alpha

    def fit(self, asset_returns: pd.Series, market_returns: pd.Series) -> Tuple[float, float, float, float]:
        """
        Fit the market model (calculate beta, alpha, R-squared, and standard error).
        
        :param asset_returns: Series of asset returns
        :param market_returns: Series of market returns
        :return: Tuple of (beta, alpha, R-squared, standard error)
        """
        X = market_returns.values.reshape(-1, 1)
        y = asset_returns.values
        
        # Add a constant to X for the intercept term
        X = np.concatenate([np.ones_like(X), X], axis=1)
        
        # Perform linear regression
        self.alpha, self.beta = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # Calculate R-squared
        y_pred = self.alpha + self.beta * market_returns
        residuals = y - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        self.r_squared = 1 - (ss_res / ss_tot)
        
        # Calculate standard error of beta
        n = len(asset_returns)
        self.std_error = np.sqrt(ss_res / (n - 2)) / np.std(market_returns) / np.sqrt(n)
        
        return self.beta, self.alpha, self.r_squared, self.std_error

    def predict(self, market_returns: pd.Series) -> pd.Series:
        """
        Predict asset returns based on market returns using the fitted model.
        
        :param market_returns: Series of market returns
        :return: Series of predicted asset returns
        """
        if self.beta is None or self.alpha is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        return self.alpha + self.beta * market_returns

    def calculate_tracking_error(self, asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate the tracking error of an asset relative to a benchmark.
        
        :param asset_returns: Series of asset returns
        :param benchmark_returns: Series of benchmark returns
        :return: Tracking error
        """
        return np.std(asset_returns - benchmark_returns) * np.sqrt(252)  # Annualized

    def calculate_information_ratio(self, asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate the information ratio of an asset relative to a benchmark.
        
        :param asset_returns: Series of asset returns
        :param benchmark_returns: Series of benchmark returns
        :return: Information ratio
        """
        excess_returns = asset_returns - benchmark_returns
        return (excess_returns.mean() * 252) / self.calculate_tracking_error(asset_returns, benchmark_returns)

    def calculate_treynor_ratio(self, asset_returns: pd.Series, market_returns: pd.Series, risk_free_rate: float) -> float:
        """
        Calculate the Treynor ratio of an asset.
        
        :param asset_returns: Series of asset returns
        :param market_returns: Series of market returns
        :param risk_free_rate: Risk-free rate (annualized)
        :return: Treynor ratio
        """
        if self.beta is None:
            self.calculate_beta(asset_returns, market_returns)
        
        asset_mean_return = asset_returns.mean() * 252  # Annualized
        return (asset_mean_return - risk_free_rate) / self.beta

    def hypothesis_test(self, confidence_level: float = 0.95) -> Tuple[float, float, bool]:
        """
        Perform a hypothesis test on beta.
        
        :param confidence_level: Confidence level for the test
        :return: Tuple of (t-statistic, p-value, is_significant)
        """
        if self.beta is None or self.std_error is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        t_stat = self.beta / self.std_error
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(self.asset_returns) - 2))
        is_significant = p_value < (1 - confidence_level)
        
        return t_stat, p_value, is_significant

    def __str__(self) -> str:
        if self.beta is None:
            return "BetaModel (not fitted)"
        return f"BetaModel (beta={self.beta:.4f}, alpha={self.alpha:.4f}, R-squared={self.r_squared:.4f})"
