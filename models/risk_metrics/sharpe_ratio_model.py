import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional

class SharpeRatioModel:
    def __init__(self):
        self.sharpe_ratio = None
        self.annualized_return = None
        self.annualized_volatility = None

    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0, periods: int = 252) -> float:
        """
        Calculate the Sharpe ratio.
        
        :param returns: Series of asset returns
        :param risk_free_rate: Risk-free rate (annualized)
        :param periods: Number of periods in a year (e.g., 252 for daily data, 12 for monthly)
        :return: Sharpe ratio
        """
        self.annualized_return = returns.mean() * periods
        self.annualized_volatility = returns.std() * np.sqrt(periods)
        self.sharpe_ratio = (self.annualized_return - risk_free_rate) / self.annualized_volatility
        return self.sharpe_ratio

    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0, periods: int = 252) -> float:
        """
        Calculate the Sortino ratio.
        
        :param returns: Series of asset returns
        :param risk_free_rate: Risk-free rate (annualized)
        :param periods: Number of periods in a year
        :return: Sortino ratio
        """
        self.annualized_return = returns.mean() * periods
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(periods)
        return (self.annualized_return - risk_free_rate) / downside_volatility

    def calculate_rolling_sharpe_ratio(self, returns: pd.Series, window: int, risk_free_rate: float = 0, periods: int = 252) -> pd.Series:
        """
        Calculate rolling Sharpe ratio.
        
        :param returns: Series of asset returns
        :param window: Rolling window size
        :param risk_free_rate: Risk-free rate (annualized)
        :param periods: Number of periods in a year
        :return: Series of rolling Sharpe ratios
        """
        rolling_return = returns.rolling(window=window).mean() * periods
        rolling_volatility = returns.rolling(window=window).std() * np.sqrt(periods)
        return (rolling_return - risk_free_rate) / rolling_volatility

    def calculate_information_ratio(self, returns: pd.Series, benchmark_returns: pd.Series, periods: int = 252) -> float:
        """
        Calculate the Information ratio.
        
        :param returns: Series of asset returns
        :param benchmark_returns: Series of benchmark returns
        :param periods: Number of periods in a year
        :return: Information ratio
        """
        excess_returns = returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(periods)
        return (excess_returns.mean() * periods) / tracking_error

    def calculate_modigliani_ratio(self, returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0, periods: int = 252) -> float:
        """
        Calculate the Modigliani ratio (M2).
        
        :param returns: Series of asset returns
        :param benchmark_returns: Series of benchmark returns
        :param risk_free_rate: Risk-free rate (annualized)
        :param periods: Number of periods in a year
        :return: Modigliani ratio
        """
        sharpe_ratio = self.calculate_sharpe_ratio(returns, risk_free_rate, periods)
        benchmark_volatility = benchmark_returns.std() * np.sqrt(periods)
        return sharpe_ratio * benchmark_volatility + risk_free_rate

    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) using the historical method.
        
        :param returns: Series of asset returns
        :param confidence_level: Confidence level for VaR calculation
        :return: VaR value
        """
        return -np.percentile(returns, 100 * (1 - confidence_level))

    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) using the historical method.
        
        :param returns: Series of asset returns
        :param confidence_level: Confidence level for CVaR calculation
        :return: CVaR value
        """
        var = self.calculate_var(returns, confidence_level)
        return -returns[returns <= -var].mean()

    def calculate_downside_deviation(self, returns: pd.Series, threshold: float = 0) -> float:
        """
        Calculate downside deviation.
        
        :param returns: Series of asset returns
        :param threshold: Minimum acceptable return
        :return: Downside deviation
        """
        downside_returns = returns[returns < threshold]
        return np.sqrt(np.mean((downside_returns - threshold)**2))

    def bootstrap_sharpe_ratio(self, returns: pd.Series, num_simulations: int = 10000, confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Perform bootstrap analysis of Sharpe ratio.
        
        :param returns: Series of asset returns
        :param num_simulations: Number of bootstrap simulations
        :param confidence_level: Confidence level for interval estimation
        :return: Tuple of (lower bound, upper bound) of Sharpe ratio confidence interval
        """
        bootstrap_sharpe_ratios = []
        for _ in range(num_simulations):
            sample = returns.sample(n=len(returns), replace=True)
            bootstrap_sharpe_ratios.append(self.calculate_sharpe_ratio(sample))
        
        confidence_interval = np.percentile(bootstrap_sharpe_ratios, [(1 - confidence_level) / 2 * 100, (1 + confidence_level) / 2 * 100])
        return tuple(confidence_interval)

    def calculate_calmar_ratio(self, returns: pd.Series, periods: int = 252) -> float:
        """
        Calculate the Calmar ratio.
        
        :param returns: Series of asset returns
        :param periods: Number of periods in a year
        :return: Calmar ratio
        """
        max_drawdown = self.calculate_max_drawdown(returns)
        annualized_return = returns.mean() * periods
        return annualized_return / abs(max_drawdown)

    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate the maximum drawdown.
        
        :param returns: Series of asset returns
        :return: Maximum drawdown
        """
        wealth_index = (1 + returns).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        return drawdowns.min()

    def calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0) -> float:
        """
        Calculate the Omega ratio.
        
        :param returns: Series of asset returns
        :param threshold: Target return threshold
        :return: Omega ratio
        """
        return_threshold = returns - threshold
        positive_returns = return_threshold[return_threshold > 0].sum()
        negative_returns = abs(return_threshold[return_threshold < 0].sum())
        return positive_returns / negative_returns

    def __str__(self) -> str:
        if self.sharpe_ratio is None:
            return "SharpeRatioModel (not calculated)"
        return f"SharpeRatioModel (Sharpe Ratio={self.sharpe_ratio:.4f}, Annualized Return={self.annualized_return:.4f}, Annualized Volatility={self.annualized_volatility:.4f})"
