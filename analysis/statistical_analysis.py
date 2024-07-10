import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests

def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    """
    Calculate returns from a series of prices.
    
    :param prices: Series of prices
    :param method: 'simple' or 'log'
    :return: Series of returns
    """
    if method == 'simple':
        return prices.pct_change()
    elif method == 'log':
        return np.log(prices / prices.shift(1))
    else:
        raise ValueError("method must be either 'simple' or 'log'")

def calculate_volatility(returns: pd.Series, window: int = 252, trading_periods: int = 252) -> pd.Series:
    """
    Calculate the volatility of returns.
    
    :param returns: Series of returns
    :param window: Rolling window for calculation
    :param trading_periods: Number of trading periods in a year
    :return: Series of volatility
    """
    return returns.rolling(window=window).std() * np.sqrt(trading_periods)

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float, periods: int = 252) -> float:
    """
    Calculate the Sharpe ratio for a series of returns.
    
    :param returns: Series of returns
    :param risk_free_rate: The risk-free rate of return
    :param periods: Number of periods in a year
    :return: Sharpe ratio
    """
    return (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(periods)

def perform_normality_test(returns: pd.Series) -> dict:
    """
    Perform normality test on returns.
    
    :param returns: Series of returns
    :return: Dictionary with test results
    """
    statistic, p_value = stats.jarque_bera(returns)
    return {
        'test': 'Jarque-Bera',
        'statistic': statistic,
        'p_value': p_value,
        'normal': p_value > 0.05
    }

def calculate_var_historic(returns: pd.Series, level: float = 0.05) -> float:
    """
    Calculate Value at Risk using historical method.
    
    :param returns: Series of returns
    :param level: VaR level (e.g., 0.05 for 95% VaR)
    :return: VaR value
    """
    return np.percentile(returns, level * 100)

def calculate_cvar_historic(returns: pd.Series, level: float = 0.05) -> float:
    """
    Calculate Conditional Value at Risk using historical method.
    
    :param returns: Series of returns
    :param level: CVaR level (e.g., 0.05 for 95% CVaR)
    :return: CVaR value
    """
    var = calculate_var_historic(returns, level)
    return returns[returns <= var].mean()

def perform_granger_causality_test(series1: pd.Series, series2: pd.Series, max_lag: int = 5) -> dict:
    """
    Perform Granger Causality test between two time series.
    
    :param series1: First time series
    :param series2: Second time series
    :param max_lag: Maximum number of lags to test
    :return: Dictionary with test results
    """
    data = pd.concat([series1, series2], axis=1)
    result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
    
    return {lag: {'ssr_ftest': test[0]['ssr_ftest'][1]} for lag, test in result.items()}