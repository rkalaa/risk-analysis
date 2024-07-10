import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, Tuple, List

def calculate_returns(prices: Union[pd.Series, np.ndarray], method: str = 'simple') -> Union[pd.Series, np.ndarray]:
    """
    Calculate returns from a series of prices.
    
    :param prices: Series or array of prices
    :param method: 'simple' or 'log'
    :return: Series or array of returns
    """
    if method == 'simple':
        return (prices[1:] / prices[:-1]) - 1
    elif method == 'log':
        return np.log(prices[1:] / prices[:-1])
    else:
        raise ValueError("method must be either 'simple' or 'log'")

def calculate_sharpe_ratio(returns: Union[pd.Series, np.ndarray], risk_free_rate: float = 0, periods: int = 252) -> float:
    """
    Calculate the Sharpe ratio for a series of returns.
    
    :param returns: Series or array of returns
    :param risk_free_rate: Risk-free rate (annualized)
    :param periods: Number of periods in a year
    :return: Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / periods
    return np.sqrt(periods) * excess_returns.mean() / excess_returns.std()

def calculate_sortino_ratio(returns: Union[pd.Series, np.ndarray], risk_free_rate: float = 0, periods: int = 252) -> float:
    """
    Calculate the Sortino ratio for a series of returns.
    
    :param returns: Series or array of returns
    :param risk_free_rate: Risk-free rate (annualized)
    :param periods: Number of periods in a year
    :return: Sortino ratio
    """
    excess_returns = returns - risk_free_rate / periods
    downside_returns = excess_returns[excess_returns < 0]
    downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(periods)
    return excess_returns.mean() * periods / downside_deviation

def calculate_maximum_drawdown(prices: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate the maximum drawdown from a series of prices.
    
    :param prices: Series or array of prices
    :return: Maximum drawdown
    """
    peak = np.maximum.accumulate(prices)
    drawdown = (prices - peak) / peak
    return np.min(drawdown)

def calculate_var(returns: Union[pd.Series, np.ndarray], confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR) using the historical method.
    
    :param returns: Series or array of returns
    :param confidence_level: Confidence level for VaR calculation
    :return: VaR value
    """
    return np.percentile(returns, 100 * (1 - confidence_level))

def calculate_cvar(returns: Union[pd.Series, np.ndarray], confidence_level: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) using the historical method.
    
    :param returns: Series or array of returns
    :param confidence_level: Confidence level for CVaR calculation
    :return: CVaR value
    """
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()

def calculate_beta(asset_returns: Union[pd.Series, np.ndarray], market_returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate the beta of an asset relative to the market.
    
    :param asset_returns: Series or array of asset returns
    :param market_returns: Series or array of market returns
    :return: Beta value
    """
    covariance = np.cov(asset_returns, market_returns)[0, 1]
    market_variance = np.var(market_returns)
    return covariance / market_variance

def calculate_alpha(asset_returns: Union[pd.Series, np.ndarray], market_returns: Union[pd.Series, np.ndarray], risk_free_rate: float = 0) -> float:
    """
    Calculate the alpha of an asset.
    
    :param asset_returns: Series or array of asset returns
    :param market_returns: Series or array of market returns
    :param risk_free_rate: Risk-free rate (annualized)
    :return: Alpha value
    """
    beta = calculate_beta(asset_returns, market_returns)
    return asset_returns.mean() - (risk_free_rate + beta * (market_returns.mean() - risk_free_rate))

def calculate_rolling_volatility(returns: Union[pd.Series, np.ndarray], window: int = 21, periods: int = 252) -> Union[pd.Series, np.ndarray]:
    """
    Calculate rolling volatility.
    
    :param returns: Series or array of returns
    :param window: Rolling window size
    :param periods: Number of periods in a year
    :return: Series or array of rolling volatility
    """
    if isinstance(returns, pd.Series):
        return returns.rolling(window=window).std() * np.sqrt(periods)
    else:
        return pd.Series(returns).rolling(window=window).std() * np.sqrt(periods)

def calculate_ewma_volatility(returns: Union[pd.Series, np.ndarray], lambda_param: float = 0.94, periods: int = 252) -> Union[pd.Series, np.ndarray]:
    """
    Calculate volatility using Exponentially Weighted Moving Average (EWMA).
    
    :param returns: Series or array of returns
    :param lambda_param: Decay factor for EWMA
    :param periods: Number of periods in a year
    :return: Series or array of EWMA volatility
    """
    if isinstance(returns, pd.Series):
        return returns.ewm(alpha=1-lambda_param).std() * np.sqrt(periods)
    else:
        return pd.Series(returns).ewm(alpha=1-lambda_param).std() * np.sqrt(periods)

def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the correlation matrix for a DataFrame of returns.
    
    :param returns: DataFrame of returns
    :return: Correlation matrix
    """
    return returns.corr()

def calculate_covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the covariance matrix for a DataFrame of returns.
    
    :param returns: DataFrame of returns
    :return: Covariance matrix
    """
    return returns.cov()

def perform_jarque_bera_test(returns: Union[pd.Series, np.ndarray]) -> Tuple[float, float]:
    """
    Perform Jarque-Bera test for normality.
    
    :param returns: Series or array of returns
    :return: Tuple of (test statistic, p-value)
    """
    return stats.jarque_bera(returns)

def calculate_kurtosis(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate the kurtosis of returns.
    
    :param returns: Series or array of returns
    :return: Kurtosis value
    """
    return stats.kurtosis(returns)

def calculate_skewness(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate the skewness of returns.
    
    :param returns: Series or array of returns
    :return: Skewness value
    """
    return stats.skew(returns)

def calculate_information_ratio(returns: Union[pd.Series, np.ndarray], benchmark_returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate the Information Ratio.
    
    :param returns: Series or array of asset returns
    :param benchmark_returns: Series or array of benchmark returns
    :return: Information Ratio
    """
    excess_returns = returns - benchmark_returns
    return excess_returns.mean() / excess_returns.std()

def calculate_treynor_ratio(returns: Union[pd.Series, np.ndarray], market_returns: Union[pd.Series, np.ndarray], risk_free_rate: float = 0) -> float:
    """
    Calculate the Treynor Ratio.
    
    :param returns: Series or array of asset returns
    :param market_returns: Series or array of market returns
    :param risk_free_rate: Risk-free rate (annualized)
    :return: Treynor Ratio
    """
    beta = calculate_beta(returns, market_returns)
    excess_return = returns.mean() - risk_free_rate
    return excess_return / beta


    # Generate some sample data
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", end="2022-12-31", freq="B")
    asset_prices = pd.Series(np.random.randn(len(dates)).cumsum() + 100, index=dates)
    market_prices = pd.Series(np.random.randn(len(dates)).cumsum() + 100, index=dates)

    asset_returns = calculate_returns(asset_prices)
    market_returns = calculate_returns(market_prices)

    print("Sharpe Ratio:", calculate_sharpe_ratio(asset_returns))
    print("Sortino Ratio:", calculate_sortino_ratio(asset_returns))
    print("Maximum Drawdown:", calculate_maximum_drawdown(asset_prices))
    print("VaR (95%):", calculate_var(asset_returns))
    print("CVaR (95%):", calculate_cvar(asset_returns))
    print("Beta:", calculate_beta(asset_returns, market_returns))
    print("Alpha:", calculate_alpha(asset_returns, market_returns))
    print("Information Ratio:", calculate_information_ratio(asset_returns, market_returns))
    print("Treynor Ratio:", calculate_treynor_ratio(asset_returns, market_returns))

    # Perform Jarque-Bera test
    jb_statistic, jb_pvalue = perform_jarque_bera_test(asset_returns)
    print("Jarque-Bera test - Statistic:", jb_statistic, "p-value:", jb_pvalue)

    print("Kurtosis:", calculate_kurtosis(asset_returns))
    print("Skewness:", calculate_skewness(asset_returns))

    # Calculate and print correlation matrix
    returns_df = pd.DataFrame({"Asset": asset_returns, "Market": market_returns})
    correlation_matrix = calculate_correlation_matrix(returns_df)
    print("\nCorrelation Matrix:")
    print(correlation_matrix)