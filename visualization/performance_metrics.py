import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

class PerformanceMetrics:
    @staticmethod
    def plot_cumulative_returns(returns: pd.DataFrame, title: str = "Cumulative Returns") -> None:
        """
        Plot cumulative returns for one or more assets.
        
        :param returns: DataFrame of returns for one or more assets
        :param title: Title of the plot
        """
        cumulative_returns = (1 + returns).cumprod()
        plt.figure(figsize=(12, 6))
        cumulative_returns.plot()
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns")
        plt.legend()
        plt.show()

    @staticmethod
    def plot_rolling_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0, window: int = 252, title: str = "Rolling Sharpe Ratio") -> None:
        """
        Plot rolling Sharpe ratio.
        
        :param returns: Series of returns
        :param risk_free_rate: Risk-free rate (annualized)
        :param window: Rolling window size
        :param title: Title of the plot
        """
        excess_returns = returns - risk_free_rate / 252
        rolling_sharpe = (excess_returns.rolling(window=window).mean() / excess_returns.rolling(window=window).std()) * np.sqrt(252)
        
        plt.figure(figsize=(12, 6))
        rolling_sharpe.plot()
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Sharpe Ratio")
        plt.show()

    @staticmethod
    def plot_underwater(returns: pd.Series, title: str = "Underwater Plot") -> None:
        """
        Plot underwater chart (drawdown over time).
        
        :param returns: Series of returns
        :param title: Title of the plot
        """
        cumulative_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        underwater = (cumulative_returns / running_max - 1) * 100
        
        plt.figure(figsize=(12, 6))
        underwater.plot(kind='area', color='red', alpha=0.3)
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Drawdown (%)")
        plt.show()

    @staticmethod
    def plot_monthly_returns_heatmap(returns: pd.Series, title: str = "Monthly Returns Heatmap") -> None:
        """
        Plot heatmap of monthly returns.
        
        :param returns: Series of returns
        :param title: Title of the plot
        """
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns = monthly_returns.unstack().T
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(monthly_returns, annot=True, fmt=".2%", cmap="RdYlGn", center=0)
        plt.title(title)
        plt.xlabel("Year")
        plt.ylabel("Month")
        plt.show()

    @staticmethod
    def plot_rolling_beta(asset_returns: pd.Series, market_returns: pd.Series, window: int = 252, title: str = "Rolling Beta") -> None:
        """
        Plot rolling beta of an asset against the market.
        
        :param asset_returns: Series of asset returns
        :param market_returns: Series of market returns
        :param window: Rolling window size
        :param title: Title of the plot
        """
        rolling_cov = asset_returns.rolling(window=window).cov(market_returns)
        rolling_market_var = market_returns.rolling(window=window).var()
        rolling_beta = rolling_cov / rolling_market_var
        
        plt.figure(figsize=(12, 6))
        rolling_beta.plot()
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Beta")
        plt.show()

    @staticmethod
    def plot_return_quantiles(returns: pd.Series, title: str = "Return Quantiles") -> None:
        """
        Plot return quantiles.
        
        :param returns: Series of returns
        :param title: Title of the plot
        """
        plt.figure(figsize=(12, 6))
        returns.hist(bins=50, density=True, cumulative=True, histtype='step')
        plt.title(title)
        plt.xlabel("Returns")
        plt.ylabel("Cumulative Probability")
        plt.show()

    @staticmethod
    def plot_rolling_volatility(returns: pd.Series, window: int = 30, title: str = "Rolling Volatility") -> None:
        """
        Plot rolling volatility.
        
        :param returns: Series of returns
        :param window: Rolling window size
        :param title: Title of the plot
        """
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        plt.figure(figsize=(12, 6))
        rolling_vol.plot()
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Annualized Volatility")
        plt.show()

    @staticmethod
    def plot_rolling_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0, window: int = 252, title: str = "Rolling Sortino Ratio") -> None:
        """
        Plot rolling Sortino ratio.
        
        :param returns: Series of returns
        :param risk_free_rate: Risk-free rate (annualized)
        :param window: Rolling window size
        :param title: Title of the plot
        """
        excess_returns = returns - risk_free_rate / 252
        negative_returns = excess_returns.copy()
        negative_returns[negative_returns > 0] = 0
        rolling_downside_std = negative_returns.rolling(window=window).std() * np.sqrt(252)
        rolling_sortino = (excess_returns.rolling(window=window).mean() * 252) / rolling_downside_std
        
        plt.figure(figsize=(12, 6))
        rolling_sortino.plot()
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Sortino Ratio")
        plt.show()

    @staticmethod
    def plot_rolling_calmar_ratio(returns: pd.Series, window: int = 252, title: str = "Rolling Calmar Ratio") -> None:
        """
        Plot rolling Calmar ratio.
        
        :param returns: Series of returns
        :param window: Rolling window size
        :param title: Title of the plot
        """
        def max_drawdown(return_series):
            comp_ret = (return_series + 1).cumprod()
            peak = comp_ret.expanding(min_periods=1).max()
            dd = (comp_ret/peak) - 1
            return dd.min()
        
        rolling_return = returns.rolling(window=window).mean() * 252
        rolling_max_drawdown = returns.rolling(window=window).apply(max_drawdown)
        rolling_calmar_ratio = -rolling_return / rolling_max_drawdown
        
        plt.figure(figsize=(12, 6))
        rolling_calmar_ratio.plot()
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Calmar Ratio")
        plt.show()

    @staticmethod
    def plot_rolling_information_ratio(returns: pd.Series, benchmark_returns: pd.Series, window: int = 252, title: str = "Rolling Information Ratio") -> None:
        """
        Plot rolling Information ratio.
        
        :param returns: Series of returns
        :param benchmark_returns: Series of benchmark returns
        :param window: Rolling window size
        :param title: Title of the plot
        """
        excess_returns = returns - benchmark_returns
        rolling_return = excess_returns.rolling(window=window).mean() * 252
        rolling_tracking_error = excess_returns.rolling(window=window).std() * np.sqrt(252)
        rolling_information_ratio = rolling_return / rolling_tracking_error
        
        plt.figure(figsize=(12, 6))
        rolling_information_ratio.plot()
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Information Ratio")
        plt.show()

    @staticmethod
    def plot_drawdown_periods(returns: pd.Series, top_n: int = 5, title: str = f"Top Drawdown Periods") -> None:
        """
        Plot top drawdown periods.
        
        :param returns: Series of returns
        :param top_n: Number of top drawdown periods to plot
        :param title: Title of the plot
        """
        def drawdown_periods(returns):
            wealth_index = (1 + returns).cumprod()
            previous_peaks = wealth_index.cummax()
            drawdowns = (wealth_index - previous_peaks) / previous_peaks
            
            drawdown_starts = np.where(drawdowns == 0)[0]
            drawdown_ends = np.roll(drawdown_starts, -1) - 1
            drawdown_ends[-1] = len(returns) - 1
            
            drawdown_periods = pd.DataFrame({
                'start': returns.index[drawdown_starts],
                'end': returns.index[drawdown_ends],
                'drawdown': drawdowns.iloc[drawdown_ends].values
            })
            
            return drawdown_periods.sort_values('drawdown').head(top_n)
        
        top_drawdowns = drawdown_periods(returns)
        wealth_index = (1 + returns).cumprod()
        
        plt.figure(figsize=(12, 6))
        wealth_index.plot()
        
        for _, drawdown in top_drawdowns.iterrows():
            plt.fill_between(wealth_index.loc[drawdown['start']:drawdown['end']].index, 
                             wealth_index.loc[drawdown['start']:drawdown['end']], 
                             wealth_index.loc[drawdown['start']], alpha=0.3)
        
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Wealth Index")
        plt.show()
