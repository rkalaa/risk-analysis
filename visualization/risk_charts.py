import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

class RiskCharts:
    @staticmethod
    def plot_returns_distribution(returns: pd.Series, title: str = "Returns Distribution") -> None:
        """
        Plot the distribution of returns.
        
        :param returns: Series of returns
        :param title: Title of the plot
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(returns, kde=True)
        plt.title(title)
        plt.xlabel("Returns")
        plt.ylabel("Frequency")
        plt.show()

    @staticmethod
    def plot_qq_plot(returns: pd.Series, title: str = "Q-Q Plot") -> None:
        """
        Plot a Q-Q plot to compare returns distribution to normal distribution.
        
        :param returns: Series of returns
        :param title: Title of the plot
        """
        plt.figure(figsize=(10, 6))
        stats.probplot(returns, dist="norm", plot=plt)
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_volatility(volatility: pd.Series, title: str = "Volatility Over Time") -> None:
        """
        Plot volatility over time.
        
        :param volatility: Series of volatility values
        :param title: Title of the plot
        """
        plt.figure(figsize=(12, 6))
        volatility.plot()
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Volatility")
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
        plt.ylabel("Volatility")
        plt.show()

    @staticmethod
    def plot_var_cvar(returns: pd.Series, confidence_level: float = 0.95, title: str = "VaR and CVaR") -> None:
        """
        Plot Value at Risk (VaR) and Conditional Value at Risk (CVaR).
        
        :param returns: Series of returns
        :param confidence_level: Confidence level for VaR and CVaR calculation
        :param title: Title of the plot
        """
        var = np.percentile(returns, 100 * (1 - confidence_level))
        cvar = returns[returns <= var].mean()
        
        plt.figure(figsize=(12, 6))
        sns.histplot(returns, kde=True)
        plt.axvline(var, color='r', linestyle='dashed', linewidth=2, label=f'VaR {confidence_level:.0%}')
        plt.axvline(cvar, color='g', linestyle='dashed', linewidth=2, label=f'CVaR {confidence_level:.0%}')
        plt.title(title)
        plt.xlabel("Returns")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

    @staticmethod
    def plot_drawdown(prices: pd.Series, title: str = "Drawdown") -> None:
        """
        Plot drawdown over time.
        
        :param prices: Series of prices
        :param title: Title of the plot
        """
        drawdown = (prices - prices.cummax()) / prices.cummax()
        plt.figure(figsize=(12, 6))
        drawdown.plot()
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.show()

    @staticmethod
    def plot_correlation_heatmap(returns: pd.DataFrame, title: str = "Correlation Heatmap") -> None:
        """
        Plot correlation heatmap for multiple assets.
        
        :param returns: DataFrame of returns for multiple assets
        :param title: Title of the plot
        """
        correlation_matrix = returns.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_beta_comparison(asset_returns: pd.Series, market_returns: pd.Series, title: str = "Beta Comparison") -> None:
        """
        Plot asset returns against market returns to visualize beta.
        
        :param asset_returns: Series of asset returns
        :param market_returns: Series of market returns
        :param title: Title of the plot
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(market_returns, asset_returns, alpha=0.5)
        plt.xlabel("Market Returns")
        plt.ylabel("Asset Returns")
        plt.title(title)
        
        # Add regression line
        coefficients = np.polyfit(market_returns, asset_returns, deg=1)
        poly = np.poly1d(coefficients)
        plt.plot(market_returns, poly(market_returns), color='r', linestyle='--')
        
        plt.text(0.05, 0.95, f'Beta: {coefficients[0]:.2f}', transform=plt.gca().transAxes, verticalalignment='top')
        plt.show()

    @staticmethod
    def plot_efficient_frontier(returns: pd.DataFrame, num_portfolios: int = 10000, risk_free_rate: float = 0.02) -> None:
        """
        Plot the efficient frontier.
        
        :param returns: DataFrame of asset returns
        :param num_portfolios: Number of random portfolios to generate
        :param risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        def portfolio_annualized_performance(weights, mean_returns, cov_matrix):
            returns = np.sum(mean_returns * weights) * 252
            std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
            return std, returns

        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        num_assets = len(mean_returns)
        results = np.zeros((3, num_portfolios))
        weights_record = []

        for i in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            weights_record.append(weights)
            portfolio_std_dev, portfolio_return = portfolio_annualized_performance(weights, mean_returns, cov_matrix)
            results[0,i] = portfolio_std_dev
            results[1,i] = portfolio_return
            results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev

        plt.figure(figsize=(10, 6))
        plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='YlGnBu', marker='o', s=10, alpha=0.3)
        plt.colorbar(label='Sharpe ratio')
        plt.xlabel('Annualized Risk')
        plt.ylabel('Annualized Returns')
        plt.title('Efficient Frontier')
        plt.show()
