import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, List, Dict
from scipy import stats

class PortfolioOptimizer:
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.01):
        """
        Initialize the PortfolioOptimizer.

        :param returns: DataFrame of asset returns
        :param risk_free_rate: Risk-free rate of return
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.num_assets = len(returns.columns)
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()

    def calculate_portfolio_performance(self, weights: np.array) -> Tuple[float, float, float]:
        """
        Calculate the performance metrics of a portfolio.

        :param weights: Array of asset weights
        :return: Tuple of portfolio return, volatility, and Sharpe ratio
        """
        portfolio_return = np.sum(self.mean_returns * weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))) * np.sqrt(252)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        return portfolio_return, portfolio_volatility, sharpe_ratio

    def negative_sharpe_ratio(self, weights: np.array) -> float:
        """
        Calculate the negative Sharpe ratio of a portfolio.

        :param weights: Array of asset weights
        :return: Negative Sharpe ratio
        """
        return -self.calculate_portfolio_performance(weights)[2]

    def calculate_efficient_frontier(self, num_portfolios: int = 10000) -> pd.DataFrame:
        """
        Calculate the efficient frontier for a set of assets.

        :param num_portfolios: Number of portfolios to simulate
        :return: DataFrame with portfolio weights, returns, volatilities, and Sharpe ratios
        """
        results = []
        weights_record = []

        for _ in range(num_portfolios):
            weights = np.random.random(self.num_assets)
            weights /= np.sum(weights)
            weights_record.append(weights)

            portfolio_return, portfolio_volatility, sharpe_ratio = self.calculate_portfolio_performance(weights)
            results.append([portfolio_return, portfolio_volatility, sharpe_ratio] + list(weights))

        columns = ['ret', 'stdev', 'sharpe'] + [ticker for ticker in self.returns.columns]
        return pd.DataFrame(results, columns=columns)

    def optimize_portfolio(self, optimization_criterion: str = 'sharpe') -> Dict[str, float]:
        """
        Optimize a portfolio based on the specified criterion.

        :param optimization_criterion: 'sharpe' for maximum Sharpe ratio, 'return' for maximum return, 'volatility' for minimum volatility
        :return: Dictionary with optimal weights, return, volatility, and Sharpe ratio
        """
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_weights = np.array([1.0 / self.num_assets] * self.num_assets)

        if optimization_criterion == 'sharpe':
            result = minimize(self.negative_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        elif optimization_criterion == 'return':
            result = minimize(lambda weights: -self.calculate_portfolio_performance(weights)[0], initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        elif optimization_criterion == 'volatility':
            result = minimize(lambda weights: self.calculate_portfolio_performance(weights)[1], initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        else:
            raise ValueError("Invalid optimization criterion. Choose 'sharpe', 'return', or 'volatility'.")

        optimal_weights = result.x
        optimal_return, optimal_volatility, optimal_sharpe = self.calculate_portfolio_performance(optimal_weights)

        return {
            'weights': dict(zip(self.returns.columns, optimal_weights)),
            'return': optimal_return,
            'volatility': optimal_volatility,
            'sharpe_ratio': optimal_sharpe
        }

    def calculate_efficient_frontier_with_target_return(self, target_return: float) -> Dict[str, float]:
        """
        Calculate the minimum variance portfolio for a target return.

        :param target_return: Target portfolio return
        :return: Dictionary with optimal weights, return, and volatility
        """
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: self.calculate_portfolio_performance(x)[0] - target_return}
        )
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_weights = np.array([1.0 / self.num_assets] * self.num_assets)

        result = minimize(lambda weights: self.calculate_portfolio_performance(weights)[1], initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

        optimal_weights = result.x
        optimal_return, optimal_volatility, optimal_sharpe = self.calculate_portfolio_performance(optimal_weights)

        return {
            'weights': dict(zip(self.returns.columns, optimal_weights)),
            'return': optimal_return,
            'volatility': optimal_volatility
        }

    def perform_monte_carlo_simulation(self, weights: Dict[str, float], num_simulations: int = 1000, time_horizon: int = 252) -> List[pd.Series]:
        """
        Perform Monte Carlo simulation for portfolio returns.

        :param weights: Dictionary of asset weights
        :param num_simulations: Number of simulations to run
        :param time_horizon: Time horizon for simulation in trading days
        :return: List of simulated portfolio value series
        """
        weight_array = np.array([weights[ticker] for ticker in self.returns.columns])
        
        simulations = []
        for _ in range(num_simulations):
            sim_returns = np.random.multivariate_normal(self.mean_returns, self.cov_matrix, time_horizon)
            portfolio_returns = np.dot(sim_returns, weight_array)
            portfolio_values = 100 * (1 + portfolio_returns).cumprod()
            simulations.append(pd.Series(portfolio_values))

        return simulations

    def calculate_var_cvar(self, weights: Dict[str, float], confidence_level: float = 0.95, time_horizon: int = 252) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR) for a portfolio.

        :param weights: Dictionary of asset weights
        :param confidence_level: Confidence level for VaR and CVaR calculation
        :param time_horizon: Time horizon for calculation in trading days
        :return: Dictionary with VaR and CVaR values
        """
        weight_array = np.array([weights[ticker] for ticker in self.returns.columns])
        portfolio_returns = np.dot(self.returns, weight_array)
        
        var = -np.percentile(portfolio_returns, 100 * (1 - confidence_level)) * np.sqrt(time_horizon)
        cvar = -portfolio_returns[portfolio_returns <= -var / np.sqrt(time_horizon)].mean() * np.sqrt(time_horizon)

        return {
            'VaR': var,
            'CVaR': cvar
        }

    def calculate_risk_contribution(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate the risk contribution of each asset in the portfolio.

        :param weights: Dictionary of asset weights
        :return: Dictionary with risk contributions for each asset
        """
        weight_array = np.array([weights[ticker] for ticker in self.returns.columns])
        portfolio_volatility = self.calculate_portfolio_performance(weight_array)[1]
        
        marginal_risk = np.dot(self.cov_matrix, weight_array) / portfolio_volatility
        risk_contribution = weight_array * marginal_risk / portfolio_volatility

        return dict(zip(self.returns.columns, risk_contribution))

    def optimize_risk_parity_portfolio(self) -> Dict[str, float]:
        """
        Optimize a risk parity portfolio.

        :return: Dictionary with optimal weights, return, volatility, and Sharpe ratio
        """
        def risk_budget_objective(weights, args):
            cov_matrix = args[0]
            target_risk = args[1]
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            risk_contribution = weights * (np.dot(cov_matrix, weights)) / portfolio_volatility
            return ((risk_contribution - target_risk) ** 2).sum()

        target_risk = 1 / self.num_assets
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_weights = np.array([1.0 / self.num_assets] * self.num_assets)

        result = minimize(risk_budget_objective, initial_weights, args=[self.cov_matrix, target_risk], method='SLSQP', bounds=bounds, constraints=constraints)

        optimal_weights = result.x
        optimal_return, optimal_volatility, optimal_sharpe = self.calculate_portfolio_performance(optimal_weights)

        return {
            'weights': dict(zip(self.returns.columns, optimal_weights)),
            'return': optimal_return,
            'volatility': optimal_volatility,
            'sharpe_ratio': optimal_sharpe
        }

    def calculate_portfolio_skewness_kurtosis(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate portfolio skewness and kurtosis.

        :param weights: Dictionary of asset weights
        :return: Dictionary with portfolio skewness and kurtosis
        """
        weight_array = np.array([weights[ticker] for ticker in self.returns.columns])
        portfolio_returns = np.dot(self.returns, weight_array)
        
        skewness = stats.skew(portfolio_returns)
        kurtosis = stats.kurtosis(portfolio_returns)

        return {
            'skewness': skewness,
            'kurtosis': kurtosis
        }

    def optimize_portfolio_with_constraints(self, min_weights: Dict[str, float], max_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Optimize a portfolio with minimum and maximum weight constraints.

        :param min_weights: Dictionary of minimum weights for each asset
        :param max_weights: Dictionary of maximum weights for each asset
        :return: Dictionary with optimal weights, return, volatility, and Sharpe ratio
        """
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((min_weights[ticker], max_weights[ticker]) for ticker in self.returns.columns)
        initial_weights = np.array([1.0 / self.num_assets] * self.num_assets)

        result = minimize(self.negative_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

        optimal_weights = result.x
        optimal_return, optimal_volatility, optimal_sharpe = self.calculate_portfolio_performance(optimal_weights)

        return {
            'weights': dict(zip(self.returns.columns, optimal_weights)),
            'return': optimal_return,
            'volatility': optimal_volatility,
            'sharpe_ratio': optimal_sharpe
        }
