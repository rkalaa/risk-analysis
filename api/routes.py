from flask_restx import Namespace, Resource, fields
from .serializers import portfolio_model, optimization_result_model, risk_metrics_model
from .validators import validate_portfolio_data
from analysis.portfolio_optimization import PortfolioOptimizer
import pandas as pd

api = Namespace('risk_assessment', description='Financial risk assessment operations')

@api.route('/optimize_portfolio')
class OptimizePortfolio(Resource):
    @api.expect(portfolio_model)
    @api.marshal_with(optimization_result_model)
    def post(self):
        """Optimize a portfolio for maximum Sharpe ratio"""
        data = api.payload
        validate_portfolio_data(data)
        
        returns = pd.DataFrame(data['returns'])
        risk_free_rate = data.get('risk_free_rate', 0.01)
        
        optimizer = PortfolioOptimizer(returns, risk_free_rate)
        result = optimizer.optimize_portfolio('sharpe')
        
        return result

@api.route('/efficient_frontier')
class EfficientFrontier(Resource):
    @api.expect(portfolio_model)
    def post(self):
        """Calculate the efficient frontier for a set of assets"""
        data = api.payload
        validate_portfolio_data(data)
        
        returns = pd.DataFrame(data['returns'])
        risk_free_rate = data.get('risk_free_rate', 0.01)
        
        optimizer = PortfolioOptimizer(returns, risk_free_rate)
        efficient_frontier = optimizer.calculate_efficient_frontier()
        
        return efficient_frontier.to_dict(orient='records')

@api.route('/calculate_risk_metrics')
class RiskMetrics(Resource):
    @api.expect(portfolio_model)
    @api.marshal_with(risk_metrics_model)
    def post(self):
        """Calculate VaR and CVaR for a given portfolio"""
        data = api.payload
        validate_portfolio_data(data)
        
        returns = pd.DataFrame(data['returns'])
        weights = data['weights']
        risk_free_rate = data.get('risk_free_rate', 0.01)
        confidence_level = data.get('confidence_level', 0.95)
        
        optimizer = PortfolioOptimizer(returns, risk_free_rate)
        risk_metrics = optimizer.calculate_var_cvar(weights, confidence_level)
        
        return risk_metrics

@api.route('/monte_carlo_simulation')
class MonteCarloSimulation(Resource):
    @api.expect(portfolio_model)
    def post(self):
        """Perform Monte Carlo simulation for portfolio returns"""
        data = api.payload
        validate_portfolio_data(data)
        
        returns = pd.DataFrame(data['returns'])
        weights = data['weights']
        risk_free_rate = data.get('risk_free_rate', 0.01)
        num_simulations = data.get('num_simulations', 1000)
        time_horizon = data.get('time_horizon', 252)
        
        optimizer = PortfolioOptimizer(returns, risk_free_rate)
        simulations = optimizer.perform_monte_carlo_simulation(weights, num_simulations, time_horizon)
        
        return [sim.tolist() for sim in simulations]

@api.route('/optimize_risk_parity')
class RiskParityOptimization(Resource):
    @api.expect(portfolio_model)
    @api.marshal_with(optimization_result_model)
    def post(self):
        """Optimize a risk parity portfolio"""
        data = api.payload
        validate_portfolio_data(data)
        
        returns = pd.DataFrame(data['returns'])
        risk_free_rate = data.get('risk_free_rate', 0.01)
        
        optimizer = PortfolioOptimizer(returns, risk_free_rate)
        result = optimizer.optimize_risk_parity_portfolio()
        
        return result