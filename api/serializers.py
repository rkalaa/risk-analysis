from flask_restx import fields, Model

portfolio_model = Model('Portfolio', {
    'returns': fields.Raw(required=True, description='Asset returns data'),
    'weights': fields.Raw(required=False, description='Asset weights'),
    'risk_free_rate': fields.Float(required=False, description='Risk-free rate'),
    'confidence_level': fields.Float(required=False, description='Confidence level for VaR and CVaR'),
    'num_simulations': fields.Integer(required=False, description='Number of simulations for Monte Carlo'),
    'time_horizon': fields.Integer(required=False, description='Time horizon for simulations')
})

optimization_result_model = Model('OptimizationResult', {
    'weights': fields.Raw(required=True, description='Optimized asset weights'),
    'return': fields.Float(required=True, description='Expected portfolio return'),
    'volatility': fields.Float(required=True, description='Portfolio volatility'),
    'sharpe_ratio': fields.Float(required=True, description='Portfolio Sharpe ratio')
})

risk_metrics_model = Model('RiskMetrics', {
    'VaR': fields.Float(required=True, description='Value at Risk'),
    'CVaR': fields.Float(required=True, description='Conditional Value at Risk')
})