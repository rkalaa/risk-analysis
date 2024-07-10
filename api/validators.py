from flask_restx import abort
import pandas as pd

def validate_portfolio_data(data):
    """
    Validate the input data for portfolio operations.
    """
    if 'returns' not in data:
        abort(400, 'Returns data is required')
    
    try:
        returns = pd.DataFrame(data['returns'])
    except ValueError:
        abort(400, 'Invalid returns data format')
    
    if returns.empty:
        abort(400, 'Returns data is empty')
    
    if 'weights' in data:
        weights = data['weights']
        if not isinstance(weights, dict):
            abort(400, 'Weights must be provided as a dictionary')
        if set(weights.keys()) != set(returns.columns):
            abort(400, 'Weights keys must match returns column names')
        if not all(isinstance(w, (int, float)) for w in weights.values()):
            abort(400, 'All weight values must be numeric')
        if abs(sum(weights.values()) - 1) > 1e-6:
            abort(400, 'Weights must sum to 1')
    
    if 'risk_free_rate' in data:
        if not isinstance(data['risk_free_rate'], (int, float)):
            abort(400, 'Risk-free rate must be numeric')
    
    if 'confidence_level' in data:
        if not isinstance(data['confidence_level'], (int, float)) or not 0 < data['confidence_level'] < 1:
            abort(400, 'Confidence level must be a float between 0 and 1')
    
    if 'num_simulations' in data:
        if not isinstance(data['num_simulations'], int) or data['num_simulations'] <= 0:
            abort(400, 'Number of simulations must be a positive integer')
    
    if 'time_horizon' in data:
        if not isinstance(data['time_horizon'], int) or data['time_horizon'] <= 0:
            abort(400, 'Time horizon must be a positive integer')

def validate_optimization_constraints(data):
    """
    Validate the input data for portfolio optimization with constraints.
    """
    if 'min_weights' not in data or 'max_weights' not in data:
        abort(400, 'Both min_weights and max_weights are required')
    
    min_weights = data['min_weights']
    max_weights = data['max_weights']
    
    if not isinstance(min_weights, dict) or not isinstance(max_weights, dict):
        abort(400, 'min_weights and max_weights must be dictionaries')
    
    if set(min_weights.keys()) != set(max_weights.keys()):
        abort(400, 'min_weights and max_weights must have the same keys')
    
    for asset in min_weights:
        if not isinstance(min_weights[asset], (int, float)) or not isinstance(max_weights[asset], (int, float)):
            abort(400, 'All weight values must be numeric')
        if min_weights[asset] < 0 or max_weights[asset] > 1:
            abort(400, 'Weight constraints must be between 0 and 1')
        if min_weights[asset] > max_weights[asset]:
            abort(400, f'Minimum weight cannot be greater than maximum weight for asset {asset}')
    
    if sum(min_weights.values()) > 1 or sum(max_weights.values()) < 1:
        abort(400, 'Weight constraints are infeasible')