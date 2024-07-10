# Import from machine_learning subfolder
from .machine_learning.random_forest import RandomForestModel
from .machine_learning.gradient_boosting import GradientBoostingModel
from .machine_learning.neural_network import NeuralNetworkModel
from .machine_learning.support_vector_machine import SVMModel
from .machine_learning.ensemble_model import EnsembleModel

# Import from risk_metrics subfolder
from .risk_metrics.var_model import VaRModel
from .risk_metrics.cvar_model import CVaRModel
from .risk_metrics.volatility_model import VolatilityModel
from .risk_metrics.beta_model import BetaModel
from .risk_metrics.sharpe_ratio_model import SharpeRatioModel

# Import from time_series subfolder
from .time_series.arima_model import ARIMAModel
from .time_series.garch_model import GARCHModel
from .time_series.var_model import VARModel
from .time_series.kalman_filter import KalmanFilter
from .time_series.state_space_model import StateSpaceModel, LocalLevelModel

# You can group the imports if you want to provide a cleaner namespace
machine_learning_models = {
    'RandomForest': RandomForestModel,
    'GradientBoosting': GradientBoostingModel,
    'NeuralNetwork': NeuralNetworkModel,
    'SVM': SVMModel,
    'Ensemble': EnsembleModel
}

risk_metrics = {
    'VaR': VaRModel,
    'CVaR': CVaRModel,
    'Volatility': VolatilityModel,
    'Beta': BetaModel,
    'SharpeRatio': SharpeRatioModel
}

time_series_models = {
    'ARIMA': ARIMAModel,
    'GARCH': GARCHModel,
    'VAR': VARModel,
    'KalmanFilter': KalmanFilter,
    'StateSpace': StateSpaceModel,
    'LocalLevel': LocalLevelModel
}

__all__ = [
    'RandomForestModel', 'GradientBoostingModel', 'NeuralNetworkModel', 'SVMModel', 'EnsembleModel',
    'VaRModel', 'CVaRModel', 'VolatilityModel', 'BetaModel', 'SharpeRatioModel',
    'ARIMAModel', 'GARCHModel', 'VARModel', 'KalmanFilter', 'StateSpaceModel', 'LocalLevelModel',
    'machine_learning_models', 'risk_metrics', 'time_series_models'
]
