import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.metrics import mean_squared_error, r2_score
from .random_forest import RandomForestModel
from .gradient_boosting import GradientBoostingModel
from .neural_network import NeuralNetworkModel
from .support_vector_machine import SVMModel

class EnsembleModel:
    def __init__(self, models: List[str] = ['rf', 'gb', 'nn', 'svm'], weights: List[float] = None):
        self.models = []
        self.weights = weights if weights else [1/len(models)] * len(models)
        
        for model in models:
            if model == 'rf':
                self.models.append(RandomForestModel())
            elif model == 'gb':
                self.models.append(GradientBoostingModel())
            elif model == 'nn':
                self.models.append(NeuralNetworkModel())
            elif model == 'svm':
                self.models.append(SVMModel())
            else:
                raise ValueError(f"Unsupported model type: {model}")

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple[float, float]:
        """
        Train all models in the ensemble.
        
        :param X: Feature DataFrame
        :param y: Target Series
        :param test_size: Proportion of the dataset to include in the test split
        :return: Tuple of RMSE and R-squared scores for the ensemble
        """
        predictions = []
        
        for model in self.models:
            model.train(X, y, test_size)
            predictions.append(model.predict(X))
        
        ensemble_predictions = np.average(predictions, axis=0, weights=self.weights)
        
        rmse = np.sqrt(mean_squared_error(y, ensemble_predictions))
        r2 = r2_score(y, ensemble_predictions)
        
        return rmse, r2

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the ensemble of models.
        
        :param X: Feature DataFrame
        :return: Array of ensemble predictions
        """
        predictions = []
        
        for model in self.models:
            predictions.append(model.predict(X))
        
        return np.average(predictions, axis=0, weights=self.weights)

    def get_feature_importance(self) -> pd.Series:
        """
        Get aggregated feature importances from models that support it.
        
        :return: Series of aggregated feature importances
        """
        feature_importances = []
        
        for model in self.models:
            if hasattr(model, 'get_feature_importance'):
                feature_importances.append(model.get_feature_importance())
        
        if not feature_importances:
            raise ValueError("None of the models in the ensemble support feature importance.")
        
        aggregated_importance = pd.concat(feature_importances, axis=1).mean(axis=1)
        return aggregated_importance.sort_values(ascending=False)

    def get_model_summary(self) -> Dict:
        """
        Get a summary of the ensemble model.
        
        :return: Dictionary containing ensemble model information
        """
        return {
            "model_type": "Ensemble",
            "models": [model.__class__.__name__ for model in self.models],
            "weights": self.weights
        }

    def set_weights(self, weights: List[float]):
        """
        Set new weights for the ensemble models.
        
        :param weights: List of weights for each model in the ensemble
        """
        if len(weights) != len(self.models):
            raise ValueError("Number of weights must match number of models in the ensemble.")
        if not np.isclose(sum(weights), 1.0):
            raise ValueError("Weights must sum to 1.")
        
        self.weights = weights

    def optimize_weights(self, X: pd.DataFrame, y: pd.Series) -> List[float]:
        """
        Optimize the weights of the ensemble models to minimize RMSE.
        
        :param X: Feature DataFrame
        :param y: Target Series
        :return: Optimized weights for the ensemble
        """
        from scipy.optimize import minimize
        
        predictions = [model.predict(X) for model in self.models]
        
        def rmse_loss(weights):
            weighted_predictions = np.average(predictions, axis=0, weights=weights)
            return np.sqrt(mean_squared_error(y, weighted_predictions))
        
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = [(0, 1)] * len(self.models)
        initial_weights = [1/len(self.models)] * len(self.models)
        
        result = minimize(rmse_loss, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        self.weights = result.x
        return self.weights