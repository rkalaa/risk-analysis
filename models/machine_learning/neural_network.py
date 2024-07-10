import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict

class NeuralNetworkModel:
    def __init__(self, hidden_layer_sizes: Tuple = (100,), activation: str = 'relu', 
                 solver: str = 'adam', random_state: int = 42):
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, 
                                  solver=solver, random_state=random_state)
        self.scaler = StandardScaler()

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple[float, float]:
        """
        Train the Neural Network model.
        
        :param X: Feature DataFrame
        :param y: Target Series
        :param test_size: Proportion of the dataset to include in the test split
        :return: Tuple of RMSE and R-squared scores
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        return rmse, r2

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        :param X: Feature DataFrame
        :return: Array of predictions
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series, param_grid: Dict) -> Dict:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        :param X: Feature DataFrame
        :param y: Target Series
        :param param_grid: Dictionary with parameters names (str) as keys and lists of parameter settings to try as values
        :return: Dictionary with best parameters
        """
        X_scaled = self.scaler.fit_transform(X)
        
        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X_scaled, y)
        
        self.model = grid_search.best_estimator_
        return grid_search.best_params_

    def get_model_summary(self) -> Dict:
        """
        Get a summary of the model.
        
        :return: Dictionary containing model information
        """
        return {
            "model_type": "Neural Network",
            "hidden_layer_sizes": self.model.hidden_layer_sizes,
            "activation": self.model.activation,
            "solver": self.model.solver,
            "n_layers": self.model.n_layers_,
            "n_outputs": self.model.n_outputs_
        }