import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict

class GradientBoostingModel:
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3, random_state: int = 42):
        self.model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, 
                                               max_depth=max_depth, random_state=random_state)
        self.feature_importance = None
        self.imputer = SimpleImputer(strategy='mean')
        self.valid_features = None

    def preprocess_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data by replacing infinity with NaN, removing features with all missing values,
        and then imputing remaining missing values.
        
        :param X: Feature DataFrame
        :return: Preprocessed DataFrame
        """
        # Replace infinity with NaN
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        
        # Remove features with all missing values
        self.valid_features = X_clean.columns[X_clean.notna().any()].tolist()
        X_clean = X_clean[self.valid_features]
        
        # Impute NaN values
        X_imputed = pd.DataFrame(self.imputer.fit_transform(X_clean), columns=X_clean.columns, index=X_clean.index)
        
        return X_imputed

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple[float, float]:
        """
        Train the Gradient Boosting model.
        
        :param X: Feature DataFrame
        :param y: Target Series
        :param test_size: Proportion of the dataset to include in the test split
        :return: Tuple of RMSE and R-squared scores
        """
        X = self.preprocess_data(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        self.feature_importance = pd.Series(self.model.feature_importances_, index=X.columns).sort_values(ascending=False)
        
        return rmse, r2

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        :param X: Feature DataFrame
        :return: Array of predictions
        """
        if self.valid_features is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        X = X[self.valid_features]  # Select only valid features
        X = self.preprocess_data(X)
        return self.model.predict(X)

    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importances.
        
        :return: Series of feature importances
        """
        if self.feature_importance is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        return self.feature_importance

    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series, param_grid: Dict) -> Dict:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        :param X: Feature DataFrame
        :param y: Target Series
        :param param_grid: Dictionary with parameters names (str) as keys and lists of parameter settings to try as values
        :return: Dictionary with best parameters
        """
        X = self.preprocess_data(X)
        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X, y)
        
        self.model = grid_search.best_estimator_
        return grid_search.best_params_

    def get_model_summary(self) -> Dict:
        """
        Get a summary of the model.
        
        :return: Dictionary containing model information
        """
        return {
            "model_type": "Gradient Boosting",
            "n_estimators": self.model.n_estimators,
            "learning_rate": self.model.learning_rate,
            "max_depth": self.model.max_depth,
            "feature_importance": self.get_feature_importance().to_dict(),
            "valid_features": self.valid_features
        }