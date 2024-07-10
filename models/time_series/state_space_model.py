import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Tuple, List, Optional
from .kalman_filter import KalmanFilter

class StateSpaceModel:
    def __init__(self, dim_state: int, dim_obs: int):
        self.dim_state = dim_state
        self.dim_obs = dim_obs
        self.kf = KalmanFilter(dim_state, dim_obs)
        self.params = None

    def set_params(self, params: np.ndarray):
        """
        Set the parameters of the state space model.
        
        :param params: Array of model parameters
        """
        self.params = params
        self._update_matrices()

    def _update_matrices(self):
        """
        Update the state space matrices based on the current parameters.
        This method should be implemented in subclasses.
        """
        raise NotImplementedError("This method should be implemented in subclasses")

    def likelihood(self, y: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the observations given the model.
        
        :param y: Array of observations
        :return: Log-likelihood value
        """
        return self.kf.likelihood(y)

    def filter(self, y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Apply the Kalman filter to the observations.
        
        :param y: Array of observations
        :return: Tuple of lists containing filtered states and state covariances
        """
        return self.kf.filter(y)

    def smooth(self, filtered_states: List[np.ndarray], filtered_covariances: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Apply the RTS smoother to the filtered estimates.
        
        :param filtered_states: List of filtered state estimates
        :param filtered_covariances: List of filtered state covariances
        :return: Tuple of lists containing smoothed states and state covariances
        """
        return self.kf.smooth(filtered_states, filtered_covariances)

    def simulate(self, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the state space model for a given number of steps.
        
        :param n_steps: Number of steps to simulate
        :return: Tuple of simulated states and observations
        """
        return self.kf.simulate(n_steps)

    def forecast(self, y: np.ndarray, n_ahead: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forecast future observations based on the current state.
        
        :param y: Array of past observations
        :param n_ahead: Number of steps to forecast
        :return: Tuple of forecasted means and standard deviations
        """
        filtered_states, _ = self.filter(y)
        x = filtered_states[-1]
        P = self.kf.P
        
        forecasts = []
        forecast_covs = []
        
        for _ in range(n_ahead):
            x, P = self.kf.predict()
            forecasts.append(np.dot(self.kf.H, x))
            forecast_covs.append(np.dot(np.dot(self.kf.H, P), self.kf.H.T) + self.kf.R)
        
        return np.array(forecasts), np.sqrt(np.array([np.diag(cov) for cov in forecast_covs]))

    def fit(self, y: np.ndarray, method: str = 'bfgs', **kwargs) -> dict:
        """
        Fit the state space model to the data using maximum likelihood estimation.
        
        :param y: Array of observations
        :param method: Optimization method ('bfgs' or 'nelder-mead')
        :param kwargs: Additional arguments to pass to the optimizer
        :return: Dictionary containing optimization results
        """
        from scipy.optimize import minimize
        
        def neg_log_likelihood(params):
            self.set_params(params)
            return -self.likelihood(y)
        
        if self.params is None:
            raise ValueError("Initial parameters must be set before fitting")
        
        result = minimize(neg_log_likelihood, self.params, method=method, **kwargs)
        self.set_params(result.x)
        
        return {
            'success': result.success,
            'message': result.message,
            'params': self.params,
            'log_likelihood': -result.fun
        }

    def aic(self, y: np.ndarray) -> float:
        """
        Calculate the Akaike Information Criterion (AIC) for the model.
        
        :param y: Array of observations
        :return: AIC value
        """
        log_likelihood = self.likelihood(y)
        k = len(self.params)
        n = len(y)
        return 2 * k - 2 * log_likelihood + 2 * k * (k + 1) / (n - k - 1)

    def bic(self, y: np.ndarray) -> float:
        """
        Calculate the Bayesian Information Criterion (BIC) for the model.
        
        :param y: Array of observations
        :return: BIC value
        """
        log_likelihood = self.likelihood(y)
        k = len(self.params)
        n = len(y)
        return k * np.log(n) - 2 * log_likelihood

    def residuals(self, y: np.ndarray) -> np.ndarray:
        """
        Calculate the residuals of the model.
        
        :param y: Array of observations
        :return: Array of residuals
        """
        filtered_states, _ = self.filter(y)
        return y - np.array([np.dot(self.kf.H, state) for state in filtered_states])

    def state_cov(self, y: np.ndarray) -> List[np.ndarray]:
        """
        Calculate the state covariance matrices.
        
        :param y: Array of observations
        :return: List of state covariance matrices
        """
        _, filtered_covariances = self.filter(y)
        return filtered_covariances

class LocalLevelModel(StateSpaceModel):
    def __init__(self):
        super().__init__(dim_state=1, dim_obs=1)
        self.params = np.array([0.1, 0.1])  # Initial values for sigma_eps and sigma_eta

    def _update_matrices(self):
        sigma_eps, sigma_eta = self.params
        self.kf.set_state_transition(np.array([[1.0]]))
        self.kf.set_observation_model(np.array([[1.0]]))
        self.kf.set_process_noise(np.array([[sigma_eta**2]]))
        self.kf.set_observation_noise(np.array([[sigma_eps**2]]))

    def get_signal_to_noise_ratio(self) -> float:
        """
        Calculate the signal-to-noise ratio of the model.
        
        :return: Signal-to-noise ratio
        """
        sigma_eps, sigma_eta = self.params
        return sigma_eta**2 / sigma_eps**2

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate some sample data
    np.random.seed(42)
    n = 200
    true_level = np.cumsum(np.random.normal(0, 0.1, n))
    y = true_level + np.random.normal(0, 0.5, n)

    # Create and fit the Local Level Model
    model = LocalLevelModel()
    model.set_params(np.array([0.5, 0.1]))  # Initial parameter guess
    fit_result = model.fit(y)
    print("Fit result:", fit_result)

    # Filter and smooth the data
    filtered_states, filtered_covs = model.filter(y)
    smoothed_states, smoothed_covs = model.smooth(filtered_states, filtered_covs)

    # Forecast future values
    forecast_mean, forecast_std = model.forecast(y, n_ahead=20)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(y, label='Observations', alpha=0.7)
    plt.plot(true_level, label='True Level', alpha=0.7)
    plt.plot([state[0] for state in filtered_states], label='Filtered', alpha=0.7)
    plt.plot([state[0] for state in smoothed_states], label='Smoothed', alpha=0.7)

    forecast_x = np.arange(n, n + 20)
    plt.plot(forecast_x, forecast_mean, label='Forecast', color='red')
    plt.fill_between(forecast_x, forecast_mean - 2*forecast_std, forecast_mean + 2*forecast_std, color='red', alpha=0.2)

    plt.legend()
    plt.title('Local Level Model: Filtering, Smoothing, and Forecasting')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

    # Calculate and print model diagnostics
    print("\nModel Diagnostics:")
    print(f"AIC: {model.aic(y):.2f}")
    print(f"BIC: {model.bic(y):.2f}")
    print(f"Signal-to-Noise Ratio: {model.get_signal_to_noise_ratio():.4f}")

    # Plot residuals
    residuals = model.residuals(y)
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(residuals)
    plt.title('Model Residuals')
    plt.xlabel('Time')
    plt.ylabel('Residual')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.hist(residuals, bins=20, density=True, alpha=0.7)
    plt.title('Residual Distribution')
    plt.xlabel('Residual')
    plt.ylabel('Density')
    plt.grid(True)

    plt.tight_layout()
    plt.show()