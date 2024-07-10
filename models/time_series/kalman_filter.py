import numpy as np
import pandas as pd
from typing import Tuple, List

class KalmanFilter:
    def __init__(self, dim_state: int, dim_obs: int):
        self.dim_state = dim_state
        self.dim_obs = dim_obs
        self.F = np.eye(dim_state)  # State transition matrix
        self.H = np.eye(dim_obs, dim_state)  # Observation matrix
        self.Q = np.eye(dim_state)  # Process noise covariance
        self.R = np.eye(dim_obs)  # Observation noise covariance
        self.P = np.eye(dim_state)  # Initial state covariance
        self.x = np.zeros(dim_state)  # Initial state estimate

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform the predict step of the Kalman filter.
        
        :return: Tuple of predicted state and predicted state covariance
        """
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x, self.P

    def update(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform the update step of the Kalman filter.
        
        :param z: Observation vector
        :return: Tuple of updated state and updated state covariance
        """
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
        return self.x, self.P

    def filter(self, measurements: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Apply the Kalman filter to a series of measurements.
        
        :param measurements: List of observation vectors
        :return: Tuple of lists containing filtered states and state covariances
        """
        filtered_states = []
        filtered_covariances = []
        
        for z in measurements:
            self.predict()
            x, P = self.update(z)
            filtered_states.append(x)
            filtered_covariances.append(P)
        
        return filtered_states, filtered_covariances

    def smooth(self, filtered_states: List[np.ndarray], filtered_covariances: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Apply the Rauch-Tung-Striebel (RTS) smoother to the filtered estimates.
        
        :param filtered_states: List of filtered state estimates
        :param filtered_covariances: List of filtered state covariances
        :return: Tuple of lists containing smoothed states and state covariances
        """
        n = len(filtered_states)
        smoothed_states = [filtered_states[-1]]
        smoothed_covariances = [filtered_covariances[-1]]
        
        for t in range(n - 2, -1, -1):
            P_pred = np.dot(np.dot(self.F, filtered_covariances[t]), self.F.T) + self.Q
            J = np.dot(np.dot(filtered_covariances[t], self.F.T), np.linalg.inv(P_pred))
            x_smooth = filtered_states[t] + np.dot(J, smoothed_states[0] - np.dot(self.F, filtered_states[t]))
            P_smooth = filtered_covariances[t] + np.dot(np.dot(J, smoothed_covariances[0] - P_pred), J.T)
            
            smoothed_states.insert(0, x_smooth)
            smoothed_covariances.insert(0, P_smooth)
        
        return smoothed_states, smoothed_covariances

    def set_state_transition(self, F: np.ndarray):
        """Set the state transition matrix."""
        if F.shape != (self.dim_state, self.dim_state):
            raise ValueError("Invalid shape for state transition matrix")
        self.F = F

    def set_observation_model(self, H: np.ndarray):
        """Set the observation model matrix."""
        if H.shape != (self.dim_obs, self.dim_state):
            raise ValueError("Invalid shape for observation model matrix")
        self.H = H

    def set_process_noise(self, Q: np.ndarray):
        """Set the process noise covariance matrix."""
        if Q.shape != (self.dim_state, self.dim_state):
            raise ValueError("Invalid shape for process noise covariance matrix")
        self.Q = Q

    def set_observation_noise(self, R: np.ndarray):
        """Set the observation noise covariance matrix."""
        if R.shape != (self.dim_obs, self.dim_obs):
            raise ValueError("Invalid shape for observation noise covariance matrix")
        self.R = R

    def set_initial_state(self, x0: np.ndarray, P0: np.ndarray):
        """Set the initial state estimate and covariance."""
        if x0.shape != (self.dim_state,):
            raise ValueError("Invalid shape for initial state estimate")
        if P0.shape != (self.dim_state, self.dim_state):
            raise ValueError("Invalid shape for initial state covariance")
        self.x = x0
        self.P = P0

    def likelihood(self, measurements: List[np.ndarray]) -> float:
        """
        Calculate the log-likelihood of the measurements given the model.
        
        :param measurements: List of observation vectors
        :return: Log-likelihood value
        """
        log_likelihood = 0
        for z in measurements:
            x_pred, P_pred = self.predict()
            y = z - np.dot(self.H, x_pred)
            S = np.dot(np.dot(self.H, P_pred), self.H.T) + self.R
            log_likelihood += -0.5 * (np.log(np.linalg.det(S)) + np.dot(np.dot(y.T, np.linalg.inv(S)), y))
            self.update(z)
        return log_likelihood

    def simulate(self, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the system for a given number of steps.
        
        :param n_steps: Number of steps to simulate
        :return: Tuple of simulated states and observations
        """
        states = np.zeros((n_steps, self.dim_state))
        observations = np.zeros((n_steps, self.dim_obs))
        
        x = self.x
        for t in range(n_steps):
            x = np.dot(self.F, x) + np.random.multivariate_normal(np.zeros(self.dim_state), self.Q)
            z = np.dot(self.H, x) + np.random.multivariate_normal(np.zeros(self.dim_obs), self.R)
            states[t] = x
            observations[t] = z
        
        return states, observations

# Example usage
if __name__ == "__main__":
    # Create a simple 2D tracking problem
    kf = KalmanFilter(dim_state=4, dim_obs=2)
    
    dt = 1.0  # Time step
    
    # Set up the state transition matrix (constant velocity model)
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    kf.set_state_transition(F)
    
    # Set up the observation model (we only observe position, not velocity)
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    kf.set_observation_model(H)
    
    # Set the process noise (assume small random accelerations)
    q = 0.01
    Q = np.array([[q*dt**4/4, 0, q*dt**3/2, 0],
                  [0, q*dt**4/4, 0, q*dt**3/2],
                  [q*dt**3/2, 0, q*dt**2, 0],
                  [0, q*dt**3/2, 0, q*dt**2]])
    kf.set_process_noise(Q)
    
    # Set the measurement noise
    r = 0.1
    R = r * np.eye(2)
    kf.set_observation_noise(R)
    
    # Generate some noisy measurements
    true_states, measurements = kf.simulate(n_steps=100)
    
    # Apply the Kalman filter
    filtered_states, filtered_covariances = kf.filter(measurements)
    
    # Apply the RTS smoother
    smoothed_states, smoothed_covariances = kf.smooth(filtered_states, filtered_covariances)
    
    # Plot the results
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(true_states[:, 0], true_states[:, 1], label='True')
    plt.plot([m[0] for m in measurements], [m[1] for m in measurements], 'x', label='Measured')
    plt.plot([s[0] for s in filtered_states], [s[1] for s in filtered_states], label='Filtered')
    plt.plot([s[0] for s in smoothed_states], [s[1] for s in smoothed_states], label='Smoothed')
    plt.legend()
    plt.title('2D Tracking with Kalman Filter')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.grid(True)
    plt.show()