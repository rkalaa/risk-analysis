import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.stattools import durbin_watson
from typing import Tuple, Dict, List
from sklearn.metrics import mean_squared_error

class VARModel:
    def __init__(self, maxlags: int = 1):
        self.maxlags = maxlags
        self.model = None
        self.results = None

    def fit(self, data: pd.DataFrame):
        """
        Fit the VAR model to the data.
        
        :param data: Multivariate time series data
        """
        self.model = VAR(data)
        self.results = self.model.fit(maxlags=self.maxlags)

    def predict(self, steps: int) -> pd.DataFrame:
        """
        Make predictions using the fitted VAR model.
        
        :param steps: Number of steps to forecast
        :return: DataFrame of forecasted values
        """
        if self.results is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        forecast = self.results.forecast(self.results.y, steps=steps)
        return pd.DataFrame(forecast, columns=self.results.names, index=pd.date_range(start=self.results.index[-1] + pd.Timedelta(days=1), periods=steps))

    def get_residuals(self) -> pd.DataFrame:
        """
        Get the residuals of the fitted model.
        
        :return: DataFrame of residuals
        """
        if self.results is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        return pd.DataFrame(self.results.resid, columns=self.results.names, index=self.results.index)

    def get_aic(self) -> float:
        """
        Get the Akaike Information Criterion (AIC) of the fitted model.
        
        :return: AIC value
        """
        if self.results is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        return self.results.aic

    def get_bic(self) -> float:
        """
        Get the Bayesian Information Criterion (BIC) of the fitted model.
        
        :return: BIC value
        """
        if self.results is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        return self.results.bic

    def granger_causality(self, caused: str, causing: str) -> Dict:
        """
        Perform Granger causality test.
        
        :param caused: Name of the variable being caused
        :param causing: Name of the causing variable
        :return: Dictionary with test results
        """
        if self.results is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        test_result = self.results.test_causality(caused, causing, kind='f')
        return {
            'Test Statistic': test_result.test_statistic,
            'p-value': test_result.pvalue,
            'df': test_result.df
        }

    def impulse_response(self, steps: int) -> pd.DataFrame:
        """
        Calculate impulse response functions.
        
        :param steps: Number of steps for impulse response
        :return: DataFrame of impulse responses
        """
        if self.results is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        irf = self.results.irf(steps)
        return pd.DataFrame(irf.irfs, columns=self.results.names)

    def forecast_error_variance_decomposition(self, steps: int) -> pd.DataFrame:
        """
        Perform forecast error variance decomposition.
        
        :param steps: Number of steps for decomposition
        :return: DataFrame of variance decompositions
        """
        if self.results is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        fevd = self.results.fevd(steps)
        return pd.DataFrame(fevd.fevd, columns=self.results.names)

    def calculate_mse(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Mean Squared Error for out-of-sample predictions.
        
        :param test_data: Out-of-sample multivariate time series data
        :return: Dictionary of MSE values for each variable
        """
        if self.results is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        forecast = self.predict(steps=len(test_data))
        mse = {}
        for column in test_data.columns:
            mse[column] = mean_squared_error(test_data[column], forecast[column])
        return mse

    def select_order(self, maxlags: int) -> Dict[str, int]:
        """
        Select the order of the VAR model based on information criteria.
        
        :param maxlags: Maximum number of lags to consider
        :return: Dictionary with selected orders based on different criteria
        """
        if self.model is None:
            raise ValueError("Model has not been initialized. Call fit() first.")
        
        results = self.model.select_order(maxlags=maxlags)
        return {
            'AIC': results.aic,
            'BIC': results.bic,
            'FPE': results.fpe,
            'HQIC': results.hqic
        }

    def test_whiteness(self, lags: int) -> Dict[str, float]:
        """
        Perform a test for whiteness (no autocorrelation) of residuals.
        
        :param lags: Number of lags to test
        :return: Dictionary with test results
        """
        if self.results is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        whiteness_test = self.results.test_whiteness(lags)
        return {
            'Statistic': whiteness_test.stat,
            'p-value': whiteness_test.pvalue,
            'df': whiteness_test.df
        }

    def test_normality(self) -> Dict[str, float]:
        """
        Perform a test for normality of residuals.
        
        :return: Dictionary with test results
        """
        if self.results is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        normality_test = self.results.test_normality()
        return {
            'Statistic': normality_test.statistic,
            'p-value': normality_test.pvalue
        }

    def test_stability(self) -> bool:
        """
        Test the stability of the VAR model.
        
        :return: True if the model is stable, False otherwise
        """
        if self.results is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        return self.results.is_stable()

    def cointegration_test(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform Johansen cointegration test.
        
        :param data: Multivariate time series data
        :return: DataFrame with test results
        """
        coint_result = coint_johansen(data, det_order=0, k_ar_diff=1)
        trace_stats = pd.DataFrame({
            'Rank': range(len(coint_result.lr1)),
            'Test Statistic': coint_result.lr1,
            'Critical Value (5%)': coint_result.cvt[:, 1]
        })
        return trace_stats

    def durbin_watson_test(self) -> Dict[str, float]:
        """
        Perform Durbin-Watson test for autocorrelation in residuals.
        
        :return: Dictionary with Durbin-Watson statistics for each variable
        """
        if self.results is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        residuals = self.get_residuals()
        dw_stats = {}
        for column in residuals.columns:
            dw_stats[column] = durbin_watson(residuals[column])
        return dw_stats

    def get_summary(self) -> str:
        """
        Get a summary of the fitted VAR model.
        
        :return: String containing the model summary
        """
        if self.results is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        return self.results.summary().as_text()

    def get_coefficients(self) -> Dict[str, pd.DataFrame]:
        """
        Get the coefficients of the fitted VAR model.
        
        :return: Dictionary with coefficient DataFrames for each variable
        """
        if self.results is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        return {name: pd.DataFrame(self.results.coefs[i], 
                                   columns=self.results.names, 
                                   index=[f'L{i+1}' for i in range(self.results.k_ar)])
                for i, name in enumerate(self.results.names)}

    def get_confidence_intervals(self, alpha: float = 0.05) -> Dict[str, pd.DataFrame]:
        """
        Get confidence intervals for the model coefficients.
        
        :param alpha: Significance level (default is 0.05 for 95% confidence interval)
        :return: Dictionary with confidence interval DataFrames for each variable
        """
        if self.results is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        return {name: pd.DataFrame(self.results.conf_int(alpha=alpha)[i], 
                                   columns=['lower', 'upper'], 
                                   index=pd.MultiIndex.from_product([self.results.names, [f'L{i+1}' for i in range(self.results.k_ar)]]))
                for i, name in enumerate(self.results.names)}

# Example usage
if __name__ == "__main__":
    # Generate some example data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
    data = pd.DataFrame({
        'A': np.random.randn(len(dates)).cumsum(),
        'B': np.random.randn(len(dates)).cumsum(),
        'C': np.random.randn(len(dates)).cumsum()
    }, index=dates)

    # Initialize and fit the VAR model
    var_model = VARModel(maxlags=5)
    var_model.fit(data)

    # Make predictions
    forecast = var_model.predict(steps=30)
    print("Forecast:\n", forecast)

    # Perform some analysis
    print("\nGranger Causality (A causes B):")
    print(var_model.granger_causality('B', 'A'))

    print("\nImpulse Response (first 5 steps):")
    print(var_model.impulse_response(steps=5))

    print("\nModel Summary:")
    print(var_model.get_summary())

    print("\nModel Stability:")
    print(var_model.test_stability())

    print("\nCointegration Test:")
    print(var_model.cointegration_test(data))

    print("\nDurbin-Watson Test:")
    print(var_model.durbin_watson_test())