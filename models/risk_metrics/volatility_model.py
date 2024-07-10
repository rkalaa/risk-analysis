import numpy as np
import pandas as pd
from scipy import stats
from arch import arch_model
from statsmodels.tsa.api import SimpleExpSmoothing
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt

class VolatilityModel:
    def __init__(self):
        self.returns = None
        self.volatility = None

    def set_returns(self, returns: pd.Series):
        """
        Set the returns series for volatility calculation.
        
        :param returns: Series of asset returns
        """
        self.returns = returns

    def calculate_historical_volatility(self, window: int = 252, trading_periods: int = 252) -> pd.Series:
        """
        Calculate historical volatility using a rolling window.
        
        :param window: Size of the rolling window
        :param trading_periods: Number of trading periods in a year
        :return: Series of annualized volatility estimates
        """
        if self.returns is None:
            raise ValueError("Returns have not been set. Call set_returns() first.")
        
        self.volatility = self.returns.rolling(window=window).std() * np.sqrt(trading_periods)
        return self.volatility

    def calculate_ewma_volatility(self, lambda_param: float = 0.94, trading_periods: int = 252) -> pd.Series:
        """
        Calculate volatility using Exponentially Weighted Moving Average (EWMA).
        
        :param lambda_param: Decay factor for EWMA
        :param trading_periods: Number of trading periods in a year
        :return: Series of annualized EWMA volatility estimates
        """
        if self.returns is None:
            raise ValueError("Returns have not been set. Call set_returns() first.")
        
        ewma_model = SimpleExpSmoothing(self.returns**2)
        ewma_fit = ewma_model.fit(smoothing_level=1-lambda_param, optimized=False)
        self.volatility = np.sqrt(ewma_fit.fittedvalues) * np.sqrt(trading_periods)
        return self.volatility

    def calculate_garch_volatility(self, p: int = 1, q: int = 1, mean: str = 'Zero', vol: str = 'GARCH', dist: str = 'Normal') -> pd.Series:
        """
        Calculate volatility using a GARCH model.
        
        :param p: The order of the GARCH terms
        :param q: The order of the ARCH terms
        :param mean: The model for the mean equation
        :param vol: The model for the volatility equation
        :param dist: The distribution for the error term
        :return: Series of annualized GARCH volatility estimates
        """
        if self.returns is None:
            raise ValueError("Returns have not been set. Call set_returns() first.")
        
        model = arch_model(self.returns, p=p, q=q, mean=mean, vol=vol, dist=dist)
        results = model.fit(disp='off')
        self.volatility = pd.Series(np.sqrt(results.conditional_volatility), index=self.returns.index)
        return self.volatility

    def calculate_parkinson_volatility(self, high: pd.Series, low: pd.Series, window: int = 252, trading_periods: int = 252) -> pd.Series:
        """
        Calculate Parkinson volatility using high and low prices.
        
        :param high: Series of high prices
        :param low: Series of low prices
        :param window: Rolling window size
        :param trading_periods: Number of trading periods in a year
        :return: Series of annualized Parkinson volatility estimates
        """
        log_hl = np.log(high / low)
        self.volatility = np.sqrt((1 / (4 * np.log(2))) * log_hl**2).rolling(window=window).mean() * np.sqrt(trading_periods)
        return self.volatility

    def calculate_garman_klass_volatility(self, open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 252, trading_periods: int = 252) -> pd.Series:
        """
        Calculate Garman-Klass volatility using open, high, low, and close prices.
        
        :param open: Series of opening prices
        :param high: Series of high prices
        :param low: Series of low prices
        :param close: Series of closing prices
        :param window: Rolling window size
        :param trading_periods: Number of trading periods in a year
        :return: Series of annualized Garman-Klass volatility estimates
        """
        log_hl = np.log(high / low)
        log_co = np.log(close / open)
        
        volatility = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
        self.volatility = np.sqrt(volatility.rolling(window=window).mean()) * np.sqrt(trading_periods)
        return self.volatility

    def calculate_yang_zhang_volatility(self, open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 252, trading_periods: int = 252) -> pd.Series:
        """
        Calculate Yang-Zhang volatility using open, high, low, and close prices.
        
        :param open: Series of opening prices
        :param high: Series of high prices
        :param low: Series of low prices
        :param close: Series of closing prices
        :param window: Rolling window size
        :param trading_periods: Number of trading periods in a year
        :return: Series of annualized Yang-Zhang volatility estimates
        """
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        
        log_ho = np.log(high / open)
        log_lo = np.log(low / open)
        log_co = np.log(close / open)
        
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        
        close_vol = self.returns.rolling(window=window).var()
        open_vol = np.log(open / close.shift(1)).rolling(window=window).var()
        
        self.volatility = np.sqrt(open_vol + k * close_vol + (1 - k) * rs.rolling(window=window).mean()) * np.sqrt(trading_periods)
        return self.volatility

    def forecast_garch_volatility(self, horizon: int = 10, p: int = 1, q: int = 1, mean: str = 'Zero', vol: str = 'GARCH', dist: str = 'Normal') -> pd.Series:
        """
        Forecast volatility using a GARCH model.
        
        :param horizon: Number of periods to forecast
        :param p: The order of the GARCH terms
        :param q: The order of the ARCH terms
        :param mean: The model for the mean equation
        :param vol: The model for the volatility equation
        :param dist: The distribution for the error term
        :return: Series of forecasted volatilities
        """
        if self.returns is None:
            raise ValueError("Returns have not been set. Call set_returns() first.")
        
        model = arch_model(self.returns, p=p, q=q, mean=mean, vol=vol, dist=dist)
        results = model.fit(disp='off')
        forecast = results.forecast(horizon=horizon)
        return pd.Series(np.sqrt(forecast.variance.values[-1]), index=pd.date_range(start=self.returns.index[-1] + pd.Timedelta(days=1), periods=horizon))

    def plot_volatility(self, methods: List[str], **kwargs):
        """
        Plot volatility estimates using different methods.
        
        :param methods: List of volatility calculation methods to plot
        :param kwargs: Additional keyword arguments for specific volatility calculation methods
        """
        if self.returns is None:
            raise ValueError("Returns have not been set. Call set_returns() first.")
        
        plt.figure(figsize=(12, 6))
        
        for method in methods:
            if method == 'historical':
                vol = self.calculate_historical_volatility(**kwargs)
                plt.plot(vol.index, vol, label='Historical Volatility')
            elif method == 'ewma':
                vol = self.calculate_ewma_volatility(**kwargs)
                plt.plot(vol.index, vol, label='EWMA Volatility')
            elif method == 'garch':
                vol = self.calculate_garch_volatility(**kwargs)
                plt.plot(vol.index, vol, label='GARCH Volatility')
            elif method == 'parkinson':
                if 'high' not in kwargs or 'low' not in kwargs:
                    raise ValueError("High and low prices are required for Parkinson volatility")
                vol = self.calculate_parkinson_volatility(**kwargs)
                plt.plot(vol.index, vol, label='Parkinson Volatility')
            elif method == 'garman_klass':
                if 'open' not in kwargs or 'high' not in kwargs or 'low' not in kwargs or 'close' not in kwargs:
                    raise ValueError("Open, high, low, and close prices are required for Garman-Klass volatility")
                vol = self.calculate_garman_klass_volatility(**kwargs)
                plt.plot(vol.index, vol, label='Garman-Klass Volatility')
            elif method == 'yang_zhang':
                if 'open' not in kwargs or 'high' not in kwargs or 'low' not in kwargs or 'close' not in kwargs:
                    raise ValueError("Open, high, low, and close prices are required for Yang-Zhang volatility")
                vol = self.calculate_yang_zhang_volatility(**kwargs)
                plt.plot(vol.index, vol, label='Yang-Zhang Volatility')
            else:
                raise ValueError(f"Unknown volatility calculation method: {method}")
        
        plt.title('Volatility Estimates')
        plt.xlabel('Date')
        plt.ylabel('Annualized Volatility')
        plt.legend()
        plt.grid(True)
        plt.show()