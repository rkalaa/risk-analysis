import pandas as pd
import numpy as np
from typing import List
from ta import add_all_ta_features
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator

class FeatureEngineer:
    def __init__(self):
        pass

    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the data.
        
        :param data: Input DataFrame with OHLCV data
        :return: DataFrame with added technical indicators
        """
        return add_all_ta_features(
            data, open="Open", high="High", low="Low", close="Close", volume="Volume"
        )

    def add_rolling_statistics(self, data: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """
        Add rolling mean and standard deviation for specified windows.
        
        :param data: Input DataFrame
        :param windows: List of rolling window sizes
        :return: DataFrame with added rolling statistics
        """
        for window in windows:
            data[f'rolling_mean_{window}'] = data['Close'].rolling(window=window).mean()
            data[f'rolling_std_{window}'] = data['Close'].rolling(window=window).std()
        return data

    def add_lagged_features(self, data: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
        """
        Add lagged features to the data.
        
        :param data: Input DataFrame
        :param lags: List of lag periods
        :return: DataFrame with added lagged features
        """
        for lag in lags:
            data[f'lag_{lag}'] = data['Close'].shift(lag)
        return data

    def add_price_momentum(self, data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """
        Add price momentum features.
        
        :param data: Input DataFrame
        :param periods: List of periods for momentum calculation
        :return: DataFrame with added momentum features
        """
        for period in periods:
            data[f'momentum_{period}'] = data['Close'] / data['Close'].shift(period) - 1
        return data

    def add_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility indicators to the data.
        
        :param data: Input DataFrame with OHLCV data
        :return: DataFrame with added volatility indicators
        """
        bb_indicator = BollingerBands(data['Close'])
        data['bb_high'] = bb_indicator.bollinger_hband()
        data['bb_low'] = bb_indicator.bollinger_lband()
        data['bb_width'] = (data['bb_high'] - data['bb_low']) / data['Close']
        
        return data

    def add_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend indicators to the data.
        
        :param data: Input DataFrame with OHLCV data
        :return: DataFrame with added trend indicators
        """
        macd = MACD(data['Close'])
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        data['macd_diff'] = macd.macd_diff()
        
        ema_indicator = EMAIndicator(data['Close'])
        data['ema'] = ema_indicator.ema_indicator()
        
        return data

    def add_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum indicators to the data.
        
        :param data: Input DataFrame with OHLCV data
        :return: DataFrame with added momentum indicators
        """
        rsi_indicator = RSIIndicator(data['Close'])
        data['rsi'] = rsi_indicator.rsi()
        
        return data

    def add_date_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add date-based features to the data.
        
        :param data: Input DataFrame with DatetimeIndex
        :return: DataFrame with added date features
        """
        data['day_of_week'] = data.index.dayofweek
        data['month'] = data.index.month
        data['quarter'] = data.index.quarter
        data['year'] = data.index.year
        data['is_month_end'] = data.index.is_month_end.astype(int)
        data['is_quarter_end'] = data.index.is_quarter_end.astype(int)
        data['is_year_end'] = data.index.is_year_end.astype(int)
        
        return data

    def add_fundamental_ratios(self, data: pd.DataFrame, financial_statements: dict) -> pd.DataFrame:
        """
        Add fundamental financial ratios to the data.
        
        :param data: Input DataFrame with stock price data
        :param financial_statements: Dictionary containing financial statements
        :return: DataFrame with added fundamental ratios
        """
        balance_sheet = financial_statements['balance_sheet']
        income_statement = financial_statements['income_statement']
        
        # Calculate P/E ratio
        data['pe_ratio'] = data['Close'] / (income_statement.loc['Net Income'] / balance_sheet.loc['Shares Outstanding'])
        
        # Calculate P/B ratio
        data['pb_ratio'] = data['Close'] / (balance_sheet.loc['Total Stockholder Equity'] / balance_sheet.loc['Shares Outstanding'])
        
        # Calculate Debt-to-Equity ratio
        data['debt_to_equity'] = balance_sheet.loc['Total Liabilities'] / balance_sheet.loc['Total Stockholder Equity']
        
        # Calculate Return on Equity (ROE)
        data['roe'] = income_statement.loc['Net Income'] / balance_sheet.loc['Total Stockholder Equity']
        
        return data