from typing import List
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()

    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the data.
        
        :param data: Input DataFrame
        :return: DataFrame with missing values handled
        """
        return data.fillna(method='ffill').fillna(method='bfill')

    def remove_outliers(self, data: pd.DataFrame, threshold: float = 3) -> pd.DataFrame:
        """
        Remove outliers from the data using z-score method.
        
        :param data: Input DataFrame
        :param threshold: Z-score threshold for outlier detection
        :return: DataFrame with outliers removed
        """
        z_scores = np.abs((data - data.mean()) / data.std())
        return data[(z_scores < threshold).all(axis=1)]

    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the data using StandardScaler.
        
        :param data: Input DataFrame
        :return: Normalized DataFrame
        """
        scaled_data = self.scaler.fit_transform(data)
        return pd.DataFrame(scaled_data, columns=data.columns, index=data.index)

    def calculate_returns(self, data: pd.DataFrame, method: str = 'simple') -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        :param data: Input DataFrame with price data
        :param method: Method for calculating returns ('simple' or 'log')
        :return: DataFrame with calculated returns
        """
        if method == 'simple':
            return data.pct_change()
        elif method == 'log':
            return np.log(data / data.shift(1))
        else:
            raise ValueError("method must be either 'simple' or 'log'")

    def winsorize_data(self, data: pd.DataFrame, limits: tuple = (0.05, 0.05)) -> pd.DataFrame:
        """
        Winsorize the data to reduce the effect of outliers.
        
        :param data: Input DataFrame
        :param limits: Tuple of lower and upper percentiles to winsorize
        :return: Winsorized DataFrame
        """
        return data.clip(lower=data.quantile(limits[0]), upper=data.quantile(1 - limits[1]), axis=1)

    def align_dates(self, *dataframes: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Align multiple DataFrames to have the same date range.
        
        :param dataframes: DataFrames to align
        :return: List of aligned DataFrames
        """
        common_dates = dataframes[0].index
        for df in dataframes[1:]:
            common_dates = common_dates.intersection(df.index)
        
        return [df.loc[common_dates] for df in dataframes]

    def resample_data(self, data: pd.DataFrame, frequency: str = 'D') -> pd.DataFrame:
        """
        Resample the data to a specified frequency.
        
        :param data: Input DataFrame
        :param frequency: Desired frequency ('D' for daily, 'W' for weekly, 'M' for monthly)
        :return: Resampled DataFrame
        """
        return data.resample(frequency).last()