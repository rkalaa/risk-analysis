import pandas as pd
import yfinance as yf
from typing import List, Optional
from sqlalchemy import create_engine
from config.settings import Config
import numpy as np

class DataLoader:
    def __init__(self):
        self.engine = create_engine(Config.DATABASE_URI)

    def load_from_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        :param file_path: Path to the CSV file
        :return: DataFrame containing the loaded data
        """
        return pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

    def load_from_database(self, table_name: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from the database.
        
        :param table_name: Name of the table to load data from
        :param start_date: Start date for data loading (optional)
        :param end_date: End date for data loading (optional)
        :return: DataFrame containing the loaded data
        """
        query = f"SELECT * FROM {table_name}"
        if start_date and end_date:
            query += f" WHERE Date BETWEEN '{start_date}' AND '{end_date}'"
        elif start_date:
            query += f" WHERE Date >= '{start_date}'"
        elif end_date:
            query += f" WHERE Date <= '{end_date}'"
        
        return pd.read_sql(query, self.engine, parse_dates=['Date'], index_col='Date')

    def load_from_yfinance(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load data from Yahoo Finance.
        
        :param tickers: List of stock tickers
        :param start_date: Start date for data loading
        :param end_date: End date for data loading
        :return: DataFrame containing the loaded data
        """
        data = yf.download(tickers, start=start_date, end=end_date)
        if len(tickers) == 1:
            return data['Adj Close'].to_frame(name=tickers[0])
        else:
            return data['Adj Close']

    def save_to_database(self, data: pd.DataFrame, table_name: str):
        """
        Save data to the database.
        
        :param data: DataFrame containing the data to be saved
        :param table_name: Name of the table to save data to
        """
        data.to_sql(table_name, self.engine, if_exists='replace')

    def load_financial_statements(self, ticker: str) -> dict:
        """
        Load financial statements for a given company.
        
        :param ticker: Stock ticker of the company
        :return: Dictionary containing balance sheet, income statement, and cash flow statement
        """
        company = yf.Ticker(ticker)
        return {
            'balance_sheet': company.balance_sheet,
            'income_statement': company.income_stmt,
            'cash_flow': company.cashflow
        }

    def load_economic_indicators(self, indicators: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load economic indicators data.
        
        :param indicators: List of economic indicator symbols
        :param start_date: Start date for data loading
        :param end_date: End date for data loading
        :return: DataFrame containing the loaded economic indicators data
        """
        # This is a placeholder. In a real-world scenario, you would connect to an economic data API.
        # For this example, we'll create some dummy data.
        date_range = pd.date_range(start=start_date, end=end_date)
        data = pd.DataFrame(index=date_range)
        for indicator in indicators:
            data[indicator] = np.random.randn(len(date_range)).cumsum()
        return data