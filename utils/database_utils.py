import pandas as pd
from sqlalchemy import create_engine, text
from typing import List, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DatabaseConnector:
    def __init__(self, db_url: Optional[str] = None):
        if db_url is None:
            db_url = os.getenv('DATABASE_URL')
        if db_url is None:
            raise ValueError("Database URL not provided and not found in environment variables")
        self.engine = create_engine(db_url)

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a SQL query and return the results as a DataFrame.
        
        :param query: SQL query string
        :return: DataFrame containing query results
        """
        with self.engine.connect() as connection:
            result = connection.execute(text(query))
            return pd.DataFrame(result.fetchall(), columns=result.keys())

    def insert_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'replace') -> None:
        """
        Insert a DataFrame into a database table.
        
        :param df: DataFrame to insert
        :param table_name: Name of the target table
        :param if_exists: How to behave if the table already exists ('fail', 'replace', or 'append')
        """
        df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)

    def get_table_names(self) -> List[str]:
        """
        Get a list of all table names in the database.
        
        :return: List of table names
        """
        return self.engine.table_names()

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.
        
        :param table_name: Name of the table to check
        :return: True if the table exists, False otherwise
        """
        return self.engine.has_table(table_name)

    def create_table(self, table_name: str, columns: List[str]) -> None:
        """
        Create a new table in the database.
        
        :param table_name: Name of the table to create
        :param columns: List of column definitions (e.g., ["id INT PRIMARY KEY", "name VARCHAR(255)"])
        """
        column_defs = ", ".join(columns)
        query = f"CREATE TABLE {table_name} ({column_defs})"
        with self.engine.connect() as connection:
            connection.execute(text(query))

    def delete_table(self, table_name: str) -> None:
        """
        Delete a table from the database.
        
        :param table_name: Name of the table to delete
        """
        query = f"DROP TABLE IF EXISTS {table_name}"
        with self.engine.connect() as connection:
            connection.execute(text(query))

    def backup_table(self, table_name: str, backup_table_name: str) -> None:
        """
        Create a backup of a table.
        
        :param table_name: Name of the table to backup
        :param backup_table_name: Name of the backup table
        """
        query = f"CREATE TABLE {backup_table_name} AS SELECT * FROM {table_name}"
        with self.engine.connect() as connection:
            connection.execute(text(query))

    def restore_table(self, backup_table_name: str, target_table_name: str) -> None:
        """
        Restore a table from a backup.
        
        :param backup_table_name: Name of the backup table
        :param target_table_name: Name of the table to restore to
        """
        self.delete_table(target_table_name)
        query = f"CREATE TABLE {target_table_name} AS SELECT * FROM {backup_table_name}"
        with self.engine.connect() as connection:
            connection.execute(text(query))