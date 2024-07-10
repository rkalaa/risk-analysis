import pyspark
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, lag, stddev, udf
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from typing import List, Dict
import os

class SparkConnector:
    def __init__(self, app_name: str = "FinancialRiskAssessment"):
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .getOrCreate()

    def read_csv(self, file_path: str, schema: StructType = None) -> 'pyspark.sql.DataFrame':
        """
        Read a CSV file into a Spark DataFrame.
        
        :param file_path: Path to the CSV file
        :param schema: Optional schema for the DataFrame
        :return: Spark DataFrame
        """
        if schema:
            return self.spark.read.csv(file_path, header=True, schema=schema)
        else:
            return self.spark.read.csv(file_path, header=True, inferSchema=True)

    def write_csv(self, df: 'pyspark.sql.DataFrame', output_path: str) -> None:
        """
        Write a Spark DataFrame to CSV files.
        
        :param df: Spark DataFrame to write
        :param output_path: Output directory for CSV files
        """
        df.write.csv(output_path, header=True, mode="overwrite")

    def apply_sql(self, df: 'pyspark.sql.DataFrame', query: str) -> 'pyspark.sql.DataFrame':
        """
        Apply a SQL query to a Spark DataFrame.
        
        :param df: Input Spark DataFrame
        :param query: SQL query to apply
        :return: Resulting Spark DataFrame
        """
        df.createOrReplaceTempView("temp_table")
        return self.spark.sql(query)

    def calculate_returns(self, df: 'pyspark.sql.DataFrame', price_col: str, date_col: str) -> 'pyspark.sql.DataFrame':
        """
        Calculate returns for a given price column.
        
        :param df: Input Spark DataFrame
        :param price_col: Name of the price column
        :param date_col: Name of the date column
        :return: Spark DataFrame with an additional 'returns' column
        """
        window = Window.orderBy(date_col)
        return df.withColumn('returns', (col(price_col) / lag(col(price_col), 1).over(window)) - 1)

    def calculate_volatility(self, df: 'pyspark.sql.DataFrame', returns_col: str, window: int = 30) -> 'pyspark.sql.DataFrame':
        """
        Calculate rolling volatility.
        
        :param df: Input Spark DataFrame
        :param returns_col: Name of the returns column
        :param window: Rolling window size
        :return: Spark DataFrame with an additional 'volatility' column
        """
        return df.withColumn('volatility', stddev(col(returns_col)).over(Window.orderBy('date').rowsBetween(-window, 0)))

    def train_linear_regression(self, df: 'pyspark.sql.DataFrame', feature_cols: List[str], label_col: str) -> Dict:
        """
        Train a linear regression model.
        
        :param df: Input Spark DataFrame
        :param feature_cols: List of feature column names
        :param label_col: Name of the label column
        :return: Dictionary containing the model and its performance metrics
        """
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        vec_df = assembler.transform(df)
        
        lr = LinearRegression(featuresCol="features", labelCol=label_col)
        model = lr.fit(vec_df)
        
        predictions = model.transform(vec_df)
        evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")
        rmse = evaluator.evaluate(predictions)
        
        return {
            "model": model,
            "rmse": rmse,
            "coefficients": model.coefficients,
            "intercept": model.intercept
        }

    def apply_udf(self, df: 'pyspark.sql.DataFrame', func, input_col: str, output_col: str, return_type) -> 'pyspark.sql.DataFrame':
        """
        Apply a user-defined function (UDF) to a Spark DataFrame.
        
        :param df: Input Spark DataFrame
        :param func: Python function to apply
        :param input_col: Name of the input column
        :param output_col: Name of the output column
        :param return_type: Spark SQL data type of the return value
        :return: Spark DataFrame with the UDF applied
        """
        spark_udf = udf(func, return_type)
        return df.withColumn(output_col, spark_udf(col(input_col)))

    def group_by_agg(self, df: 'pyspark.sql.DataFrame', group_cols: List[str], agg_dict: Dict[str, str]) -> 'pyspark.sql.DataFrame':
        """
        Perform groupBy and aggregate operations on a Spark DataFrame.
        
        :param df: Input Spark DataFrame
        :param group_cols: List of columns to group by
        :param agg_dict: Dictionary of column names and aggregate functions
        :return: Aggregated Spark DataFrame
        """
        return df.groupBy(group_cols).agg(agg_dict)

    def join_dataframes(self, df1: 'pyspark.sql.DataFrame', df2: 'pyspark.sql.DataFrame', join_col: str, join_type: str = "inner") -> 'pyspark.sql.DataFrame':
        """
        Join two Spark DataFrames.
        
        :param df1: First Spark DataFrame
        :param df2: Second Spark DataFrame
        :param join_col: Column to join on
        :param join_type: Type of join (e.g., "inner", "left", "right", "full")
        :return: Joined Spark DataFrame
        """
        return df1.join(df2, on=join_col, how=join_type)

    def stop_spark(self) -> None:
        """
        Stop the Spark session.
        """
        self.spark.stop()
