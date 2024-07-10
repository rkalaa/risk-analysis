from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="financial-risk-assessment",
    version="1.0.0",
    author="Habib Rahman",
    author_email="lhlrahman@gmail.com",
    description="A comprehensive financial risk assessment model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/financial-risk-assessment",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0,<2.0.0",
        "pandas>=1.3.0,<2.0.0",
        "scipy>=1.7.0,<2.0.0",
        "scikit-learn>=1.0.0,<2.0.0",
        "tensorflow>=2.8.0,<3.0.0",
        "keras>=2.8.0,<3.0.0",
        "xgboost>=1.5.0,<2.0.0",
        "pyspark>=3.2.0,<4.0.0",
        "matplotlib>=3.5.0,<4.0.0",
        "seaborn>=0.11.0,<0.12.0",
        "plotly>=5.6.0,<6.0.0",
        "dash>=2.3.0,<3.0.0",
        "yfinance>=0.1.70,<0.2.0",
        "flask>=2.0.0,<3.0.0",
        "flask-restx>=0.5.0,<0.6.0",
        "sqlalchemy>=1.4.0,<2.0.0",
        "pymysql>=1.0.0,<2.0.0",
        "psycopg2-binary>=2.9.0,<3.0.0",
        "hdfs>=2.6.0,<3.0.0",
        "boto3>=1.21.0,<2.0.0",
        "docker>=5.0.0,<6.0.0",
        "kubernetes>=23.0.0,<24.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.1.0,<8.0.0",
            "pytest-cov>=3.0.0,<4.0.0",
            "black>=22.3.0,<23.0.0",
            "flake8>=4.0.0,<5.0.0",
            "mypy>=0.940,<1.0.0",
        ],
        "docs": [
            "sphinx>=4.5.0,<5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "run-risk-assessment=scripts.run_risk_assessment:main",
            "backtest-models=scripts.backtest_models:main",
            "generate-reports=scripts.generate_reports:main",
        ],
    },
)