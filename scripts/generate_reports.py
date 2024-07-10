import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
import os
from fpdf import FPDF

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import VaRModel, CVaRModel, VolatilityModel, BetaModel, SharpeRatioModel
from data import DataLoader, DataPreprocessor

class RiskReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Financial Risk Assessment Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_risk_metrics(returns, risk_free_rate=0.02):
    var_model = VaRModel()
    cvar_model = CVaRModel()
    vol_model = VolatilityModel()
    beta_model = BetaModel()
    sharpe_model = SharpeRatioModel()

    risk_metrics = {}
    for column in returns.columns:
        risk_metrics[column] = {
            'VaR (95%)': var_model.calculate_var(returns[column], level=0.95),
            'CVaR (95%)': cvar_model.calculate_cvar(returns[column], level=0.95),
            'Annualized Volatility': vol_model.calculate_historical_volatility(returns[column]).iloc[-1],
            'Beta': beta_model.calculate_beta(returns[column], returns['^GSPC']),
            'Sharpe Ratio': sharpe_model.calculate_sharpe_ratio(returns[column], risk_free_rate)
        }
    
    return pd.DataFrame(risk_metrics).T

def plot_returns_distribution(returns):
    plt.figure(figsize=(12, 6))
    for column in returns.columns:
        sns.kdeplot(returns[column], label=column)
    plt.title('Returns Distribution')
    plt.xlabel('Returns')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('returns_distribution.png')
    plt.close()

def plot_correlation_heatmap(returns):
    plt.figure(figsize=(10, 8))
    sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap')
    plt.savefig('correlation_heatmap.png')
    plt.close()

def plot_cumulative_returns(returns):
    plt.figure(figsize=(12, 6))
    (1 + returns).cumprod().plot()
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.savefig('cumulative_returns.png')
    plt.close()

def generate_pdf_report(risk_metrics):
    pdf = RiskReport()
    pdf.add_page()

    # Risk Metrics Table
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Risk Metrics', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    col_width = pdf.w / 6
    row_height = pdf.font_size + 2
    for row in risk_metrics.itertuples():
        pdf.cell(col_width, row_height, str(row.Index), 1)
        for item in row[1:]:
            pdf.cell(col_width, row_height, f"{item:.4f}", 1)
        pdf.ln(row_height)

    pdf.ln(10)

    # Add plots
    for img in ['returns_distribution.png', 'correlation_heatmap.png', 'cumulative_returns.png']:
        pdf.image(img, x=10, w=190)
        pdf.ln(5)

    pdf.output('risk_assessment_report.pdf', 'F')

def main():
    # Load data
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', '^GSPC']
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    data_loader = DataLoader()
    data = data_loader.load_from_yfinance(tickers, start_date, end_date)
    
    preprocessor = DataPreprocessor()
    returns = preprocessor.calculate_returns(data)
    returns = preprocessor.handle_missing_values(returns)

    # Generate risk metrics
    risk_metrics = generate_risk_metrics(returns)
    print("Risk Metrics:")
    print(risk_metrics)

    # Generate plots
    plot_returns_distribution(returns)
    plot_correlation_heatmap(returns)
    plot_cumulative_returns(returns)

    # Generate PDF report
    generate_pdf_report(risk_metrics)
    print("PDF report generated: risk_assessment_report.pdf")

if __name__ == "__main__":
    main()