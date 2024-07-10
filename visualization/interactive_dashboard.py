import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from typing import List, Dict
import yfinance as yf

from utils.math_utils import calculate_sharpe_ratio, calculate_sortino_ratio, calculate_max_drawdown, calculate_var, calculate_cvar

class FinancialDashboard:
    def __init__(self, tickers: List[str]):
        self.tickers = tickers
        self.data = self.load_data()
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()

    def load_data(self) -> Dict[str, pd.DataFrame]:
        data = {}
        for ticker in self.tickers:
            df = yf.download(ticker, start="2018-01-01", end="2023-01-01")
            df['Returns'] = df['Adj Close'].pct_change()
            data[ticker] = df
        return data

    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1('Financial Risk Dashboard'),
            
            dcc.Dropdown(
                id='ticker-dropdown',
                options=[{'label': ticker, 'value': ticker} for ticker in self.tickers],
                value=self.tickers[0]
            ),
            
            dcc.Tabs([
                dcc.Tab(label='Price and Returns', children=[
                    dcc.Graph(id='price-chart'),
                    dcc.Graph(id='returns-histogram')
                ]),
                dcc.Tab(label='Risk Metrics', children=[
                    html.Div(id='risk-metrics'),
                    dcc.Graph(id='rolling-volatility')
                ]),
                dcc.Tab(label='Performance Metrics', children=[
                    html.Div(id='performance-metrics'),
                    dcc.Graph(id='cumulative-returns')
                ])
            ])
        ])

    def setup_callbacks(self):
        @self.app.callback(
            [Output('price-chart', 'figure'),
             Output('returns-histogram', 'figure'),
             Output('risk-metrics', 'children'),
             Output('rolling-volatility', 'figure'),
             Output('performance-metrics', 'children'),
             Output('cumulative-returns', 'figure')],
            [Input('ticker-dropdown', 'value')]
        )
        def update_graphs(selected_ticker):
            df = self.data[selected_ticker]
            
            # Price Chart
            price_chart = go.Figure()
            price_chart.add_trace(go.Scatter(x=df.index, y=df['Adj Close'], mode='lines', name='Price'))
            price_chart.update_layout(title=f'{selected_ticker} Price', xaxis_title='Date', yaxis_title='Price')

            # Returns Histogram
            returns_hist = go.Figure()
            returns_hist.add_trace(go.Histogram(x=df['Returns'].dropna(), nbinsx=50, name='Returns'))
            returns_hist.update_layout(title=f'{selected_ticker} Returns Distribution', xaxis_title='Returns', yaxis_title='Frequency')

            # Risk Metrics
            risk_free_rate = 0.02  # Assume 2% risk-free rate
            sharpe_ratio = calculate_sharpe_ratio(df['Returns'], risk_free_rate)
            sortino_ratio = calculate_sortino_ratio(df['Returns'], risk_free_rate)
            max_drawdown = calculate_max_drawdown(df['Adj Close'])
            var_95 = calculate_var(df['Returns'])
            cvar_95 = calculate_cvar(df['Returns'])

            risk_metrics = html.Div([
                html.H3('Risk Metrics'),
                html.P(f'Sharpe Ratio: {sharpe_ratio:.2f}'),
                html.P(f'Sortino Ratio: {sortino_ratio:.2f}'),
                html.P(f'Max Drawdown: {max_drawdown:.2%}'),
                html.P(f'VaR (95%): {var_95:.2%}'),
                html.P(f'CVaR (95%): {cvar_95:.2%}')
            ])

            # Rolling Volatility
            rolling_vol = df['Returns'].rolling(window=30).std() * np.sqrt(252)
            vol_chart = go.Figure()
            vol_chart.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol, mode='lines', name='30-day Rolling Volatility'))
            vol_chart.update_layout(title=f'{selected_ticker} 30-day Rolling Volatility', xaxis_title='Date', yaxis_title='Annualized Volatility')

            # Performance Metrics
            total_return = (df['Adj Close'].iloc[-1] / df['Adj Close'].iloc[0]) - 1
            annualized_return = (1 + total_return) ** (252 / len(df)) - 1
            annualized_volatility = df['Returns'].std() * np.sqrt(252)

            performance_metrics = html.Div([
                html.H3('Performance Metrics'),
                html.P(f'Total Return: {total_return:.2%}'),
                html.P(f'Annualized Return: {annualized_return:.2%}'),
                html.P(f'Annualized Volatility: {annualized_volatility:.2%}')
            ])

            # Cumulative Returns
            cumulative_returns = (1 + df['Returns']).cumprod()
            cum_returns_chart = go.Figure()
            cum_returns_chart.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns, mode='lines', name='Cumulative Returns'))
            cum_returns_chart.update_layout(title=f'{selected_ticker} Cumulative Returns', xaxis_title='Date', yaxis_title='Cumulative Returns')

            return price_chart, returns_hist, risk_metrics, vol_chart, performance_metrics, cum_returns_chart

    def run_server(self, debug=True, port=8050):
        self.app.run_server(debug=debug, port=port)
