# Financial Risk Assessment API Documentation

## Base URL

`https://api.financialriskmodel.com/v1`

## Authentication

All API requests require authentication using OAuth 2.0. Include the access token in the Authorization header:

```
Authorization: Bearer YOUR_ACCESS_TOKEN
```

## Endpoints

### 1. Portfolio Optimization

#### POST /optimize_portfolio

Optimizes a portfolio for maximum Sharpe ratio.

**Request Body:**

```json
{
  "returns": {
    "AAPL": [0.01, 0.02, -0.01, ...],
    "GOOGL": [0.015, -0.01, 0.02, ...],
    "MSFT": [-0.005, 0.025, 0.01, ...]
  },
  "risk_free_rate": 0.02
}
```

**Response:**

```json
{
  "weights": {
    "AAPL": 0.4,
    "GOOGL": 0.35,
    "MSFT": 0.25
  },
  "expected_return": 0.12,
  "volatility": 0.18,
  "sharpe_ratio": 0.56
}
```

### 2. Risk Metrics Calculation

#### POST /calculate_risk_metrics

Calculates VaR and CVaR for a given portfolio.

**Request Body:**

```json
{
  "returns": {
    "AAPL": [0.01, 0.02, -0.01, ...],
    "GOOGL": [0.015, -0.01, 0.02, ...],
    "MSFT": [-0.005, 0.025, 0.01, ...]
  },
  "weights": {
    "AAPL": 0.4,
    "GOOGL": 0.35,
    "MSFT": 0.25
  },
  "confidence_level": 0.95
}
```

**Response:**

```json
{
  "VaR": 0.025,
  "CVaR": 0.035
}
```

### 3. Monte Carlo Simulation

#### POST /monte_carlo_simulation

Performs Monte Carlo simulation for portfolio returns.

**Request Body:**

```json
{
  "returns": {
    "AAPL": [0.01, 0.02, -0.01, ...],
    "GOOGL": [0.015, -0.01, 0.02, ...],
    "MSFT": [-0.005, 0.025, 0.01, ...]
  },
  "weights": {
    "AAPL": 0.4,
    "GOOGL": 0.35,
    "MSFT": 0.25
  },
  "num_simulations": 10000,
  "time_horizon": 252
}
```

**Response:**

```json
{
  "simulations": [
    [100, 102, 98, ...],
    [100, 101, 103, ...],
    ...
  ]
}
```

## Error Handling

The API uses standard HTTP response codes. Errors are returned in the following format:

```json
{
  "error": "Error message",
  "code": "ERROR_CODE"
}
```

## Rate Limiting

Requests are limited to 100 per minute per API key. The following headers are included in the response:

- `X-RateLimit-Limit`: The number of allowed requests in the current period
- `X-RateLimit-Remaining`: The number of remaining requests in the current period
- `X-RateLimit-Reset`: The time at which the current rate limit window resets

## Versioning

The API is versioned using URL path versioning. The current version is v1.