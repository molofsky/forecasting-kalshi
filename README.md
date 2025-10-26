# Price-Pure Prediction of Daily Price Changes in Binary Event Contracts

# Overview 

This repository inclues time series forecasting from high volume binary event contacts, investigating whether the innately sentiment- and news-driven prices of these contracts can be accurately predicted purely on the basis of timeprice data. This project benchmarks the performance of linear regression, decision trees, ensembles, artificial neural networks, and autoregressive integrated moving average methods on Kalshi binary event contracts.

# What is in this repo?
- `data`: Contains dataset collection, preprocessing, and feature engineering.
- `models`: All PyTorch and scikit-learn models for predicting daily price changes.
- `plots`: Includes model performance, visualization, and summary statistics.

# Dataset

This dataset derives from 10,065 examples of high trading volume binary event contracts spanning all topics, years, resolved, and active collected from the Kalshi prediction market. The rows are daily examples, columns features. The target output is one-day-ahead percentage price change. There are 20 price/date-derived features. Features of low importance in influencing the target are pruned to optimize model performance. The models are trained using k-fold cross validation to combat sampling bias.

| Feature | Description |
|--------------|-----------------|
| `price_change_1d` | 1-day price difference |
| `price_change_3d` | 3-day price difference |
| `price_change_7d` | 7-day price difference  |
| `rolling_mean_7d` | 7-day rolling mean of price |
| `rolling_std_7d` | 7-day rolling standard deviation of price |
| `momentum_1d` | 1-day momentum |
| `momentum_3d` | 3-day momentum |
| `momentum_7d` | 7-day momentum  |
| `time_since_start` | Days since first observation |
| `price_acceleration` | Change in daily price difference  |
| `volatility_ratio` | Ratio of volatility to mean |
| `above_7d_mean` | 1 if price > 7-day mean else 0 |
| `distance_from_7d_mean` | Absolute distance from rolling 7-day mean |
| `price_7d_high` | 7-day rolling maximum price |
| `price_7d_low` | 7-day rolling minimum price |
| `mean_crossover` | 1 if price crosses above rolling mean |
| `price_range_ratio` | (7-day high − 7-day low) / rolling mean |
| `volatility_trend` | Change in 7-day volatility |
| `days_to_7d_high` | Count of days since last 7-day high |
| `target` | Next-day percentage price change |

# Running models

1. Create a Python virtual environment

   ```
   python3 -m venv forecasting-kalshi-env
   source forecasting-kalshi-env/bin/activate
   ```
   
2. Install Python dependencies

   ```
   pip install -r requirements.txt
   ```

3. Pick a model and start training
   ```
   python models/ann.py
   ```

# Contributing

Please format the code before submitting changes.

   ```
   python -m black *.py
   ```

# Results

The results from five model architectures, evaluated using mean squared error (MSE) and mean absolute error (MAE) as loss metrics, are shown below.

| Model     | MSE  |  MAE   |
|----------------|-----------|-----------|
| Linear Regression        | 0.0425    | 0.0979    |
| Decision Tree       | 0.1470    | 0.0918    |
| Gradient Boosting      | 0.0397    | 0.0840    |
| Histogram-Based Gradient Boosting     | 0.0395    | 0.0845    |
| Categorical Gradient Boosting       | 0.0395    | 0.0835    |