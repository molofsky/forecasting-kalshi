import numpy as np
import pandas as pd
from sklearn.metrics import (r2_score, mean_absolute_error,
    mean_absolute_percentage_error, explained_variance_score
)
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

df = pd.read_csv("data/kalshi.csv", parse_dates=['date'])

df = df.sort_values('date')

# Display the first few rows to verify sorting
print("First 5 rows after sorting by date:")
print(df.head())

# Drop unwanted columns but retain 'date' for ARIMA
df_model = df.drop(columns=['market', 'market_question'])

# Define target and features
# Exclude 'date' from features to prevent non-numeric data in ANN and other models
X = df_model.drop(['target', 'date'], axis=1)
y = df_model['target']

# Perform a time-based train-test split by disabling shuffling
train_size = int(len(df_model) * 0.8)

X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Verify the split
print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

def evaluate_model(name, y_true, y_pred, additional_metrics=False):
    n = len(y_true)
    p = X_test.shape[1]

    residuals = y_true - y_pred
    mse = np.sum(residuals ** 2) / n
    rmse = np.sqrt(mse)
    ssr = np.sum(residuals ** 2)
    se = np.sqrt(ssr / (n - p - 1))
    r2 = r2_score(y_true, y_pred) 
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    aic = n * np.log(mse) + 2 * (p + 1)
    bic = n * np.log(mse) + np.log(n) * (p + 1) 
    dw = np.sum(np.diff(residuals) ** 2) / ssr 

    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    
    print(f"{name}:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  Sum of squared residuals: {ssr:.6f}")
    print(f"  S.E. of regression: {se:.6f}")
    print(f"  R²: {r2:.6f}")
    print(f"  Adjusted R²: {adj_r2:.6f}")
    print(f"  Akaike Information Criterion (AIC): {aic:.6f}")
    print(f"  Schwarz Criterion (BIC): {bic:.6f}")
    print(f"  Durbin-Watson Statistic: {dw:.6f}")
    if additional_metrics:
        print(f"  MAE: {mae:.6f}")
        print(f"  MAPE: {mape:.6f}")
        print(f"  EVS: {evs:.6f}")
    print()

# ARIMA requires a univariate time series
# We'll model 'target' as the time series

# Extract the 'target' series
series = df_model['target']

# Split the series into train and test based on the earlier split
train_arima, test_arima = series.iloc[:train_size], series.iloc[train_size:]

# Optional: Plot the series to visualize
plt.figure(figsize=(12,6))
plt.plot(train_arima.index, train_arima, label='Train')
plt.plot(test_arima.index, test_arima, label='Test')
plt.title('Target Time Series for ARIMA')
plt.legend()
plt.show()

# Use auto_arima to find the best (p,d,q) parameters
auto_model = pm.auto_arima(
    train_arima, 
    seasonal=False, 
    stepwise=True, 
    suppress_warnings=True,
    error_action='ignore'
)
print("ARIMA Model Summary:")
print(auto_model.summary())

# Fit the ARIMA model with the optimal order
model_arima = ARIMA(train_arima, order=auto_model.order)
model_arima_fit = model_arima.fit()

# Forecast the test set
forecast_arima = model_arima_fit.forecast(steps=len(test_arima))

# Align the forecast index with the test index
forecast_arima.index = test_arima.index

# Evaluate ARIMA
evaluate_model("ARIMA", test_arima, forecast_arima)

# Feature Scaling for ANN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the ANN model
model_ann = Sequential([
    Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)  # Output layer for regression
])

# Compile the model
model_ann.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the ANN model
history_ann = model_ann.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Plot ANN Training History
plt.figure(figsize=(12,6))
plt.plot(history_ann.history['loss'], label='Training Loss')
plt.plot(history_ann.history['val_loss'], label='Validation Loss')
plt.title('ANN Model Training History')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

# Predict with ANN
y_pred_ann = model_ann.predict(X_test_scaled).flatten()

# Evaluate ANN
evaluate_model("ANN", y_test, y_pred_ann, additional_metrics=True)

# Plot actual vs predicted for ARIMA and ANN
plt.figure(figsize=(14,7))
plt.plot(test_arima.index, test_arima, label='Actual', color='blue')
plt.plot(test_arima.index, forecast_arima, label='ARIMA Forecast', color='red')
plt.plot(test_arima.index, y_pred_ann, label='ANN Prediction', color='green', alpha=0.7)
plt.title('Actual vs ARIMA Forecast vs ANN Prediction')
plt.xlabel('Date')
plt.ylabel('Target')
plt.legend()
plt.show()

# Optional: Scatter plot for ANN
plt.figure(figsize=(8,8))
plt.scatter(y_test, y_pred_ann, alpha=0.5, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('ANN Model: Actual vs Predicted')
plt.show()
