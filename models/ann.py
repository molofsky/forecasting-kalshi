import pandas as pd
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    mean_absolute_percentage_error
)
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import ParameterGrid

# Load the data with 'date' parsed as datetime
df = pd.read_csv("data/kalshi.csv", parse_dates=['date'])

# Sort the data by date to ensure chronological order
df = df.sort_values('date')

# Drop unwanted columns but retain 'date' for ARIMA
df_model = df.drop(columns=['market', 'market_question'])

# Define target and features
X = df_model.drop(['target', 'date'], axis=1)
y = df_model['target']

# Perform a time-based train-test split
train_size = int(len(df_model) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def evaluate_model(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name} Performance:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  MAPE: {mape:.6f}")
    print(f"  R²: {r2:.6f}")
    print()

# Define the model-building function
def build_ann_model(input_dim, neurons1, neurons2, dropout_rate, learning_rate):
    model = Sequential([
        Dense(neurons1, input_dim=input_dim, activation='relu'),
        Dropout(dropout_rate),
        Dense(neurons2, activation='relu'),
        Dropout(dropout_rate),
        Dense(1)  # Output layer for regression
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# Define the parameter grid
param_grid = {
    'neurons1': [32, 64, 128],
    'neurons2': [16, 32, 64],
    'dropout_rate': [0.1, 0.2, 0.3],
    'learning_rate': [0.001, 0.01, 0.0001],
    'batch_size': [16, 32, 64]
}

# Perform grid search
best_mse = float('inf')
best_params = None

for params in ParameterGrid(param_grid):
    print(f"Testing parameters: {params}")
    model = build_ann_model(
        input_dim=X_train_scaled.shape[1],
        neurons1=params['neurons1'],
        neurons2=params['neurons2'],
        dropout_rate=params['dropout_rate'],
        learning_rate=params['learning_rate']
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=50,  
        batch_size=params['batch_size'],
        callbacks=[early_stop],
        verbose=0
    )
    y_pred = model.predict(X_test_scaled).flatten()
    mse = mean_squared_error(y_test, y_pred)
    print(f"  MSE: {mse:.6f}")
    if mse < best_mse:
        best_mse = mse
        best_params = params

print(f"Best Parameters: {best_params}")
print(f"Best MSE: {best_mse}")

final_model = build_ann_model(
    input_dim=X_train_scaled.shape[1],
    neurons1=best_params['neurons1'],
    neurons2=best_params['neurons2'],
    dropout_rate=best_params['dropout_rate'],
    learning_rate=best_params['learning_rate']
)
history = final_model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=best_params['batch_size'],
    callbacks=[early_stop],
    verbose=1
)

y_pred_final = final_model.predict(X_test_scaled).flatten()
evaluate_model("Final ANN Model", y_test, y_pred_final)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
