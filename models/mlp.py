import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load your data
df = pd.read_csv("data/kalshi.csv")
df = df.drop(columns=[c for c in ['market', 'market_question', 'date'] if c in df.columns], errors='ignore')
df = df.dropna()

# Assume 'target' is a numeric column representing returns
X = df.drop('target', axis=1)
y = df['target']

X = X.select_dtypes(include=[np.number]).values
y = y.values

# Normalize features
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

# Create sequences for time-series
def create_sequences(X, y, lag=30):
    Xs, ys = [], []
    for i in range(lag, len(X)):
        Xs.append(X[i-lag:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

lag = 30
X_seq, y_seq = create_sequences(X, y, lag=lag)

# Split chronologically (e.g., last 20% for test)
train_size = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1,1)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Take the last output
        out = out[:, -1, :]
        out = self.fc(out)
        return out

input_size = X_train.shape[2]
model = LSTMModel(input_size=input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

model.eval()
y_pred = model(X_test_t).detach().numpy().flatten()
y_true = y_test

mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
mean_pred = np.mean(y_pred)
std_pred = np.std(y_pred, ddof=1)
sharpe =  np.divide(np.mean(y_pred) , np.std(y_pred, ddof=1))

print("LSTM Results:")
print(f"MSE: {mse}")
print(f"R²: {r2}")
print(f"Sharpe Ratio: {sharpe}")
