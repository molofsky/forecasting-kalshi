import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, explained_variance_score
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from catboost import CatBoostRegressor

df = pd.read_csv("data/kalshi.csv")

df = df.drop(columns=[col for col in ['market', 'market_question', 'date'] if col in df.columns])

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

def evaluate_model(name, y_true, y_pred, additional_metrics=False):
    n = len(y_true)
    _ , p = X_test.shape

    residuals = y_true - y_pred
    mse = mean_squared_error(y_true, y_pred)
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
    actual_sharpe = np.divide(np.mean(y_true), np.std(y_true, ddof=1))
    pred_sharpe = np.divide(np.mean(y_pred), np.std(y_pred, ddof=1))
    
    print("{}:".format(name))
    print("  MSE: {}".format(mse))
    print("  RMSE: {}".format(rmse))
    print("  Sum of squared residuals: {}".format(ssr))
    print("  S.E. of regression: {}".format(se))
    print("  R²: {}".format(r2))
    print("  Adjusted R²: {}".format(adj_r2))
    print("  Akaike Information Criterion (AIC): {}".format(aic))
    print("  Schwarz Criterion (BIC): {}".format(bic))
    print("  Durbin-Watson Statistic: {}".format(dw))
    if additional_metrics:
        print("  MAE: {}".format(mae))
        print("  MAPE: {}".format(mape))
        print("  EVS: {}".format(evs))
    print("  Actual Sharpe Ratio: {}".format(actual_sharpe))
    print("  Predicted Sharpe Ratio: {}".format(pred_sharpe))
    print()

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
evaluate_model("Linear Regression", y_test, y_pred_lr, additional_metrics=True)

sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
sgd.fit(scaler.fit_transform(X_train), y_train)
y_pred_sgd = sgd.predict(scaler.fit_transform(X_test))
evaluate_model("SGD Regressor", y_test, y_pred_sgd)

gb = GradientBoostingRegressor(
    subsample=0.5,
    n_estimators=100,
    max_depth=7,
    learning_rate=0.001,
    random_state=42
)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
evaluate_model("Gradient Boosting", y_test, y_pred_gb, additional_metrics=True)

hgb = HistGradientBoostingRegressor(
    l2_regularization= 0.9385527090157502,
    learning_rate=0.0010778765841014328,
    max_iter=376,
    max_leaf_nodes=15
)
hgb.fit(X_train, y_train)
y_pred_hgb = hgb.predict(X_test)
evaluate_model("HistGradient Boosting", y_test, y_pred_hgb, additional_metrics=True)

cat = CatBoostRegressor(
    depth=4,
    iterations=121,
    learning_rate=0.0017066305219717408,
    random_state=42,
    verbose=0
)
cat.fit(X_train, y_train)
y_pred_cat = cat.predict(X_test)
evaluate_model("CatBoost", y_test, y_pred_cat, additional_metrics=True)
