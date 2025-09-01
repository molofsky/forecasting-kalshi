import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, explained_variance_score

from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
from scipy.stats import uniform, randint

df = pd.read_csv("data/kalshi.csv")

df = df.drop(columns=[col for col in ['market', 'market_question', 'date'] if col in df.columns])
df = df.dropna()

X = df.drop('target', axis=1)
y = df['target']

X = X.select_dtypes(include=[np.number])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def evaluate_model(name, y_true, y_pred, additional_metrics=False):
    n = len(y_true)
    _ , p = X_test.shape

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

gb_params = {
    'n_estimators': randint(100, 500),
    'learning_rate': uniform(0.001, 0.1),
    'max_depth': randint(3, 10),
    'subsample': uniform(0.5, 0.5)
}
gb_model = GradientBoostingRegressor(random_state=42)
gb_search = RandomizedSearchCV(
    gb_model, gb_params, n_iter=20, cv=3, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1
)
gb_search.fit(X_train, y_train)
best_gb = gb_search.best_estimator_
y_pred_gb = best_gb.predict(X_test)
print("Best Gradient Boosting Params:", gb_search.best_params_)
evaluate_model("Gradient Boosting", y_test, y_pred_gb)

sgd_params = {
    'alpha': uniform(1e-5, 1e-2),
    'eta0': uniform(1e-4, 0.1),
    'penalty': ['l2', 'l1', 'elasticnet'],
    'max_iter': [1000, 2000, 3000]
}
sgd_model = SGDRegressor(random_state=42)
sgd_search = RandomizedSearchCV(
    sgd_model, sgd_params, n_iter=20, cv=3, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1
)
sgd_search.fit(X_train_scaled, y_train)
best_sgd = sgd_search.best_estimator_
y_pred_sgd = best_sgd.predict(X_test_scaled)
print("Best SGD Regressor Params:", sgd_search.best_params_)
evaluate_model("SGD Regressor", y_test, y_pred_sgd)

cat_params = {
    'iterations': randint(100, 500),
    'learning_rate': uniform(0.001, 0.1),
    'depth': randint(3, 10)
}
cat_model = CatBoostRegressor(random_state=42, verbose=0)
cat_search = RandomizedSearchCV(
    cat_model, cat_params, n_iter=20, cv=3, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1
)
cat_search.fit(X_train, y_train)
best_cat = cat_search.best_estimator_
y_pred_cat = best_cat.predict(X_test)
print("Best CatBoost Params:", cat_search.best_params_)
evaluate_model("CatBoost", y_test, y_pred_cat)
