import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, explained_variance_score

from sklearn.linear_model import BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor

df = pd.read_csv("data/kalshi.csv")

df = df.drop(columns=[col for col in ['market', 'market_question', 'date'] if col in df.columns])

X = df.drop('target', axis=1)
y = df['target']

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

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
evaluate_model("Linear Regression", y_test, y_pred_lr)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)
evaluate_model("Ridge", y_test, y_pred_ridge)

lasso = Lasso(alpha=0.001, max_iter=10000)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)
evaluate_model("Lasso", y_test, y_pred_lasso)

elastic = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)
elastic.fit(X_train_scaled, y_train)
y_pred_elastic = elastic.predict(X_test_scaled)
evaluate_model("ElasticNet", y_test, y_pred_elastic)

bayesian_ridge = BayesianRidge()
bayesian_ridge.fit(X_train_scaled, y_train)
y_pred_br = bayesian_ridge.predict(X_test_scaled)
evaluate_model("Bayesian Ridge", y_test, y_pred_br)

kernel_ridge = KernelRidge(alpha=1.0, kernel='rbf')
kernel_ridge.fit(X_train_scaled, y_train)
y_pred_kr = kernel_ridge.predict(X_test_scaled)
evaluate_model("Kernel Ridge", y_test, y_pred_kr)

# SVR with Hyperparameter Tuning
svr_params = {
    'C': np.logspace(-3, 2, 10),
    'epsilon': np.linspace(0, 0.1, 5),
    'gamma': ['scale', 'auto']
}
svr_model = SVR(kernel='rbf')
svr_search = RandomizedSearchCV(svr_model, svr_params, n_iter=10, cv=3, random_state=42, scoring='neg_mean_squared_error', n_jobs=-1)
svr_search.fit(X_train_scaled, y_train)
best_svr = svr_search.best_estimator_
y_pred_svr = best_svr.predict(X_test_scaled)
print("Best SVR Params:", svr_search.best_params_)
evaluate_model("SVR", y_test, y_pred_svr)

rf = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
evaluate_model("Random Forest", y_test, y_pred_rf)

et = ExtraTreesRegressor(n_estimators=200, random_state=42)
et.fit(X_train, y_train)
y_pred_et = et.predict(X_test)
evaluate_model("Extra Trees Regressor", y_test, y_pred_et)

sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
sgd.fit(X_train_scaled, y_train)
y_pred_sgd = sgd.predict(X_test_scaled)
evaluate_model("SGD Regressor", y_test, y_pred_sgd)

gb_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.001, 0.01, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.5, 0.75, 1.0]
}
gb_model = GradientBoostingRegressor(random_state=42)
gb_search = RandomizedSearchCV(gb_model, gb_params, n_iter=10, cv=3, random_state=42, scoring='neg_mean_squared_error', n_jobs=-1)
gb_search.fit(X_train, y_train)
best_gb = gb_search.best_estimator_
y_pred_gb = best_gb.predict(X_test)
print("Best Gradient Boosting Params:", gb_search.best_params_)
evaluate_model("Gradient Boosting", y_test, y_pred_gb)

cat = CatBoostRegressor(iterations=200, learning_rate=0.01, depth=5, random_state=42, verbose=0)
cat.fit(X_train, y_train)
y_pred_cat = cat.predict(X_test)
evaluate_model("CatBoost", y_test, y_pred_cat)

mlp = MLPRegressor(hidden_layer_sizes=(50,50), alpha=0.001, learning_rate_init=0.001, max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)
y_pred_mlp = mlp.predict(X_test_scaled)
evaluate_model("MLPRegressor", y_test, y_pred_mlp)
