import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def train_decision_tree(X, y, params):
    model = DecisionTreeRegressor(random_state=42, **params)
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, r2, mae, y_pred

def main():
    df = load_data('kalshi.csv')
    
    features = [
        'price_change_7d',
        'time_since_start',
        'price_acceleration',
    ]
    
    X = df[features]
    y = df['target']
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)  # 10-fold cross-validation
    mae_list = []

    # Define parameter grid
    param_grid = {
        'max_depth': [None, 5, 10, 15, 20, 25],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'max_features': ['auto', 'sqrt', 'log2'],
        'criterion': ['squared_error']
    }
    
    # Perform grid search
    grid_search = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=kf, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = train_decision_tree(X_train, y_train, grid_search.best_params_)

        mse_test, r2, mae, y_pred = evaluate_model(model, X_test, y_test)
        mae_list.append(mae)

    # Calculate average error
    average_error = np.mean(mae_list)
    print(f'Average Error (Mean Absolute Error) of the best model: {average_error:.4f}')
    
    # Print Model Feature Importances
    if hasattr(best_model, 'feature_importances_'):
        print('Model Feature Importances:')
        for feature, importance in zip(features, best_model.feature_importances_):
            print(f'{feature}: {importance:.4f}')
    
    # Print features with low importance
    low_importance_features = [(feature, importance) for feature, importance in zip(features, best_model.feature_importances_) if abs(importance) < 0.1]
    
    if low_importance_features:
        print('\nFeatures with low importance (magnitude < 0.1):')
        for feature, importance in low_importance_features:
            print(f'{feature}: {importance:.4f}')
    else:
        print('\nNo features with low importance (magnitude < 0.1).')

if __name__ == "__main__":
    main()
