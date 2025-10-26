import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def train_linear_regression(X, y):
    model = LinearRegression()
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
        'price', 'price_change_1d', 'price_change_3d', 'price_change_7d',
        'rolling_mean_7d', 'rolling_std_7d', 'price_acceleration',
        'volatility_ratio',
        'price_7d_high', 'price_7d_low',
    ]
    
    X = df[features]
    y = df['target']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the linear regression model
    model = train_linear_regression(X_train, y_train)
    
    # Evaluate the model on the test set
    mse_test, r2, mae, y_pred = evaluate_model(model, X_test, y_test)
    
    # Calculate the dates for the test set
    test_dates = df['date'].iloc[X_test.index]

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Date': test_dates,
        'Actual Prices': y_test,
        'Predicted Prices': y_pred
    })

    # Plotting the actual vs predicted prices
    plt.figure(figsize=(12, 6))
    plt.plot(plot_df['Date'], plot_df['Actual Prices'], label='Actual Prices', color='blue', marker='o', linestyle='-')
    plt.plot(plot_df['Date'], plot_df['Predicted Prices'], label='Predicted Prices', color='orange', marker='x', linestyle='--')
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Prices', fontsize=16)
    plt.title('Actual vs Predicted Prices for Market Question', fontsize=18)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Calculate average forecast error
    average_forecast_error = np.mean((y_test - y_pred) / y_test)
    print(f'Average Forecast Error of the linear regression model: {average_forecast_error:.4f}')
    
    # Calculate training error
    y_train_pred = model.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_pred)
    
    print(f'Mean Squared Error (Training): {mse_train:.4f}')
    print(f'Mean Squared Error (Testing): {mse_test:.4f}')
    print(f'R^2 Score (Testing): {r2:.4f}')
    
    # Calculate average error
    average_error = (mae + mean_absolute_error(y_train, y_train_pred)) / 2
    print(f'Average Error (Mean Absolute Error): {average_error:.4f}')
    
    # Print Model Coefficients as Feature Importances
    print('Model Coefficients (Feature Importances):')
    for feature, coef in zip(features, model.coef_):
        print(f'{feature}: {coef:.4f}')
    
    # Plot Feature Importances
    plt.figure(figsize=(10, 6))
    plt.barh(features, model.coef_, color='skyblue')
    plt.xlabel('Coefficient Value', fontsize=16)
    plt.title('Feature Importances', fontsize=18)
    plt.axvline(0, color='grey', linestyle='--')
    plt.savefig("feature_importances.png")  # Save the feature importance plot
    plt.show()

    # Calculate and plot the covariance matrix
    covariance_matrix = X.cov()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(covariance_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Covariance Matrix of Features', fontsize=18)
    plt.savefig("covariance_matrix.png")
    plt.show()

    # Print features with low importance
    low_importance_features = [(feature, coef) for feature, coef in zip(features, model.coef_) if abs(coef) < 0.1]
    
    if low_importance_features:
        print('\nFeatures with low importance (correlation coefficient < 0.1 in magnitude):')
        for feature, coef in low_importance_features:
            print(f'{feature}: {coef:.4f}')
    else:
        print('\nNo features with low importance (correlation coefficient < 0.1 in magnitude).')

    # Calculate residuals
    residuals = y_test - y_pred

    # Plot residuals
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, color='blue', alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Values', fontsize=16)
    plt.ylabel('Residuals', fontsize=16)
    plt.title('Residuals Plot', fontsize=18)
    plt.savefig("residuals_plot.png")  
    plt.show()

if __name__ == "__main__":
    main()