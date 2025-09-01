import sys
import pandas as pd
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns

def main(market_question):
    df = pd.read_csv("data/kalshi.csv")
    train_df = df[df['market_question'] != market_question]
    X_train = train_df.drop(columns=['target', 'date', 'market', 'market_question'])
    y_train = train_df['target']

    test_df = df[df['market_question'] == market_question]
    test_dates = test_df['date'].values
    X_test = test_df.drop(columns=['target', 'date', 'market', 'market_question'])
    y_test = test_df['target']

    cat = CatBoostRegressor(
        depth=4,
        iterations=121,
        learning_rate=0.0017066305219717408,
        random_state=42,
        verbose=0
    )

    cat.fit(X_train, y_train)

    y_pred = cat.predict(X_test)

    plt.figure(figsize=(12, 10))
    corr_matrix = X_train.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Heatmap')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(test_dates[:10], y_test, label='Actual values', linestyle='-', marker='s', c='blue')
    plt.plot(test_dates[:10], y_pred, label='Predicted values', linestyle='-', marker='d', c='red')

    plt.xlabel('Sample period')
    plt.ylabel('Contract prices')
    plt.xticks(rotation=45)
    plt.title(market_question)
    plt.legend()
    plt.grid(axis='y', linestyle=':')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        market_question = sys.argv[1]
    else:
        market_question = 'Will Trump win the Electoral College and Trump win the Popular Vote?'
    main(market_question)