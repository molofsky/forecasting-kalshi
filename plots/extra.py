import pandas as pd
import matplotlib.pyplot as plt

def plot_kalshi_data(csv_file):
    df = pd.read_csv(csv_file, header=None)

    df.columns = ['id', 'market_question', 'date', 'price', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 
                  'col10', 'col11', 'col12', 'col13', 'col14', 'col15', 'col16', 'col17', 'col18', 
                  'col19', 'col20', 'col21', 'col22', 'col23']

    market_question = "Will Trump win the Electoral College and Trump win the Popular Vote?"
    filtered_df = df[df['market_question'] == market_question]

    filtered_df['date'] = pd.to_datetime(filtered_df['date'])

    cutoff_date = pd.to_datetime("2024-11-22")
    
    filtered_df = filtered_df[filtered_df['date'] <= cutoff_date]
    filtered_df['smoothed_price'] = filtered_df['price'].rolling(window=3).mean() 

    plt.figure(figsize=(10, 5)) 
    plt.plot(filtered_df['date'], filtered_df['smoothed_price'], marker='o', linestyle='-', color='b')
    plt.title(market_question, fontsize=16) 
    plt.xlabel('Date', fontsize=14) 
    plt.ylabel('Price', fontsize=14)
    plt.xticks(rotation=45, fontsize=12) 
    plt.yticks(fontsize=12) 
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_kalshi_data('data/kalshi.csv')