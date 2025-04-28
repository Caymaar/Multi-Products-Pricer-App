import yfinance as yf
import os


def main():
    data_folder = "data/prices"
    
    # Create data folder if it doesn't exist
    os.makedirs(data_folder, exist_ok=True)
    
    # Get stocks from config
    stocks = {
        'APPLE': 'AAPL',
        'MICROSOFT': 'MSFT',
        'AMAZON': 'AMZN',
        'TESLA': 'TSLA',
        'SP500': 'SPY',
        'ASML': 'ASML.AS',
        'LVMH': 'MC.PA',
        'EUROSTOXX': '^STOXX50E',
    }

    for name, symbol in stocks.items():
        print(f"Downloading data for {name} ({symbol})...")
        ticker = yf.Ticker(symbol)
        # Get full historical data (max available)
        df = ticker.history(period="max")

        df = df[['Close']]

        # Put Date index as YYYY-MM-DD
        df.index = df.index.strftime('%Y-%m-%d')

        # Rename the column to 'Close'
        df.rename(columns={'Close': name}, inplace=True)

        # Save to CSV file with the ticker name
        file_name = os.path.join(data_folder, f"{name}.csv")
        df.to_csv(file_name)
        print(f"Data saved to {file_name}")

if __name__ == "__main__":
    main()