
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

class data:
    def __init__(self, ticker, window_length, t):
        self.b = []
        self.t = t
        self.ticker = ticker
        self.window_length = window_length
        self.data = self.download_data()

    def download_data(self):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*5)  # 5 years of data
        print(f"Downloading data for {self.ticker} from {start_date} to {end_date}")
        df = yf.download(self.ticker, start=start_date, end=end_date)
        print(f"Downloaded {len(df)} rows of data")
        if df.empty:
            print(f"Warning: No data downloaded for {self.ticker}")
            return df
        df['Close'] = df['Adj Close']
        df = df.drop(columns=['Adj Close'])
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['ROC'] = self.calculate_roc(df['Close'])
        df['CCI'] = self.calculate_cci(df['High'], df['Low'], df['Close'])
        df['MACD'], df['Signal'] = self.calculate_macd(df['Close'])
        df['EXPMA'] = self.calculate_expma(df['Close'])
        df['VMACD'], _ = self.calculate_macd(df['Volume'])
        print(f"Processed data shape: {df.shape}")
        return df

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_roc(self, prices, period=12):
        return prices.pct_change(period)

    def calculate_cci(self, high, low, close, period=20):
        tp = (high + low + close) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (tp - sma) / (0.015 * mad)

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def calculate_expma(self, prices, period=20):
        return prices.ewm(span=period, adjust=False).mean()

    def process(self):
        df = self.data.loc[:, ['Close', 'Open', 'High', 'Low', 'RSI', 'ROC', 'CCI', 'MACD', 'EXPMA', 'VMACD']]
        self.b = [df[i-self.window_length:i] for i in range(self.window_length, len(df))]
        print(f"Processed {len(self.b)} data points")
        return self.b

    def train_data(self):
        self.b = self.process()
        train_data = self.b[:self.t]
        print(f"Returning {len(train_data)} training data points")
        return train_data

    def trade_data(self):
        self.b = self.process()
        trade_data = self.b[self.t:]
        print(f"Returning {len(trade_data)} trading data points")
        return trade_data

if __name__ == "__main__":
    D = data(ticker="SPY", window_length=15, t=2000)
    b = D.trade_data()
    print(type(b), len(b), type(b[0]) if b else "No data")
    if b:
        print(b[0])
    else:
        print("No data to display")