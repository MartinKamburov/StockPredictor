import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

#The look_back value is used for the LSTM model to look back at the previous days to predict the next day's closing price.
def prepare_data(ticker, period="10y", look_back=1250):
    # Download last 10 years of data
    df = yf.download(ticker, period=period)

    print(df)

    # Select features: Open, Close, Volume, High, Low
    df['MA_50'] = df['Close'].rolling(window=50).mean()  # 50-day moving average
    df['MA_200'] = df['Close'].rolling(window=200).mean()  # 200-day moving average
    df['RSI'] = compute_rsi(df['Close'])  # RSI (custom function needed)

    # Drop NaN rows caused by rolling calculations
    df.dropna(inplace=True)

    # Select features: Open, Close, Volume, High, Low
    data = df[['Open', 'Close', 'Volume', 'High', 'Low', 'MA_50', 'MA_200', 'RSI']].values

    # Scale the data. This helps to ensure that all features like open, close, volume, etc. are on a similar scale.
    # This is important to ensure that no single feature dominates the calculations in an algorithm.
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)

    X = []  # Features
    Y = []  # Target

    # Predicts the next day's Close
    for i in range(look_back, len(scaled_data)): # Start from 'look_back' days because we need to look back that many days
        X.append(scaled_data[i-look_back:i, :])  # Use the previous 'look_back' days as features
        Y.append(scaled_data[i, 1])  # The Close value of the next day

    X = np.array(X)
    Y = np.array(Y)

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    # Split into train and test sets
    train_size = int(len(X)*0.8)
    X_train, X_test = X[:train_size], X[train_size:] # 80% train, 20% test
    Y_train, Y_test = Y[:train_size], Y[train_size:] # 80% train, 20% test

    if not os.path.exists('data'):
        os.makedirs('data')

    np.save('data/X_train.npy', X_train)
    np.save('data/Y_train.npy', Y_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/Y_test.npy', Y_test)

    joblib.dump(scaler, 'data/scaler.pkl')
    print("Data prepared and saved successfully.")

# RSI computation helper function
def compute_rsi(series, window=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

if __name__ == "__main__":
    prepare_data()