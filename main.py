import numpy as np
import yfinance as yf
import joblib
from tensorflow.keras.models import load_model
from prepare_data import compute_rsi


def predict_next_day(ticker, look_back=1250):
    # Load model and scaler
    model = load_model('model/model.keras')
    scaler = joblib.load('data/scaler.pkl')
    
    # Download data to predict next day's close
    recent_data = yf.download(ticker, period="10y")
    recent_data['MA_50'] = recent_data['Close'].rolling(window=50).mean()
    recent_data['MA_200'] = recent_data['Close'].rolling(window=200).mean()
    recent_data['RSI'] = compute_rsi(recent_data['Close'])

    # Drop NaN rows caused by rolling calculations
    recent_data.dropna(inplace=True)
    recent_data = recent_data[['Open', 'Close', 'Volume', 'High', 'Low', 'MA_50', 'MA_200', 'RSI']]

    # Make sure we have enough data
    if len(recent_data) < look_back:
        raise ValueError(f"Not enough data to form a sequence of length {look_back}.")

    # Scale the features
    scaled_features = scaler.transform(recent_data.values)
    X_input = scaled_features[-look_back:, :]  # Use the last 'look_back' days
    X_input = np.reshape(X_input, (1, look_back, scaled_features.shape[1]))  # Adjust for 5 features


    # Make prediction
    prediction_scaled = model.predict(X_input)

    # Inverse transform the predicted scaled value
    dummy = np.zeros((1, X_input.shape[2]))
    dummy[0, 1] = prediction_scaled[0, 0]
    prediction_full = scaler.inverse_transform(dummy)
    predicted_close = prediction_full[0, 1]

    return predicted_close

# if __name__ == "__main__":
#     next_day_prediction = predict_next_day("AMD", 1250)
#     print(f"Predicted next day's closing price: {next_day_prediction:.2f}")