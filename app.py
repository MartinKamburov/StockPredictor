import streamlit as st
from prepare_data import prepare_data
from train_model import train_model
from main import predict_next_day

# Streamlit app title
st.title("Stock Predictor")

# Dropdown menu to select the action
action = st.selectbox("Choose an action:", 
                      ["Prepare Data", "Train Model", "Predict Next Day's Price"])

# User input for stock ticker
ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, TSLA, AMD):", value="Type here...")

# Additional input for look-back period
look_back = st.number_input("Look-back period (e.g., 1250):", min_value=50, max_value=5000, value=1250)

# Perform actions based on the selected option
if action == "Prepare Data":
    if st.button("Prepare Data"):
        try:
            st.info(f"Preparing data for ticker: {ticker} with look-back period {look_back}")
            prepare_data(ticker=ticker, look_back=look_back)
            st.success("Data prepared successfully!")
        except Exception as e:
            st.error(f"An error occurred while preparing data: {e}")

elif action == "Train Model":
    if st.button("Train Model"):
        try:
            st.info("Training the model with the prepared data...")
            train_model()
            st.success("Model trained and saved successfully!")
        except Exception as e:
            st.error(f"An error occurred while training the model: {e}")

elif action == "Predict Next Day's Price":
    if st.button("Predict"):
        try:
            # Predict the next day's closing price
            predicted_price = predict_next_day(ticker=ticker, look_back=look_back)
            
            # Display the result with larger text and bigger box
            st.markdown(
                f"""
                <div style="background-color:#DFFFD8; padding:20px; border-radius:10px; text-align:center;">
                    <h2 style="color:#2E7D32;">Predicted Next Day's Closing Price: ${predicted_price:.2f}</h2>
                </div>
                """, 
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
