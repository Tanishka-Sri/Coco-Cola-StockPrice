import streamlit as st
import base64
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import joblib
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ✅ Set Streamlit page config
st.set_page_config(page_title="Coke Meter", layout="centered")

# ✅ Function to add background image using base64
def add_bg_and_text_style(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: white;
        }}
        .markdown-text-container, .stTextInput > div > div > input,
        .stDateInput, .stDataFrame, .stText, .css-10trblm, .css-1v0mbdj {{
            color: black !important;
        }}
        .stButton>button {{
            color: white !important;
            background-color: #0000000;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# ✅ Apply the background image
add_bg_and_text_style("background.jpg")

# Title and description
st.title("🥤 CokeMeter")
st.markdown("Predict the next day's closing price for Coca-Cola (KO) stock using an LSTM model.")

# Sidebar inputs
ticker = st.sidebar.text_input("Enter stock ticker", value="KO")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))

# Load model and scaler
try:
    model = tf.keras.models.load_model("model.h5")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Download stock data
df = yf.download(ticker, start=start_date, end=end_date)

# If MultiIndex, flatten
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Ensure 'Close' column exists
if 'Close' not in df.columns or df.empty:
    st.error("No 'Close' data available. Try another ticker or date range.")
    st.stop()

data = df[['Close']].copy()

# Plot historical data
st.subheader(f"📈 Historical Close Price: {ticker}")
st.line_chart(data['Close'])

# Prediction logic
if len(data) >= 60:
    last_60 = data['Close'].values[-60:].reshape(-1, 1)
    last_60_scaled = scaler.transform(last_60)
    X_test = np.reshape(last_60_scaled, (1, 60, 1))
    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)
    st.subheader("🔮 Predicted Next Day Closing Price")
    st.success(f"${predicted_price[0][0]:.2f}")
else:
    st.warning("Not enough data to make a prediction (need at least 60 days).")

# Model summary
#with st.expander("📘 Model Details"):
    #model.summary(print_fn=lambda x: st.text(x))

# About
st.markdown("""
---
### ℹ️ How It Works
- Downloads latest **Coca-Cola (KO)** stock data from Yahoo Finance.
- Uses last 60 days of closing prices.
- Applies MinMaxScaler, feeds into LSTM model.
- Predicts next day closing price.
- Trained using historical stock price data.

""")

