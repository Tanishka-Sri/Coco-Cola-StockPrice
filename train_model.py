# train_model.py
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import pickle

# Load data
df = yf.download('KO', start='2010-01-01', end='2024-12-31')
data = df[['Close']]  # or df.filter(['Close'])

if data.empty:
    raise ValueError("Downloaded data is empty. Check ticker or internet connection.")

print("Data shape:", data.shape)
print(data.head())

# Normalize
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
# Prepare training data
train_len = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_len]

X_train, y_train = [], []
for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=32, epochs=5)

# Save model
model.save('model.h5')
