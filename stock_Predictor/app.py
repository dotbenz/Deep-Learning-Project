import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ----------------------
# Streamlit Config
# ----------------------
st.set_page_config(page_title="Stock Prediction App", layout="wide")
st.title("📈 Stock Prediction Web App")

# ----------------------
# Select Stock
# ----------------------
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
           "META", "NVDA", "JPM", "NFLX", "INTC"]

ticker = st.selectbox("Select a stock", TICKERS)
years = st.slider("Years of data", 1, 10, 3)

# ----------------------
# Load Data
# ----------------------
data = yf.download(ticker, period=f"{years}y")

st.subheader(f"{ticker} - Raw Data (Last {years} years)")
st.dataframe(data.head(10))   # Show full data

st.subheader("Raw Data Description")
st.dataframe(data.describe())

# ----------------------
# Plot Close Price
# ----------------------
st.subheader("Closing Price History")
st.line_chart(data["Close"])

# ----------------------
# Preprocessing
# ----------------------
df = data[['Close']]
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df)

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(dataset, time_step=60):
    x, y = [], []
    for i in range(len(dataset) - time_step - 1):
        x.append(dataset[i:(i+time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(x), np.array(y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# ----------------------
# LSTM Model
# ----------------------
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=32, epochs=5, verbose=1)

# ----------------------
# Predictions
# ----------------------
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# ----------------------
# Plot Actual vs Predictions
# ----------------------
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(df.index, df['Close'], label="Actual Price")
train_range = range(time_step, len(train_predict)+time_step)
ax.plot(df.index[train_range], train_predict, label="Train Predict")
test_range = range(len(train_predict)+(time_step*2)+1, len(df)-1)
ax.plot(df.index[test_range], test_predict, label="Test Predict")
ax.legend()
st.subheader("Train/Test Predictions")
st.pyplot(fig)

# ----------------------
# Future Prediction (Next 30 Days)
# ----------------------
st.subheader("📊 Future Forecast (Next 30 Days)")

last_60_days = scaled_data[-time_step:]
future_input = last_60_days.reshape(1, time_step, 1)

future_predictions = []
n_future = 30

for _ in range(n_future):
    future_price = model.predict(future_input, verbose=0)   # shape (1,1)
    future_predictions.append(future_price[0,0])
    # reshape to (1,1,1) before appending
    future_price_reshaped = future_price.reshape(1,1,1)
    future_input = np.append(future_input[:,1:,:], future_price_reshaped, axis=1)

# Inverse transform
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1,1))

# Future dates
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=n_future)

# Plot Forecast
fig2, ax2 = plt.subplots(figsize=(12,6))
ax2.plot(df.index, df["Close"], label="Historical Price")
ax2.plot(future_dates, future_predictions, label="Future Prediction", linestyle="--")
ax2.legend()
st.pyplot(fig2)


# Show Forecast Data at the End
forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Close": future_predictions.flatten()
})

# Right justify only the Predicted_Close column
styled_df = forecast_df.style.set_properties(
    subset=["Predicted_Close"],
    **{"text-align": "left"}
).hide(axis="index")

st.subheader("📋 Forecast Data (Next 30 Days)")
st.dataframe(styled_df, use_container_width=True)


