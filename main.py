import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load Data
data = yf.download("AAPL", start="2015-01-01", end="2024-01-01")
data = data[['Close']]

# Plot
plt.plot(data)
plt.title("Stock Price")
plt.show()

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create dataset
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), 0])
        y.append(data[i+time_step, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data)
X = X.reshape(X.shape[0], X.shape[1], 1)

# LSTM Model 
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(60,1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=5, batch_size=32)

# Prediction
predicted = model.predict(X)
predicted = scaler.inverse_transform(predicted)

actual = data.values[60+1:]
# Evaluation
rmse = np.sqrt(mean_squared_error(actual, predicted))
print("RMSE:", rmse)

# Plot results
plt.plot(actual, label="Actual")
plt.plot(predicted, label="Predicted")
plt.legend()
plt.show()

# ARIMA
model_arima = ARIMA(data, order=(5,1,0))
model_arima_fit = model_arima.fit()
forecast = model_arima_fit.forecast(steps=30)

print("Next 30 days forecast:")
print(forecast) 