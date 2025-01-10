# Stock-Prediction
# Problem Definition
This aims to predict the future stock price of a company based on historical stock data. The features used will include the previous days' stock prices (such as opening, closing, highest, and lowest prices), and predicting the closing price for the next day.

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Downloading stock data 
stock_symbol = 'AAPL'  # Can replace with any stock symbol (e.g., 'GOOG', 'AMZN')
start_date = '2010-01-01'
end_date = '2025-01-01'

# Downloading stock data
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Showing the first few rows of the dataset
stock_data.head()

# Step 3: Data Preprocessing

# Checking for missing values
print("Missing values:")
print(stock_data.isnull().sum())

# Since stock prices rarely have missing values, we can drop any rows with missing values
stock_data = stock_data.dropna()

# Feature Engineering: Use previous days' features to predict the next day's closing price
# Use the past 'n' days' data as features (here, we will use a simple lag of 1 day)
stock_data['Prev Close'] = stock_data['Close'].shift(1)  # Lag 1: Yesterday's closing price

# Remove the first row, as it will have a NaN value for the 'Prev Close' column
stock_data = stock_data.dropna()

# Step 4: Feature Selection
# We'll use 'Prev Close' as our feature and 'Close' as the target variable
X = stock_data[['Prev Close']]  # Feature: Previous day's close
y = stock_data['Close']  # Target: Today's close price

# Step 5: Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# Step 7: Model Training using Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Model Prediction
y_pred = model.predict(X_test)

# Step 9: Model Evaluation
# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse}')

# Step 10: Plotting the results

# Plot the actual vs predicted closing prices
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, color='blue', label='Actual Closing Price')
plt.plot(y_test.index, y_pred, color='red', label='Predicted Closing Price')
plt.title(f'{stock_symbol} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.show()

# Step 11: Predict the future closing price (Next day's price prediction)
latest_data = np.array([stock_data['Close'].iloc[-1]]).reshape(-1, 1)
latest_data_scaled = scaler.transform(latest_data)
predicted_price = model.predict(latest_data_scaled)

print(f"Predicted next day's closing price: ${predicted_price[0]:.2f}")
