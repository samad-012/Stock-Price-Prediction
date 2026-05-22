# STOCK PRICE PREDICTOR

# Import required libraries
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load historical stock data
stock_symbol = "AAPL"
data = yf.download(stock_symbol, start="2015-01-01", end="2025-01-01")

# Step 2: Preprocess the data
data = data[['Close']].dropna()
data['Prediction'] = data[['Close']].shift(-30)  # predict next 30 days

# Step 3: Split data into training/testing
X = np.array(data.drop(['Prediction'], axis=1))[:-30]
y = np.array(data['Prediction'])[:-30]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Model prediction
predictions = model.predict(X_test)

# Step 6: Evaluation
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Step 7: Visualization
plt.figure(figsize=(10,5))
plt.plot(y_test, label="Actual Prices", color='blue')
plt.plot(predictions, label="Predicted Prices", color='red')
plt.title(f"{stock_symbol} Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()
