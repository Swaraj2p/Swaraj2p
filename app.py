from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime
import pytz  # For timezone handling

app = Flask(__name__)

# Function to get stock data from Yahoo Finance
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    hist['Date'] = hist.index
    return hist, stock.info['currency']

# Function to calculate technical indicators
def add_technical_indicators(data):
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['RSI'] = 100 - (100 / (1 + (data['Close'].diff(1).clip(lower=0).mean() / data['Close'].diff(1).clip(upper=0).mean())))
    data['MFI'] = (data['Close'] * data['Volume']).rolling(window=14).mean()
    data['STOCH'] = ((data['Close'] - data['Low'].rolling(window=14).min()) / (data['High'].rolling(window=14).max() - data['Low'].rolling(window=14).min())) * 100
    return data.dropna()

# Function to predict stock price for a specific date
def predict_price(data, date):
    data = add_technical_indicators(data)

    # Features: Using selected technical indicators
    X = data[['SMA_20', 'SMA_50', 'MACD', 'RSI', 'MFI', 'STOCH', 'Volume']].values
    y = data['Close'].values

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict for the provided date
    last_known_data = data.iloc[-1][['SMA_20', 'SMA_50', 'MACD', 'RSI', 'MFI', 'STOCH', 'Volume']].values.reshape(1, -1)
    predicted_price = model.predict(last_known_data)[0]

    # Return predicted price for the selected date
    return {
        'date': date,
        'price': predicted_price,
        'trend': 'up' if predicted_price > data['Close'].iloc[-1] else 'down'
    }

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.json['ticker']
    date = request.json['date']
    
    # Convert date string to datetime object (naive, without timezone)
    try:
        date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'error': 'Invalid date format'}), 400

    # Get the stock data
    stock_data, currency = get_stock_data(ticker)

    # Convert historical stock dates to naive dates (if timezone-aware)
    stock_data.index = stock_data.index.tz_localize(None)

    # Check if the input date is in the historical data range
    if stock_data.index[-1].date() < date < datetime.datetime.now().date():
        return jsonify({'error': 'Date out of range'}), 400

    # Predict the price for the given date
    future_price = predict_price(stock_data, date.strftime('%Y-%m-%d'))

    # Get the historical data (real-time values)
    historical_data = stock_data[['Date', 'Close']].tail(10).to_dict(orient='records')

    # Return historical data, predicted price, and currency
    return jsonify({
        'historical': historical_data,
        'predicted': future_price,
        'currency': currency
    })

if __name__ == '__main__':
    app.run(debug=True)
