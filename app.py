from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os
import joblib  # for saving the scaler
import time
from requests.exceptions import HTTPError

app = Flask(__name__)
MODEL_PATH = "model/macd_model.keras"
SCALER_PATH = "model/scaler.save"

# --- Helper Functions ---
def fetch_stock_data(ticker):
    max_retries = 3
    base_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1} of {max_retries} to fetch data for {ticker}")
            
            # Add a small delay before each attempt
            time.sleep(base_delay * (attempt + 1))
            
            # Download data directly using download method
            data = yf.download(
                ticker,
                period='1y',
                interval='1d',
                progress=False,
                ignore_tz=True
            )
            
            if data.empty:
                raise ValueError(f"No data received for ticker {ticker}")
                
            print(f"Successfully downloaded {len(data)} rows of data for {ticker}")
            
            # Calculate indicators
            short_ema = data['Close'].ewm(span=12, adjust=False).mean()
            long_ema = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = short_ema - long_ema
            data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data.dropna(inplace=True)
            
            if len(data) < 30:
                raise ValueError(f"Not enough data points after calculating indicators. Got {len(data)} points, need at least 30.")
            
            return data
            
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                raise ValueError(f"Failed to download data for {ticker} after {max_retries} attempts. Please try again later.")
            continue

def prepare_features(data):
    X = data[['MACD', 'Signal']].values
    y = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)[:-1]
    X = X[:-1]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def build_and_train_model(X, y):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=8, verbose=0)
    return model

# --- Routes ---
@app.route("/", methods=['GET', 'POST'])
def index():
    prediction = None
    ticker = None
    error_message = None

    if request.method == 'POST':
        ticker = request.form['ticker'].upper()
        try:
            data = fetch_stock_data(ticker)
            
            # Debug information
            print(f"Data length: {len(data)}")
            print("First few rows of data:")
            print(data.head())
            print("\nData columns:", data.columns.tolist())

            # Defensive check: make sure there's enough usable data
            if data.empty or len(data) < 30:
                raise ValueError("Not enough historical data to calculate MACD and Signal. Try a different ticker.")

            X, y, scaler = prepare_features(data)

             # More checks in case prepare_features returns too little data
            if X.shape[0] == 0 or y.shape[0] == 0:
                raise ValueError("MACD/Signal values could not be computed. Try a different ticker.")

            # Train and save model ONCE if not already saved
            if not os.path.exists(MODEL_PATH):
                model = build_and_train_model(X, y)
                model.save(MODEL_PATH)
                joblib.dump(scaler, SCALER_PATH)
            else:
                model = tf.keras.models.load_model(MODEL_PATH)
                scaler = joblib.load(SCALER_PATH)

            latest_data = data[['MACD', 'Signal']].values[-1].reshape(1, -1)
            latest_scaled = scaler.transform(latest_data)
            prob = model.predict(latest_scaled)[0][0]
            prediction = "UP 📈" if prob > 0.5 else "DOWN 📉"
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction, ticker=ticker)


if __name__ == "__main__":
    app.run(debug=True)
