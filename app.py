from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# --- Helper Functions ---
def fetch_stock_data(ticker):
    data = yf.download(ticker, period='6mo')
    short_ema = data['Close'].ewm(span=12, adjust=False).mean()
    long_ema = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = short_ema - long_ema
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data.dropna(inplace=True)
    return data

def prepare_features(data):
    X = data[['MACD', 'Signal']].values
    y = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)[:-1]  # 1: up, 0: down
    X = X[:-1]  # Align with y
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- Routes ---
@app.route("/", methods=['GET', 'POST'])
def index():
    prediction = None
    ticker = None

    if request.method == 'POST':
        ticker = request.form['ticker'].upper()
        try:
            data = fetch_stock_data(ticker)
            X, y, scaler = prepare_features(data)

            model = build_model()
            model.fit(X, y, epochs=10, batch_size=8, verbose=0)

            latest_data = data[['MACD', 'Signal']].values[-1].reshape(1, -1)
            latest_scaled = scaler.transform(latest_data)
            prob = model.predict(latest_scaled)[0][0]
            prediction = "UP" if prob > 0.5 else "DOWN"
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction, ticker=ticker)

if __name__ == "__main__":
    app.run(debug=True)
