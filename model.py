import pandas as pd
import numpy as np

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

import yfinance as yf
import plotly.graph_objects as go

def normalize_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    required = ["Open", "High", "Low", "Close", "Volume"]
    df = df[required]

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna()


def interactive_price_chart(df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        name="Close Price",
        line=dict(color="#00ffc6", width=2)
    ))

    fig.update_layout(
        template="plotly_dark",
        title="Live Stock Price Movement",
        xaxis_title="Date",
        yaxis_title="Price",
        height=450
    )

    return fig.to_html(full_html=False)


def interactive_candlestick_chart(df):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        increasing_line_color="#00ff9f",
        decreasing_line_color="#ff4d6d"
    )])

    fig.update_layout(
        template="plotly_dark",
        title="Interactive Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=520
    )

    return fig.to_html(full_html=False)


def interactive_residual_plot(residuals):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=residuals,
        mode="lines",
        name="Residuals",
        line=dict(color="#ffaa55", width=2)
    ))

    fig.update_layout(
        template="plotly_dark",
        title="Residual Diagnostics (Model Errors)",
        xaxis_title="Time",
        yaxis_title="Residual Value",
        height=420
    )

    return fig.to_html(full_html=False)


def run_models(df):
    df = normalize_df(df)

    data = df["Close"]
    diff = data.diff().dropna()

    train_size = int(len(diff) * 0.8)
    train, test = diff[:train_size], diff[train_size:]

    ar = ARIMA(train, order=(2, 0, 0)).fit()
    ar_rmse = np.sqrt(mean_squared_error(test, ar.forecast(len(test))))

    ma = ARIMA(train, order=(0, 0, 2)).fit()
    ma_rmse = np.sqrt(mean_squared_error(test, ma.forecast(len(test))))

    arma = ARIMA(train, order=(2, 0, 2)).fit()
    arma_rmse = np.sqrt(mean_squared_error(test, arma.forecast(len(test))))

    rmse = {
        "AR": round(float(ar_rmse), 2),
        "MA": round(float(ma_rmse), 2),
        "ARMA": round(float(arma_rmse), 2)
    }

    best_model = min(rmse, key=rmse.get)

    return {
        "interactive_plot": interactive_price_chart(df),
        "interactive_candle": interactive_candlestick_chart(df),
        "interactive_residual": interactive_residual_plot(arma.resid),
        "ar_summary": str(ar.summary()),
        "ma_summary": str(ma.summary()),
        "arma_summary": str(arma.summary()),
        "best_model": best_model,
        "rmse": rmse
    }


def forecast_future_ml(df, days):
    df = normalize_df(df)
    data = df["Close"].values

    X, y = [], []
    for i in range(5, len(data)):
        X.append(data[i-5:i])
        y.append(data[i])

    X, y = np.array(X), np.array(y)

    model = LinearRegression()
    model.fit(X, y)

    last = data[-5:].tolist()
    preds = []

    for _ in range(days):
        p = model.predict([last])[0]
        preds.append(round(float(p), 2))
        last = last[1:] + [p]

    return preds


def trading_recommendation(df):
    df = normalize_df(df)
    data = df["Close"].values

    X, y = [], []
    for i in range(5, len(data)):
        X.append(data[i-5:i])
        y.append(data[i])

    X, y = np.array(X), np.array(y)

    model = LinearRegression()
    model.fit(X, y)

    accuracy = model.score(X, y) * 100
    last = data[-5:].tolist()

    predicted_price = float(model.predict([last])[0])
    current_price = float(data[-1])

    change = ((predicted_price - current_price) / current_price) * 100

    if change > 1:
        rec = "BUY 📈"
        confidence = min(abs(change) * 10, 95)
    elif change < -1:
        rec = "SELL 📉"
        confidence = min(abs(change) * 10, 95)
    else:
        rec = "HOLD 🤝"
        confidence = 60

    return {
        "recommendation": rec,
        "accuracy": round(accuracy, 2),
        "confidence": round(confidence, 2),
        "predicted_price": round(predicted_price, 2),
        "current_price": round(current_price, 2)
    }

def lstm_predict(df, days=7):
    df = normalize_df(df)

    if len(df) < 70:
        return ["Not enough data for LSTM"]

    data = df["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    last = scaled[-60:]
    preds = []

    for _ in range(days):
        pred = model.predict(last.reshape(1, 60, 1), verbose=0)[0][0]
        preds.append(pred)
        last = np.vstack([last[1:], [[pred]]])

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
    return [round(float(x), 2) for x in preds]


def live_stock(symbol):
    df = yf.download(symbol, period="1d", interval="1m", progress=False)
    df = normalize_df(df)
    return round(float(df["Close"].iloc[-1]), 2)
