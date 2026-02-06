from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from model import lstm_predict, run_models, forecast_future_ml, trading_recommendation,live_stock

app = Flask(__name__)

global_df=None
current_symbol=None
results=None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    global global_df, current_symbol, results

    current_symbol = request.form.get("symbol").upper()

    import yfinance as yf
    df = yf.download(current_symbol, period="2y", progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Open','High','Low','Close','Volume']]

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna()

    global_df = df
    results = run_models(df)
    results["trade"] = trading_recommendation(df)

    return render_template("dashboard.html", plots=results, symbol=current_symbol)

@app.route("/chat", methods=["POST"])
def chat():
    global global_df, results

    msg=request.json["message"].lower()

    if "best model" in msg:
        return jsonify({"reply":f"Best model is {results['best_model']} with lowest RMSE."})

    if "rmse" in msg:
        return jsonify({"reply":f"RMSE values: {results['rmse']}"})
    
    if "live" in msg:
        price=live_stock(current_symbol)
        return jsonify({"reply":f"📡 Live price of {current_symbol}: ₹{price}"})

    if "tomorrow" in msg:
        pred=forecast_future_ml(global_df,1)
        return jsonify({"reply":f"Predicted price tomorrow: ₹{pred[0]}"})
    
    if "lstm" in msg:
        pred=lstm_predict(global_df,7)
        return jsonify({"reply":f"LSTM next 7 days prediction: {pred}"})

    if "next" in msg:
        import re
        num=re.findall(r'\d+',msg)
        days=int(num[0]) if num else 5
        pred=forecast_future_ml(global_df,days)
        return jsonify({"reply":f"Next {days} days prediction: {pred}"})

    if "date" in msg:
        import re
        num=re.findall(r'\d+',msg)
        days=int(num[-1]) if num else 5
        pred=forecast_future_ml(global_df,days)
        return jsonify({"reply":f"Prediction after {days} days: ₹{pred[-1]}"})
    
    if "buy" in msg or "sell" in msg or "recommend" in msg:
        trade=trading_recommendation(global_df)
        return jsonify({"reply":
            f"Recommendation: {trade['recommendation']}\n"
            f"Current Price: ₹{trade['current_price']}\n"
            f"Predicted Next Price: ₹{trade['predicted_price']}\n"
            f"Model Accuracy: {trade['accuracy']}%\n"
            f"Confidence Level: {trade['confidence']}%"})

    return jsonify({"reply":"Ask: tomorrow price, next 10 days, best model, rmse, live"})
    
if __name__=="__main__":
    app.run(debug=True)
