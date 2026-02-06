# NeuralTradeX

## AI-Powered Stock Intelligence & Forecasting Terminal

**NeuralTradeX** is an advanced AI-driven financial analytics platform that analyzes live stock market data, predicts future prices using machine learning and deep learning, and provides intelligent buy/sell recommendations through an interactive trading dashboard.

It simulates a real-world AI trading terminal similar to Bloomberg or Zerodha analytics systems.

---

# 🚀 Key Features

### 📡 Live Stock Data Integration

* Real-time stock data using Yahoo Finance API
* Supports global stocks (AAPL, TSLA, RELIANCE, etc.)
* Automatic historical data fetching

### 📊 Advanced Time Series Modeling

* Autoregressive (AR) model
* Moving Average (MA) model
* ARMA model
* Best model selection using RMSE

### 🤖 Machine Learning Prediction

* Linear regression based short-term forecasting
* Next-day and multi-day prediction
* Model performance evaluation

### 🧠 Deep Learning Forecasting

* LSTM neural network prediction
* Multi-day future stock price prediction
* Time-series learning from historical data

### 🕯️ Interactive Trading Charts

* Interactive candlestick chart (Plotly)
* Live stock price trend chart
* Zoom, pan and hover analytics
* Residual diagnostics visualization

### 💰 AI Trading Recommendation Engine

* Buy / Sell / Hold prediction
* Confidence score
* Model accuracy
* Predicted vs current price analysis

### 💬 AI Trading Assistant (Chat Interface)

Ask questions like:

```
Live price?
Next 5 days prediction?
Should I buy or sell?
LSTM forecast?
```

### 📉 Model Diagnostics

* Residual analysis
* RMSE comparison
* Best model selection
* Statistical summaries

---

# 🏗️ System Architecture

```
Live Stock API (Yahoo Finance)
          ↓
Data Preprocessing
          ↓
AR / MA / ARMA Models
          ↓
ML Regression Forecast
          ↓
LSTM Deep Learning Model
          ↓
Buy/Sell Recommendation Engine
          ↓
Interactive Dashboard (Flask)
          ↓
AI Chat Trading Assistant
```

---

# 💻 Tech Stack

### Programming

* Python
* Flask

### Data & ML

* Pandas
* NumPy
* Scikit-learn
* Statsmodels

### Deep Learning

* TensorFlow / Keras (LSTM)

### Visualization

* Plotly
* Matplotlib

### Data Source

* Yahoo Finance API (yfinance)

### Frontend

* HTML/CSS (Bloomberg-style UI)
* JavaScript (chat + dynamic updates)

---

# 📊 Supported Features

| Feature            | Description             |
| ------------------ | ----------------------- |
| Live stock price   | Real-time market data   |
| AR/MA/ARMA models  | Time series modeling    |
| LSTM prediction    | Deep learning forecast  |
| Buy/Sell engine    | AI trading decision     |
| Interactive charts | Candlestick + analytics |
| AI chat assistant  | Ask trading queries     |
| Model evaluation   | RMSE + diagnostics      |

---

# ⚙️ Installation & Setup

### 1. Clone repository

```
git clone https://github.com/yourusername/NeuralTradeX.git
cd NeuralTradeX
```

### 2. Install dependencies

```
pip install flask pandas numpy matplotlib scikit-learn statsmodels plotly yfinance tensorflow
```

### 3. Run application

```
python app.py
```

Open browser:

```
http://127.0.0.1:5000
```

Enter stock symbol → View dashboard.

---

# 📈 Example Stock Symbols

```
AAPL
TSLA
MSFT
GOOGL
RELIANCE.NS
TCS.NS
INFY.NS
```

---

# 🧪 Future Enhancements

* Reinforcement learning trading agent
* Portfolio optimization
* Multi-stock comparison
* News sentiment integration
* Crypto prediction module
* Cloud deployment

---

# 🏆 Project Highlights

* Real-time AI trading analytics
* Combines statistical + deep learning models
* Interactive financial dashboard
* Intelligent trading recommendations
* End-to-end financial AI system

---

# 👨‍💻 Developed By

Navin S
AI & Data Science Project — NeuralTradeX
Advanced Financial Intelligence System

---

# ⭐ If you like this project
Star ⭐ the repository and contribute.
