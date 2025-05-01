# Algorithmic Trading Project with Neural Networks

## Description

This project is a complete algorithmic trading system combining MetaTrader 5 (MQL5) for high-frequency data processing and Python for deep learning-based forecasting. It focuses on predicting the closing prices of Forex pairs (e.g., EUR/USD) using LSTM models. The system includes data preprocessing, technical indicator computation, feature selection, model training, and real-time socket-based communication between MQL5 and Python.

---

## Main Features

1. **Data Preparation**  
   - Collects 10-second candlestick data using tick aggregation.  
   - Calculates a wide range of technical indicators.  
   - Exports time-aligned datasets in CSV format.

2. **Modeling**  
   - LSTM architecture using TensorFlow/Keras.  
   - Supports iterative training and model persistence.

3. **Feature Selection**  
   - Uses RFECV with Random Forests to rank feature importance.

4. **Real-Time Communication**  
   - MQL5 sends live market indicators to Python over TCP sockets.  
   - Python returns a predicted signal: BUY, SELL, or WAIT.

5. **Evaluation & Visualization**  
   - Tracks performance metrics (e.g., RMSE).  
   - Graphs of predicted vs. actual prices using matplotlib.

---

## File Structure and Descriptions

### 🧠 Python (Machine Learning Side)

- **`dnn-MonoOutput-tensorflow.py`**  
  Contains the full training pipeline for a mono-output LSTM model. Loads preprocessed data, splits it into train/test sets, builds the LSTM model with Keras, trains it, evaluates performance, and saves the trained model.

- **`RFECV.py`**  
  Performs Recursive Feature Elimination with Cross-Validation using a Random Forest Regressor. Outputs a ranking of the most relevant features for model training.

- **`socket_server.py`**  
  Lightweight Python server listening on `127.0.0.1:9090`. Receives real-time feature data from MQL5, runs inference using the trained LSTM model, and returns trading decisions.

### 📈 MQL5 (Data Acquisition & Indicator Engine)

- **`main.mq5`**  
  Main entry point for the MQL5 script.  
  - Collects market data for specified dates/times.  
  - Computes technical indicators.  
  - Writes the data into `DataMQL5.csv`.  
  - Sends data via socket to Python and receives trading signals.

- **`indicators.mqh`**  
  Implements a comprehensive list of technical indicators in MQL5:  
  - Trend: ADX, AO, DEMA, Ichimoku  
  - Momentum: RSI, CCI, Momentum  
  - Volatility: ATR  
  - Volume: Bears/Bulls Power  
  - Oscillators: MACD, STOCH, RVI, Ultimate Oscillator  
  Each function returns time-series data in reverse chronological order (most recent first).

- **`market_data.mqh`**  
  Defines the logic to extract and build 10-second candle data from ticks.  
  Features include:  
  - Tick aggregation  
  - Custom fields like `variation_closeOpen`, `ticks_closeHigh`, etc.  
  - 6 candles per minute (10-second resolution)

- **`MetaData.mqh`**  
  Computes statistical metrics over time windows:  
  - Variance  
  - Covariance  
  - Correlation coefficient across multiple timeframes  
  Also retrieves the most correlated symbols with a given asset, used for multi-asset analysis.

- **`structure.mqh`**  
  Declares a custom struct `MyMqlRates`, an extension of the standard `MqlRates`.  
  Includes extra fields for:  
  - Ticks-based positioning  
  - Close-high/low variations  
  - Average prices  
  - Volume profile

- **`indicators.txt`, `market_data.txt`, `MetaData.txt`, `structure.txt`, `Main.txt`**  
  Raw versions of the `.mqh` files, possibly used for versioning or as legacy exports.

---

## Requirements

### MetaTrader 5
- MQL5 scripting enabled
- Access to tick-level market data

### Python 3.x
- `tensorflow`, `keras`  
- `scikit-learn`, `pandas`, `numpy`, `matplotlib`  
- `joblib`, `socket`, `csv`, etc.

---

## Installation

1. Copy all `.mqh` files to your MetaTrader 5 `Include` directory.
2. Start the Python server:
   ```bash
   python socket_server.py
   ```
3. Attach the MQL5 script to a chart in MetaTrader to begin data extraction and signal exchange.

---

## Usage

### 1. Train the Model

```bash
python dnn-MonoOutput-tensorflow.py
```

- Loads the dataset from `DataMQL5.csv`  
- Trains the model and saves it to disk

### 2. Predict in Real-Time

- Launch `socket_server.py`  
- Run the MQL5 script (`main.mqh`)  
- Predictions (BUY/SELL/WAIT) are returned and printed in MetaTrader's terminal

### 3. Feature Selection

```bash
python RFECV.py
```

- Displays and ranks the most impactful technical indicators for your model.
