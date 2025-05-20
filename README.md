# Stock-Market-Price-Pattern-Prediction-Model-

# LSTM-Based Stock Price & Pattern Prediction

**Stock Market Price Pattern Prediction Model** is a Python-based deep learning model built using LSTM (Long Short-Term Memory) networks to predict stock prices and detect technical patterns. It leverages historical stock data, technical indicators, and deep learning to make future price forecasts and recognize formations like **rounding bottoms** and **up flag patterns**.

## Features

* **Stock Price Prediction** using LSTM models trained on historical closing prices
* **Candlestick Charting** with moving average overlays (5, 13, 25-day EMAs)
* **Pattern Detection** for:

  * **Rounding Bottoms** (long-term bullish reversal)
  * **Up Flag Patterns** (short-term bullish continuation)
* Two versions of the LSTM model:

  * **Model 1**: Full-featured with early stopping
  * **Model 2**: Lightweight variant with Huber loss
* 📈 Visualizations for predictions vs. actuals and training history
* ⚙️ Modular code structure: easy to adapt for any stock ticker


## Tech Stack

* **Language**: Python
* **Libraries**:

  * `yfinance`, `numpy`, `pandas`, `matplotlib`, `plotly`
  * `scikit-learn`, `tensorflow`, `keras`
  * `mplfinance`, `scipy`



## Getting Started

### 1. Clone the repository:

```bash
git clone https://github.com/yourusername/StockPatternNet.git
cd StockPatternNet
```

### 2. Install required libraries:

```bash
pip install -r requirements.txt
```

### 3. Run the model:

```bash
python main.py  # or whichever file includes the main() function
```

---

## Sample Outputs

* Model Evaluation Metrics:

  * RMSE: \~4.73
  * MAE: \~3.71
  * MAPE: \~1.57%
  * R² Score: \~0.76

* Rounding Bottom and Flag Patterns identified from 2–10 years of data

---

## Disclaimer

This project is for **educational and research purposes only**. It is **not** financial advice and should not be used for real-world trading or investment decisions.

---

## Credits

Developed as part of an academic AI Lab project using publicly available financial data and open-source tools.


