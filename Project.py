# @title LSTM MODEL 1

!pip install yfinance
!pip install mplfinance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import math
from datetime import datetime, timedelta


np.random.seed(42)


def get_stock_data(ticker, period='1y'):
    """Fetch stock data using yfinance"""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df


def plot_candlestick(data, title):
    """Create a candlestick chart"""

    plot_data = data.copy()


    mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', y_on_right=False)


    fig, axes = mpf.plot(
        plot_data,
        type='candle',
        style=s,
        title=title,
        volume=True,
        figsize=(12, 8),
        returnfig=True
    )


    plt.savefig('candlestick_chart.png')
    print(f"Candlestick chart for {title} created")


def prepare_data_for_lstm(data, feature='Close', look_back=60):
    """Prepare data for LSTM model"""

    dataset = data[feature].values.reshape(-1, 1)


    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)


    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])


    X, y = np.array(X), np.array(y)


    X = np.reshape(X, (X.shape[0], X.shape[1], 1))


    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, X_test, y_train, y_test, scaler


def build_lstm_model(X_train, y_train, X_test, y_test):
    """Build and train LSTM model"""

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))


    model.compile(optimizer='adam', loss='mean_squared_error')


    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )

    return model, history


def evaluate_model(model, X_test, y_test, scaler):
    """Evaluate the LSTM model"""

    predictions = model.predict(X_test)


    predictions_scaled = np.zeros((len(predictions), 1))
    predictions_scaled[:, 0] = predictions[:, 0]
    predictions = scaler.inverse_transform(predictions_scaled)

    y_test_scaled = np.zeros((len(y_test), 1))
    y_test_scaled[:, 0] = y_test
    y_test = scaler.inverse_transform(y_test_scaled)


    mse = mean_squared_error(y_test, predictions)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)


    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100


    print("\nModel Evaluation Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
    print(f"R² Score: {r2:.4f}")

    return predictions, y_test


def plot_predictions(predictions, actual, title):
    """Plot predictions vs actual values"""
    plt.figure(figsize=(12, 6))
    plt.plot(actual, color='blue', label='Actual Stock Price')
    plt.plot(predictions, color='red', label='Predicted Stock Price')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig('prediction_plot.png')
    print(f"Prediction plot for {title} created")


def main():

    ticker = 'AAPL'

    print(f"Analyzing stock data for {ticker}")


    stock_data = get_stock_data(ticker, period='2y')
    print(f"Downloaded {len(stock_data)} days of stock data")


    print("\nStock Data Preview:")
    print(stock_data.head())


    plot_candlestick(stock_data, f"{ticker} Stock Price")


    X_train, X_test, y_train, y_test, scaler = prepare_data_for_lstm(stock_data, feature='Close', look_back=60)
    print(f"\nData prepared for LSTM: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")


    print("\nTraining LSTM model...")
    model, history = build_lstm_model(X_train, y_train, X_test, y_test)


    predictions, actual = evaluate_model(model, X_test, y_test, scaler)


    plot_predictions(predictions, actual, f"{ticker} Stock Price Prediction")


    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')
    print("Training history plot created")

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()

# @title 5-13-25 DAY EMAs
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

# Function to fetch stock data
def get_stock_data(ticker, period='1y'):
    """Fetch stock data using yfinance"""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df

# Fetch stock data
data = get_stock_data('AAPL', period='2y')

# Calculate EMAs
data['EMA_5'] = data['Close'].ewm(span=5, adjust=False).mean()
data['EMA_13'] = data['Close'].ewm(span=13, adjust=False).mean()
data['EMA_25'] = data['Close'].ewm(span=25, adjust=False).mean()

# Create Candlestick Chart
fig = go.Figure()
fig.add_trace(go.Candlestick(x=data.index,
                             open=data['Open'],
                             high=data['High'],
                             low=data['Low'],
                             close=data['Close'],
                             name='Candlestick'))

# Add EMAs as line plots
fig.add_trace(go.Scatter(x=data.index, y=data['EMA_5'], mode='lines', name='5-Day EMA', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=data.index, y=data['EMA_13'], mode='lines', name='13-Day EMA', line=dict(color='red')))
fig.add_trace(go.Scatter(x=data.index, y=data['EMA_25'], mode='lines', name='25-Day EMA', line=dict(color='green')))

# Customize Layout
fig.update_layout(title='AAPL Candlestick Chart with EMAs',
                  xaxis_title='Date',
                  yaxis_title='Price (USD)',
                  xaxis_rangeslider_visible=False,
                  template='plotly_dark')

fig.show()

# @title 5-13-25 DAY EMAs (SEPARATE GRAPHS)
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

# Function to fetch stock data
def get_stock_data(ticker, period='1y'):
    """Fetch stock data using yfinance"""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df

# Fetch stock data
data = get_stock_data('AAPL', period='2y')

# Calculate EMAs
data['EMA_5'] = data['Close'].ewm(span=5, adjust=False).mean()
data['EMA_13'] = data['Close'].ewm(span=13, adjust=False).mean()
data['EMA_25'] = data['Close'].ewm(span=25, adjust=False).mean()

# Function to create candlestick chart with a specific EMA
def plot_candlestick_with_ema(data, ema_column, ema_name, color):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'],
                                 name='Candlestick'))
    fig.add_trace(go.Scatter(x=data.index, y=data[ema_column], mode='lines', name=ema_name, line=dict(color=color)))
    fig.update_layout(title=f'AAPL Candlestick Chart with {ema_name}',
                      xaxis_title='Date',
                      yaxis_title='Price (USD)',
                      xaxis_rangeslider_visible=False,
                      template='plotly_dark')
    fig.show()

# Plot separate charts for each EMA
plot_candlestick_with_ema(data, 'EMA_5', '5-Day EMA', 'blue')
plot_candlestick_with_ema(data, 'EMA_13', '13-Day EMA', 'red')
plot_candlestick_with_ema(data, 'EMA_25', '25-Day EMA', 'green')

# @title 10-20 DAY EMAs (OPTIONAL FOR CONSIDERATION)
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

# Function to fetch stock data
def get_stock_data(ticker, period='1y'):
    """Fetch stock data using yfinance"""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df

# Fetch stock data
data = get_stock_data('AAPL', period='2y')

# Calculate 10-day and 20-day EMAs
data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()

# Function to create candlestick chart with a specific EMA
def plot_candlestick_with_ema(data, ema_column, ema_name, color):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'],
                                 name='Candlestick'))
    fig.add_trace(go.Scatter(x=data.index, y=data[ema_column], mode='lines', name=ema_name, line=dict(color=color)))
    fig.update_layout(title=f'AAPL Candlestick Chart with {ema_name}',
                      xaxis_title='Date',
                      yaxis_title='Price (USD)',
                      xaxis_rangeslider_visible=False,
                      template='plotly_dark')
    fig.show()

# Plot separate charts for 10-day and 20-day EMAs
plot_candlestick_with_ema(data, 'EMA_10', '10-Day EMA', 'orange')
plot_candlestick_with_ema(data, 'EMA_20', '20-Day EMA', 'yellow')

# @title LSTM MODEL 2 (WITH LESS PARAMETERS - FOR REVIEW)
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from datetime import datetime, timedelta

# Function to fetch stock data
def get_stock_data(ticker, period='2y'):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df

# Fetch stock data
data = get_stock_data('AAPL', period='2y')

# Prepare data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

look_back = 60
X, y = [], []
for i in range(look_back, len(data_scaled)):
    X.append(data_scaled[i-look_back:i, 0])
    y.append(data_scaled[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='huber')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Predict prices
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Create DataFrame for plotting
pred_dates = data.index[-len(y_pred):]
pred_data = pd.DataFrame({'Date': pred_dates, 'Actual': y_test_actual.flatten(), 'Predicted': y_pred.flatten()})

# Plot Actual vs Predicted Candlestick Chart
fig = go.Figure()
fig.add_trace(go.Candlestick(x=pred_data['Date'],
                             open=pred_data['Actual'],
                             high=pred_data['Actual'] * 1.01,
                             low=pred_data['Actual'] * 0.99,
                             close=pred_data['Actual'],
                             name='Actual'))
fig.add_trace(go.Scatter(x=pred_data['Date'], y=pred_data['Predicted'], mode='lines', name='Predicted', line=dict(color='orange')))
fig.update_layout(title='AAPL Actual vs Predicted Prices (LSTM)',
                  xaxis_title='Date',
                  yaxis_title='Price (USD)',
                  xaxis_rangeslider_visible=False,
                  template='plotly_dark')
fig.show()

# @title ACCURACY - LSTM MODEL 2
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta
import math

# Function to fetch stock data
def get_stock_data(ticker, period='2y'):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df

# Fetch stock data
data = get_stock_data('AAPL', period='2y')

# Prepare data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

look_back = 60
X, y = [], []
for i in range(look_back, len(data_scaled)):
    X.append(data_scaled[i-look_back:i, 0])
    y.append(data_scaled[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Predict prices
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate evaluation metrics
mse = mean_squared_error(y_test_actual, y_pred)
rmse = math.sqrt(mse)
mae = mean_absolute_error(y_test_actual, y_pred)
r2 = r2_score(y_test_actual, y_pred)
mape = np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100

# Print metrics
print("\nModel Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
print(f"R² Score: {r2:.4f}")

# @title ROUNDING BOTTOM PATTERN PREDICTION
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from scipy.signal import argrelextrema

# Function to fetch stock data
def get_stock_data(ticker, period='10y'):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df

# Fetch stock data (at least 10 years to find 2-year patterns)
data = get_stock_data('AAPL', period='10y')

# Function to detect rounding bottom patterns
def detect_rounding_bottom(data, window=100):
    patterns = []
    prices = data['Close'].values
    local_min = argrelextrema(prices, np.less, order=window)[0]
    local_max = argrelextrema(prices, np.greater, order=window)[0]

    for i in range(len(local_min) - 1):
        start = local_min[i]
        end = local_min[i + 1]

        # Find high of spike before rounding bottom
        max_before_bottom = local_max[(local_max < start)]
        if len(max_before_bottom) == 0:
            continue
        spike_index = max_before_bottom[-1]  # Last max before bottom
        spike_high = prices[spike_index]  # High of spike

        # Ensure price stays below high of spike for at least 2 years
        if all(prices[start:end] < spike_high):
            target_price = 5 * spike_high  # Target as per image
            patterns.append((spike_index, start, end, spike_high, target_price))

    return patterns

# Detect patterns
patterns = detect_rounding_bottom(data)

# Plot the results using Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))

# Highlight detected rounding bottom patterns
for spike, start, end, spike_high, target in patterns:
    fig.add_vrect(
        x0=data.index[start], x1=data.index[end],
        fillcolor='green', opacity=0.3, line_width=0,
        annotation_text='Rounding Bottom', annotation_position='top left'
    )
    fig.add_hline(y=spike_high, line=dict(color="red", dash="dash"), name="Spike High")
    fig.add_hline(y=target, line=dict(color="blue", dash="dot"), name="Target Price")

fig.update_layout(title='AAPL Stock Price with Rounding Bottom Detection',
                  xaxis_title='Date',
                  yaxis_title='Price (USD)',
                  template='plotly_dark')
fig.show()

# @title DETECTED ROUNDING BOTTOM DATES

detected_patterns = []
if patterns:
    for spike, start, end, spike_high, target in patterns:
        detected_patterns.append({
            'Spike Date': data.index[spike],
            'Start Date': data.index[start],
            'End Date': data.index[end],
            'High of Spike': round(spike_high, 2),
            'Target Price (5X)': round(target, 2)
        })

# Convert to DataFrame and display
patterns_df = pd.DataFrame(detected_patterns)
patterns_df

# @title DETECTED UP FLAG PRICES

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from scipy.signal import argrelextrema

# Function to fetch stock data
def get_stock_data(ticker, period='2y'):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df

# Fetch stock data
data = get_stock_data('AAPL', period='2y')

# Detect Up Flag pattern
def detect_up_flag(data, window=20):
    patterns = []
    prices = data['Close'].values

    for i in range(window, len(prices) - window):
        segment = prices[i - window:i + window]
        start_of_pole = np.argmin(segment) + (i - window)
        high_of_flag = np.argmax(segment) + (i - window)
        low_of_flag = min(segment)

        if start_of_pole < high_of_flag:
            x = prices[high_of_flag] - prices[start_of_pole]
            target = low_of_flag + x  # Target formula
            patterns.append({'Start of Pole': start_of_pole, 'High of Flag': high_of_flag, 'Low of Flag': low_of_flag, 'Target': target})

    return patterns

up_flag_patterns = detect_up_flag(data)

# Display detected patterns
print("Up Flag Patterns:")
print(pd.DataFrame(up_flag_patterns))

# @title DETECTED UP FLAG PATTERN


# Plot candlestick chart with Up Flag pattern
fig = go.Figure()
fig.add_trace(go.Candlestick(x=data.index,
                             open=data['Open'],
                             high=data['High'],
                             low=data['Low'],
                             close=data['Close'],
                             name='Candlesticks'))

# Highlight detected Up Flag patterns
for pattern in up_flag_patterns:
    start, high, low, target = pattern['Start of Pole'], pattern['High of Flag'], pattern['Low of Flag'], pattern['Target']

    fig.add_trace(go.Scatter(x=[data.index[start]], y=[data['Close'].iloc[start]], mode='markers', marker=dict(color='blue', size=10), name='Start of Pole'))
    fig.add_trace(go.Scatter(x=[data.index[high]], y=[data['Close'].iloc[high]], mode='markers', marker=dict(color='green', size=10), name='High of Flag'))
    fig.add_trace(go.Scatter(x=[data.index[low]], y=[data['Close'].iloc[low]], mode='markers', marker=dict(color='red', size=10), name='Low of Flag'))
    fig.add_trace(go.Scatter(x=[data.index[low]], y=[target], mode='markers', marker=dict(color='purple', size=10), name='Target'))

fig.update_layout(title='Up Flag Pattern Detection', xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=True, template='plotly_dark')
fig.show()
