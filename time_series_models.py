import logging
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

logging.basicConfig(
    filename="logs/time_series.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------- Data Fetch ----------------------
def fetch_stock_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        data = data[["Open", "High", "Low", "Close", "Volume"]]
        data.index.name = "Date"
        return data
    except Exception as e:
        logger.error(f"Error fetching stock data for {ticker}: {e}")
        return pd.DataFrame()


# ---------------------- Metrics ----------------------
def calculate_metrics(actual, predicted):
    try:
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        mape = (
            np.mean(np.abs((actual - predicted) / actual[actual != 0])) * 100
            if np.any(actual != 0)
            else np.nan
        )
        return {"RMSE": rmse, "MAE": mae, "MAPE": mape}
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}


# ---------------------- ARIMA ----------------------
def run_arima(df, steps=5):
    try:
        series = df["Close"]
        model = ARIMA(series, order=(5, 1, 0))
        fit = model.fit()
        forecast_res = fit.get_forecast(steps=steps)
        forecast = forecast_res.predicted_mean
        ci = forecast_res.conf_int(alpha=0.05)

        if not isinstance(forecast.index, pd.DatetimeIndex):
            forecast.index = df.index[-len(forecast):]
            ci.index = forecast.index

        metrics = calculate_metrics(series[-steps:], forecast[: len(series[-steps:])])
        return forecast, fit, ci.iloc[:, 0], ci.iloc[:, 1], metrics
    except Exception as e:
        logger.error(f"ARIMA error: {e}")
        return None, None, None, None, {}


# ---------------------- SARIMA ----------------------
def run_sarima(df, steps=5):
    try:
        series = df["Close"]
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        fit = model.fit(disp=False)
        forecast_res = fit.get_forecast(steps=steps)
        forecast = forecast_res.predicted_mean
        ci = forecast_res.conf_int(alpha=0.05)

        if not isinstance(forecast.index, pd.DatetimeIndex):
            forecast.index = df.index[-len(forecast):]
            ci.index = forecast.index

        metrics = calculate_metrics(series[-steps:], forecast[: len(series[-steps:])])
        return forecast, fit, ci.iloc[:, 0], ci.iloc[:, 1], metrics
    except Exception as e:
        logger.error(f"SARIMA error: {e}")
        return None, None, None, None, {}


# ---------------------- SARIMAX (with sentiment) ----------------------
def run_sarimax(df, sentiment_series=None, steps=5):
    try:
        series = df["Close"]
        if sentiment_series is None or sentiment_series.empty:
            sentiment_series = pd.Series([0] * len(series), index=series.index)

        model = SARIMAX(series, exog=sentiment_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        fit = model.fit(disp=False)

        future_sentiment = np.zeros((steps,))
        forecast_res = fit.get_forecast(steps=steps, exog=future_sentiment.reshape(-1, 1))
        forecast = forecast_res.predicted_mean
        ci = forecast_res.conf_int(alpha=0.05)

        if not isinstance(forecast.index, pd.DatetimeIndex):
            forecast.index = df.index[-len(forecast):]
            ci.index = forecast.index

        metrics = calculate_metrics(series[-steps:], forecast[: len(series[-steps:])])
        return forecast, fit, ci.iloc[:, 0], ci.iloc[:, 1], metrics
    except Exception as e:
        logger.error(f"SARIMAX error: {e}")
        return None, None, None, None, {}


# ---------------------- Prophet ----------------------
def run_prophet(df, steps=5):
    try:
        prophet_df = df.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
        model = Prophet()
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=steps)
        forecast = model.predict(future)

        pred = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].set_index("ds").tail(steps)
        if not isinstance(pred.index, pd.DatetimeIndex):
            pred.index = df.index[-len(pred):]

        metrics = calculate_metrics(prophet_df["y"].iloc[-steps:], pred["yhat"].iloc[: len(prophet_df["y"].iloc[-steps:])])
        return pred["yhat"], model, pred["yhat_lower"], pred["yhat_upper"], metrics
    except Exception as e:
        logger.error(f"Prophet error: {e}")
        return None, None, None, None, {}


# ---------------------- LSTM ----------------------
def run_lstm(df, steps=5, look_back=60):
    try:
        series = df["Close"].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(series)

        X, y = [], []
        for i in range(look_back, len(scaled)):
            X.append(scaled[i - look_back:i, 0])
            y.append(scaled[i, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)

        inputs = scaled[-look_back:]
        X_test = inputs.reshape((1, look_back, 1))
        preds_scaled = []
        for _ in range(steps):
            pred = model.predict(X_test, verbose=0)[0][0]
            preds_scaled.append(pred)
            inputs = np.append(inputs[1:], pred)
            X_test = inputs.reshape((1, look_back, 1))

        preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
        forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=steps, freq="B")
        forecast = pd.Series(preds, index=forecast_index)

        if not isinstance(forecast.index, pd.DatetimeIndex):
            forecast.index = df.index[-len(forecast):]

        metrics = calculate_metrics(df["Close"].iloc[-steps:], preds[: len(df["Close"].iloc[-steps:])])
        return forecast, model, None, None, metrics
    except Exception as e:
        logger.error(f"LSTM error: {e}")
        return None, None, None, None, {}


# ---------------------- Plotting ----------------------
def plot_forecast(df, forecast, lower=None, upper=None, title="Forecast"):
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df["Close"], label="Historical", color="blue")

        if forecast is not None:
            if not isinstance(forecast.index, pd.DatetimeIndex):
                if len(forecast) <= len(df):
                    forecast.index = df.index[-len(forecast):]
                else:
                    forecast.index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=len(forecast), freq="B")

            plt.plot(forecast.index, forecast.values, label="Forecast", linestyle="--", color="orange")

        if lower is not None and upper is not None:
            try:
                if not isinstance(lower.index, pd.DatetimeIndex):
                    lower.index = forecast.index
                    upper.index = forecast.index
                plt.fill_between(forecast.index, lower, upper, color="orange", alpha=0.2, label="Confidence Interval")
            except Exception as e:
                logger.error(f"Confidence interval plotting failed: {e}")

        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Price (INR)")
        plt.legend()
        plt.tight_layout()
        return plt
    except Exception as e:
        logger.error(f"plot_forecast error: {e}")
        return None
