# === 🔧 Standard Library ===
import os
import io
import datetime as dt  # Explicitly alias datetime to avoid conflicts
import logging

# === 📊 Core Libraries ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
import yfinance as yf
import feedparser
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error
from io import BytesIO
from sklearn.metrics import roc_curve, auc
import streamlit.components.v1 as components
import gymnasium as gym

# === ⚙️ Streamlit & Config ===
import streamlit as st
# Auto-refresh helper (optional)
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    # Fallback no-op if the package isn't installed (prevents hard crashes)
    def st_autorefresh(*args, **kwargs):
        return None

# ⬇️⬇️ NEW: unique-key helper for all Plotly renders ⬇️⬇️
from uuid import uuid4
def plotly_chart_unique(fig, prefix: str):
    """Render a Plotly figure with a guaranteed-unique Streamlit key."""
    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_{uuid4().hex}")
# ⬆️⬆️ NEW helper ends ⬆️⬆️

# === 📈 Forecasting Models ===
from time_series_models import (
    fetch_stock_data, run_arima, run_sarima, run_sarimax, run_prophet, run_lstm, plot_forecast
)
# fetch_google_news_sentiment function must use VADER
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from utils_live_sentiment import get_live_daily_sentiment, align_sentiment_to_index


# ---------- Stable download helpers (prevent state-loss on rerun) ----------
def put_download(slot: str, data_bytes: bytes, mime: str):
    """Save download bytes/mime in session_state to survive reruns."""
    store = st.session_state.setdefault("_downloads", {})
    store[slot] = {"data": data_bytes, "mime": mime}

def dl_button(slot: str, label: str, file_name: str, *, key: str):
    """Render a download button from stashed bytes (no recompute on rerun)."""
    blob = st.session_state.get("_downloads", {}).get(slot)
    if blob and blob.get("data"):
        st.download_button(
            label=label,
            data=blob["data"],
            file_name=file_name,
            mime=blob.get("mime", "application/octet-stream"),
            key=key,
        )
    else:
        st.caption("⏳ Preparing file… run the step above first.")


# === 🤖 ML Strategy & Evaluation ===
from ml_strategy import (
    fetch_ohlcv_data, add_technical_indicators, create_labels,
    train_xgboost_model, train_rf_model, train_logistic_model,
    simulate_ml_pnl, benchmark_models, run_ml_strategy, plot_trade_signals,
    compute_risk_metrics  
)

# === 🧠 Explainability ===
from ml_explainability import generate_explainability

# === 🎮 RL Agent ===
from rl_dqn_agent import train_dqn_agent, simulate_trading, evaluate_dqn_performance

# === 📰 News Sentiment ===
from news_sentiment_utils import fetch_google_news_sentiment

# === 📄 PDF Reporting ===
from report_generator import generate_html_report, html_to_pdf_bytes

# === 🕒 Timezone Support ===
import pytz

# Additional import for Telegram
import requests

# Function to send Telegram alert
def send_telegram_alert(message: str):
    import requests, logging
    token = st.secrets.get("TELEGRAM_TOKEN")
    chat_id = st.secrets.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logging.info("🔕 Telegram credentials not set; skipping notification.")
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        params = {"chat_id": chat_id, "text": message}
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            logger.info(f"Telegram alert sent: {message}")
        else:
            logger.warning(f"Telegram alert failed: {resp.text}")
    except Exception as e:
        logger.warning(f"Telegram send exception: {e}")

# Ensure log directory exists
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)

# Configure logging to file
logging.basicConfig(
    filename=os.path.join(log_dir, "app.log"),
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Example usage in your code
logger.info("Application started.")

# === Streamlit Config ===
st.set_page_config(page_title="AI Trading Strategy Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- 🎨 UI Styling ---
st.markdown("""
    <style>
    html, body, .stApp {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
        font-family: 'Segoe UI', sans-serif;
    }

    .stApp {
        background-color: #0d1117;
        color: #f0f6fc;
    }

    .css-1v0mbdj, .css-1d391kg {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(5px);
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    .stSlider > div {
        padding-top: 0.5rem;
        padding-bottom: 0.2rem;
    }

    h1 {
        font-size: 2.2rem;
        font-weight: bold;
    }

    .stButton>button, .stDownloadButton>button {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)
st.title("📊 AI-Based Algorithmic Trading Platform")

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ⚙️ Configuration Panel")

    # --- Top: Stock & Date Selection ---
    st.markdown("### 📊 Stock Selection")

    # Stock symbol input with placeholder instead of default value
    raw_symbol = st.text_input(
        "Enter Stock Symbol:",
        value="",  # keep empty so placeholder shows
        placeholder="RELIANCE",  # shows faded RELIANCE
        key="stock_input"
    ).upper().strip()

    # Check if user already provided a suffix (.NS or .BO)
    if raw_symbol.endswith(".NS") or raw_symbol.endswith(".BO"):
        selected_symbol = raw_symbol
        st.caption("📌 Detected full ticker format, skipping exchange selection.")
    elif raw_symbol:  # only show exchange if something is typed
        exchange = st.radio(
            "Select Exchange:",
            ["NSE", "BSE"],
            index=0,  # Default NSE
            horizontal=True,
            key="exchange_choice"
        )
        selected_symbol = f"{raw_symbol}.NS" if exchange == "NSE" else f"{raw_symbol}.BO"
    else:
        selected_symbol = None
        st.warning("⚠️ Please enter a stock symbol.")

    st.markdown("### 📅 Date Range")
    start_date = st.date_input("Start Date", value=dt.date(2022, 1, 1), key="start_date")
    end_date = st.date_input("End Date", value=dt.date.today(), key="end_date")

    # Validate Stock
    if selected_symbol:
        try:
            test_data = yf.Ticker(selected_symbol).history(period="1d")
            st.success(f"✅ Selected: {selected_symbol}")
        except Exception:
            st.error("❌ Could not validate symbol.")

    # --- Middle: Auto Refresh ---
    st.markdown("### 🔁 Auto Refresh")
    refresh_rate = st.slider("Auto Refresh (sec)", 0, 3000, 300, step=10)
    # Pause auto-refresh when long tasks are running
    if refresh_rate > 0 and not st.session_state.get("busy", False):
        st_autorefresh(interval=refresh_rate * 1000, key="refresh_timer")


    st.divider()

    # --- Reset Session State ---
    if st.button("Reset Session State"):
        for key in ["ml_df", "ml_equity", "ml_final_cash", "dqn_df", "dqn_worth", "dqn_log"]:
            if key in st.session_state:
                del st.session_state[key]
        st.success("✅ Session state cleared. Run ML and DQN strategies again.")

    # --- Roadmap ---
    with st.expander("📌 Project Roadmap"):
        st.markdown("""
        - ✅ Time-Series Forecasting  
        - ✅ ML Strategy  
        - ✅ DQN Agent  
        - ✅ News Sentiment  
        - ✅ Live Feed  
        - ☁️ Streamlit Cloud Deployment
        """)

# --- Helper Functions ---
def get_live_price(ticker):
    """
    Returns (last_trade_price, prev_close_for_change).
    Using previous close instead of today's open gives a non-zero % even after hours.
    """
    t = yf.Ticker(ticker)
    last = np.nan
    prev_close = np.nan
    try:
        intr = t.history(period="1d", interval="1m")
        if not intr.empty:
            last = float(intr["Close"].dropna().iloc[-1])
    except Exception:
        pass
    try:
        d2 = t.history(period="2d", interval="1d")
        if len(d2) >= 2:
            prev_close = float(d2["Close"].iloc[-2])
        elif len(d2) == 1:
            prev_close = float(d2["Close"].iloc[-1])
    except Exception:
        pass
    return (last if not np.isnan(last) else None,
            prev_close if not np.isnan(prev_close) else None)


def is_market_open():
    logger = logging.getLogger(__name__)
    try:
        # Use explicitly imported datetime with timezone
        now = dt.datetime.now(pytz.timezone("Asia/Kolkata"))
        logger.debug(f"Current time: {now.strftime('%H:%M:%S')} IST")
        weekday = now.weekday()
        if weekday >= 5:  # Saturday (5) or Sunday (6)
            logger.info("Market closed: Weekend")
            return False
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        logger.debug(f"Market hours: {market_open.strftime('%H:%M:%S')} to {market_close.strftime('%H:%M:%S')}")
        return market_open <= now <= market_close
    except Exception as e:
        logger.error(f"Error in is_market_open: {str(e)}")
        return False

def generate_comparison_summary(ml_result_df, ml_accuracy, dqn_worths, dqn_trades):
    ml_buys = ml_result_df[ml_result_df["Signal"] == 1].shape[0]
    ml_sells = ml_result_df[ml_result_df["Signal"] == -1].shape[0]
    dqn_buys = dqn_trades[dqn_trades["type"] == "Buy"].shape[0]
    dqn_sells = dqn_trades[dqn_trades["type"] == "Sell"].shape[0]
    final_net_worth = dqn_trades["net_worth"].iloc[-1] if not dqn_trades.empty else 10000

    summary_df = pd.DataFrame({
        "Strategy": ["ML Strategy", "DQN Agent"],
        "Accuracy / Net Worth": [f"{ml_accuracy:.2%}", f"₹{final_net_worth:,.2f}"],
        "Buy Trades": [ml_buys, dqn_buys],
        "Sell Trades": [ml_sells, dqn_sells]
    })
    return summary_df

# ================= Utility: Safe Download =================
def safe_download(label, filepath=None, default_name=None, mime_type="application/octet-stream", data_bytes=None, show_warning=False):
    """Utility to safely create download buttons with file existence checks or direct bytes"""
    import os, streamlit as st, logging
    if data_bytes is not None:
        st.download_button(
            label=label,
            data=data_bytes,
            file_name=os.path.basename(default_name) if default_name else "download.bin",
            mime=mime_type
        )
    elif filepath and os.path.exists(filepath):
        with open(filepath, "rb") as f:
            file_bytes = f.read()
        st.download_button(
            label=label,
            data=file_bytes,
            file_name=os.path.basename(default_name) if default_name else os.path.basename(filepath),
            mime=mime_type
        )
    else:
        if show_warning:
            st.warning(f"⚠️ File {filepath or default_name} not found. Download unavailable.")
        logging.warning(f"File {filepath or default_name} not found")


# --- Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Forecasting",         # tab1
    "🧠 ML Strategy",         # tab2
    "🎮 DQN Strategy",        # tab3
    "📊 Comparison",          # tab4
    "📰 News Sentiment",      # tab5
    "📡 Live Feed"            # tab6
])
# --- Tab 1: Forecasting ---
with tab1:
    st.markdown("## 📈 Forecasting Module")
    st.markdown("Use time series models to predict future stock prices.")

    # (NEW) Toggle to suppress SARIMAX diagnostics without deleting code
    SHOW_SARIMAX_DIAGNOSTICS = False

    # --- Show prior downloads ONLY if we have them (side-by-side) ---
    if ("last_forecast_model" in st.session_state) and ("_downloads" in st.session_state):
        prev_model = st.session_state["last_forecast_model"]
        has_prev_csv = bool(st.session_state["_downloads"].get("forecast_csv"))
        has_prev_metrics = bool(st.session_state["_downloads"].get("forecast_metrics_csv"))
        if has_prev_csv or has_prev_metrics:
            c_prev_1, c_prev_2 = st.columns(2)
            if has_prev_csv:
                with c_prev_1:
                    dl_button(
                        "forecast_csv",
                        f"📥 Previous {prev_model} – Download Forecast CSV",
                        "forecast.csv",
                        key="dl_forecast_csv_rerun",
                    )
            if has_prev_metrics:
                with c_prev_2:
                    dl_button(
                        "forecast_metrics_csv",
                        f"📊 Previous {prev_model} – Download Metrics CSV",
                        "forecast_metrics.csv",
                        key="dl_forecast_metrics_csv_rerun",
                    )

    col1, col2 = st.columns(2)
    with col1:
        forecast_model = st.selectbox(
            "Select Forecasting Model",
            ["ARIMA", "SARIMA", "SARIMAX", "Prophet", "LSTM"],
            key="forecast_model",
        )
        forecast_days = st.slider("Forecast Days", 1, 15, 5, key="forecast_days")
    with col2:
        chart_mode = st.radio(
            "Chart Type", ["📊 Plotly (Interactive)", "🖼️ Matplotlib (Static)"], key="chart_mode"
        )
        export_svg = st.checkbox("📤 Export as SVG instead of PNG", key="export_svg")
        enable_sentiment = (
            st.checkbox("Include Sentiment Analysis", value=False, key="sentiment_toggle")
            if forecast_model == "SARIMAX"
            else False
        )

    run_forecast = st.button("🔮 Run Forecast", type="primary", key="run_forecast")

    if run_forecast:
        try:
            st.info(f"🔍 Running {forecast_model} Forecast for **{selected_symbol}**...")
            logger.info(f"Fetching data for {selected_symbol} from {start_date} to {end_date}")
            df = fetch_stock_data(selected_symbol, start=start_date, end=end_date)

            if df.empty:
                st.error("❌ No data found in selected range. Please adjust the date range or ticker.")
                logger.error(f"No data for {selected_symbol} from {start_date} to {end_date}")
                st.stop()

            # Debug print DataFrame details
            st.write(f"DataFrame head:\n{df.head().to_markdown()}")

            # Validate data for LSTM
            if forecast_model == "LSTM" and len(df) < 60:
                st.error(
                    f"❌ Insufficient data points ({len(df)}) for LSTM. Requires at least 60 days."
                )
                logger.error(
                    f"Insufficient data points ({len(df)}) for LSTM. Requires at least 60 days."
                )
                st.stop()

            # Optional sentiment series for SARIMAX
            sentiment_input = None
            if forecast_model == "SARIMAX" and enable_sentiment:
                from utils_live_sentiment import get_live_daily_sentiment, align_sentiment_to_index
                with st.spinner("📰 Pulling live news sentiment…"):
                    sent_daily, sent_source = get_live_daily_sentiment(selected_symbol, lookback_days=30)

                if isinstance(sent_daily, pd.Series) and not sent_daily.empty:
                    # Standardize (z-score) to add variation & avoid scale issues
                    s_mean = float(sent_daily.mean()) if np.isfinite(sent_daily.mean()) else 0.0
                    s_std  = float(sent_daily.std())  if np.isfinite(sent_daily.std()) and sent_daily.std() != 0 else 1.0
                    s_z = (sent_daily - s_mean) / s_std

                    # Align to price index (business days); remaining gaps -> 0.0
                    sentiment_input = align_sentiment_to_index(s_z, df.index)

                    # Diagnostics for you
                    st.caption(
                        f"🕒 Sentiment source: {sent_source} | span={len(sent_daily)} | "
                        f"mean={s_mean:+.3f} | std={float(sent_daily.std()):.3f}"
                    )

                    # (CHANGED) Show last 14 days as a TABLE instead of dropdown
                    try:
                        lastN = sent_daily.sort_index().tail(14)
                        df_sent_prev = pd.DataFrame({
                            "Date": [d.strftime("%Y-%m-%d") for d in lastN.index],
                            "Sentiment": [float(v) for v in lastN.values],
                        })
                        st.dataframe(df_sent_prev, use_container_width=True)
                        # Store for reuse elsewhere if needed
                        st.session_state["sarimax_sent_series"] = lastN
                        st.session_state["sarimax_sent_source"] = sent_source
                    except Exception:
                        pass

                    # If still flat after alignment (e.g., only 1 news date), blend with price returns
                    if float(np.nanstd(np.asarray(sentiment_input, dtype=float))) == 0.0:
                        # Keeping your existing behavior, but suppressing noisy UI messages
                        rets = pd.Series(df['Close']).pct_change().fillna(0.0)
                        r_mean = float(rets.mean()) if np.isfinite(rets.mean()) else 0.0
                        r_std  = float(rets.std())  if np.isfinite(rets.std()) and rets.std() != 0 else 1.0
                        rets_z = (rets - r_mean) / r_std
                        rets_aligned = align_sentiment_to_index(rets_z, df.index)
                        sentiment_input = 0.5 * sentiment_input + 0.5 * rets_aligned
                else:
                    # Keep logic but avoid extra warnings in UI
                    sentiment_input = None

            # Run model
            forecast_result = model_fit = lower = upper = None
            metrics = {}
            with st.spinner(f"Running {forecast_model} model..."):
                if forecast_model == "ARIMA":
                    forecast_result, model_fit, lower, upper, metrics = run_arima(
                        df, steps=forecast_days
                    )
                elif forecast_model == "SARIMA":
                    forecast_result, model_fit, lower, upper, metrics = run_sarima(
                        df, steps=forecast_days
                    )
                elif forecast_model == "SARIMAX":
                    forecast_result, model_fit, lower, upper, metrics = run_sarimax(
                        df, sentiment_input, steps=forecast_days
                    )
                elif forecast_model == "Prophet":
                    logger.info("Starting Prophet forecast")
                    forecast_result, model_fit, lower, upper, metrics = run_prophet(
                        df, steps=forecast_days
                    )
                elif forecast_model == "LSTM":
                    logger.info("Starting LSTM forecast")
                    # LSTM typically returns only the forecast series; CI is not available
                    forecast_result, model_fit, _, _, metrics = run_lstm(
                        df, steps=forecast_days
                    )
                    lower, upper = None, None  # explicitly clear CI for LSTM

            if forecast_result is None or model_fit is None:
                st.error("🚨 Forecasting model failed to generate results.")
                logger.error(f"{forecast_model} failed to generate results")
                st.stop()
            
            
            # --- SARIMAX exogenous diagnostics (whether it actually used sentiment) ---
            if forecast_model == "SARIMAX":
                used_exog = isinstance(sentiment_input, (pd.Series, np.ndarray)) and (
                    np.nanstd(np.asarray(sentiment_input, dtype=float)) > 1e-12
                )
                beta_msg = "—"
                try:
                    if used_exog and hasattr(model_fit, "params"):
                        # Try to locate the exogenous coefficient name
                        # Common keys include 'exog', 'x1', or param names that contain 'exog'
                        for k, v in (model_fit.params.items() if hasattr(model_fit.params, "items") else enumerate(model_fit.params)):
                            name = k if isinstance(k, str) else str(k)
                            if ("exog" in name.lower()) or ("x" in name.lower()):
                                beta_msg = f"{float(v):+.4f}"
                                break
                except Exception:
                    pass
                # (SUPPRESSED) keep code but don't show the message
                if SHOW_SARIMAX_DIAGNOSTICS:
                    st.info(f"SARIMAX exogenous sentiment used: {'Yes ✅' if used_exog else 'No (flat/missing) ⚠️'} | β_sent: {beta_msg}")


            # Ensure forecast index is forward-looking business days (no weekends/holidays)
            last_hist_date = pd.to_datetime(df.index[-1])
            bdays = pd.bdate_range(
                start=last_hist_date + pd.Timedelta(days=1), periods=len(forecast_result)
            )
            # Build forecast DataFrame
            forecast_df = pd.DataFrame({"Forecast": np.asarray(forecast_result)}, index=bdays)

            chart_filename = (
                f"{selected_symbol}_{forecast_model}_Forecast.png"
                if not export_svg
                else f"{selected_symbol}_{forecast_model}_Forecast.svg"
            )

            # ---------- Chart ----------
            st.markdown("### 📊 Forecast Chart")
            if chart_mode.startswith("📊"):
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(x=df.index, y=df["Close"], name="Historical", line=dict(color="blue"))
                )
                fig.add_trace(
                    go.Scatter(
                        x=forecast_df.index,
                        y=forecast_df["Forecast"],
                        name="Forecast",
                        line=dict(color="orange"),
                    )
                )
                if lower is not None and upper is not None:
                    # Align CI to business-day index length if needed
                    lo = pd.Series(lower, index=forecast_df.index)[: len(forecast_df)]
                    up = pd.Series(upper, index=forecast_df.index)[: len(forecast_df)]
                    fig.add_trace(
                        go.Scatter(
                            x=list(forecast_df.index) + list(forecast_df.index[::-1]),
                            y=list(up.values) + list(lo.values[::-1]),
                            fill="toself",
                            fillcolor="rgba(255,165,0,0.2)",
                            line=dict(color="rgba(255,255,255,0)"),
                            name="Confidence Interval",
                            showlegend=True,
                        )
                    )
                fig.update_layout(
                    title=f"{selected_symbol} – {forecast_model} Forecast",
                    xaxis_title="Date",
                    yaxis_title="Price (INR)",
                    template="plotly_white",
                    height=500,
                )
                plotly_chart_unique(fig, "forecast_main")
                st.caption("💡 Use the camera icon on the chart to export as PNG.")
            else:
                plt.figure(figsize=(10, 5))
                plt.plot(df.index, df["Close"], label="Historical", color="steelblue")
                plt.plot(
                    forecast_df.index,
                    forecast_df["Forecast"],
                    label="Forecast",
                    linestyle="--",
                    color="orange",
                )
                if lower is not None and upper is not None:
                    lo = pd.Series(lower, index=forecast_df.index)[: len(forecast_df)]
                    up = pd.Series(upper, index=forecast_df.index)[: len(forecast_df)]
                    plt.fill_between(
                        forecast_df.index, lo.values, up.values, color="orange", alpha=0.2, label="Confidence Interval"
                    )
                plt.title(f"{selected_symbol} – {forecast_model} Forecast")
                plt.xlabel("Date")
                plt.ylabel("Price (INR)")
                plt.legend()
                plt.tight_layout()
                plt.savefig(chart_filename)
                plt.close()
                st.image(chart_filename, caption="Matplotlib Forecast", use_container_width=True)
                logging.info(f"Saved forecast chart to {chart_filename}")

            # ---------- Metrics ----------
            st.markdown("### 📈 Forecast Accuracy Metrics")
            rmse = mae = mape = np.nan  # defaults to avoid 'not defined' on failure
            try:
                if len(df["Close"]) >= forecast_days:
                    # Compare last available historical closes to first N forecast points
                    actuals = df["Close"].iloc[-forecast_days:].values.astype(float)
                    preds = forecast_df["Forecast"].iloc[:forecast_days].values.astype(float)
                    if len(actuals) > 0 and len(preds) > 0 and len(actuals) == len(preds):
                        rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
                        mae = float(mean_absolute_error(actuals, preds))
                        # avoid division by zero for MAPE
                        nz = actuals != 0
                        if np.any(nz):
                            mape = float(np.mean(np.abs((actuals[nz] - preds[nz]) / actuals[nz])) * 100)
                        col1, col2, col3 = st.columns(3)
                        col1.metric("RMSE", f"{rmse:.2f}")
                        col2.metric("MAE", f"{mae:.2f}")
                        col3.metric("MAPE", "—" if np.isnan(mape) else f"{mape:.2f}%")
                    else:
                        st.warning("Cannot calculate accuracy metrics: Insufficient data for comparison.")
                else:
                    st.warning("Cannot calculate accuracy metrics: Not enough historical data.")
            except Exception as _merr:
                logging.warning(f"Metrics computation issue: {_merr}")
                st.warning("Cannot calculate accuracy metrics due to a computation issue.")

            # ---- SARIMAX diagnostics: show whether exogenous sentiment was used
            if forecast_model == "SARIMAX" and isinstance(metrics, dict) and "exog_used" in metrics:
                used = "Yes ✅" if metrics.get("exog_used") else "No (flat/missing) ⚠️"
                beta = metrics.get("beta_sent")
                beta_txt = "—" if (beta is None or (isinstance(beta, float) and np.isnan(beta))) else f"{beta:.3f}"
                # (SUPPRESSED) keep code but don't show the message
                if SHOW_SARIMAX_DIAGNOSTICS:
                    st.caption(f"SARIMAX exogenous sentiment used: **{used}** | β_sent: **{beta_txt}**")

            # ---------- Downloads (Current + stash for reruns) ----------
            export_df = pd.DataFrame(
                {
                    "Date": forecast_df.index,
                    "Forecast": forecast_df["Forecast"].values,
                    "Lower CI": (pd.Series(lower, index=forecast_df.index).values if lower is not None else None),
                    "Upper CI": (pd.Series(upper, index=forecast_df.index).values if upper is not None else None),
                }
            )

            # Bytes
            forecast_csv_bytes = export_df.to_csv(index=False).encode()
            metrics_csv_bytes = (
                pd.DataFrame(
                    {"Metric": ["RMSE", "MAE", "MAPE"], "Value": [rmse, mae, mape]}
                )
                .to_csv(index=False)
                .encode()
            )

            # Stash for stable reruns
            put_download("forecast_csv", forecast_csv_bytes, "text/csv")
            put_download("forecast_metrics_csv", metrics_csv_bytes, "text/csv")
            st.session_state["last_forecast_model"] = forecast_model

            # Show CURRENT model downloads side-by-side
            c_now_1, c_now_2 = st.columns(2)
            with c_now_1:
                dl_button(
                    "forecast_csv",
                    f"📥 Current {forecast_model} – Download Forecast CSV",
                    "forecast.csv",
                    key="dl_forecast_csv_now",
                )
            with c_now_2:
                dl_button(
                    "forecast_metrics_csv",
                    f"📊 Current {forecast_model} – Download Metrics CSV",
                    "forecast_metrics.csv",
                    key="dl_forecast_metrics_csv_now",
                )

            # Matplotlib chart file download only if we actually saved one
            if (not chart_mode.startswith("📊")) and os.path.exists(chart_filename):
                with open(chart_filename, "rb") as f:
                    st.download_button(
                        label=f"🖼️ Current {forecast_model} – Download Chart ({'PNG' if not export_svg else 'SVG'})",
                        data=f,
                        file_name=chart_filename,
                        mime="image/png" if not export_svg else "image/svg+xml",
                        key="dl_forecast_chart_file",
                    )

        except Exception as e:
            st.error(f"🚨 Forecasting Error: {e}")
            logging.error(f"Forecasting error: {e}")

# --- Tab 2: ML Strategy ---
with tab2:
    st.markdown("## 📊 Machine Learning Strategy")
    st.markdown("Train ML models to generate trading signals and analyze performance.")
    
    col1, col2 = st.columns(2)
    with col1:
        ml_model = st.selectbox("Select ML Model", ["Random Forest", "Logistic Regression", "XGBoost"], key="ml_model")
        initial_cash = st.number_input("Initial Cash", min_value=1000, value=10000, step=1000, key="ml_cash")
    with col2:
        train_split = st.slider("Training Split (%)", 50, 90, 80, key="ml_split")
        enable_explainability = st.checkbox("Enable Explainability", value=True, key="ml_explain")
    with st.expander("⚙️ Technical indicators", expanded=True):
        indicator_options = ["MA10","MA50","SMA100","EMA20","EMA200","RSI","MACD","BBANDS","ADX","PIVOTS"]
        selected_indicators = st.multiselect(
            "Choose indicators to compute",
            indicator_options,
            default=["MA10","MA50","RSI","MACD","EMA20","BBANDS"]
        )

    run_ml = st.button("🚀 Run ML Strategy", type="primary", key="run_ml")
    
    if run_ml:
        st.session_state["busy"] = True
         # If user picked XGBoost, ensure it’s importable before we proceed
        if ml_model == "XGBoost":
            try:
                import xgboost as _xgb  # noqa: F401
            except Exception as _xgb_err:
                st.error("❌ XGBoost is not installed or failed to import. Install with: `pip install xgboost`.")
                logging.error(f"XGBoost import failed: {_xgb_err}")
                st.session_state["busy"] = False
                st.stop()
        try:
            st.info(f"🔍 Running {ml_model} Strategy for **{selected_symbol}**...")
            with st.spinner("📥 Fetching data and running ML strategy..."):
                df, final_cash, equity_curve, trained_model = run_ml_strategy(
                    selected_symbol, start=start_date, end=end_date, model=ml_model,
                    initial_cash=initial_cash, train_size=train_split / 100, indicators=selected_indicators
                )
            if df is None or df.empty:
                st.error("❌ ML Strategy failed. Check data or parameters.")
                logging.error("run_ml_strategy returned None or empty DataFrame")
                st.stop()
            
            # Store results
            st.session_state.ml_df = df
            st.session_state.ml_equity = equity_curve
            st.session_state.ml_final_cash = final_cash
            st.session_state.ml_model_obj = trained_model
            st.session_state.ml_model_type = ml_model

            # Define feature columns (aligned with ml_strategy.py)
            expected_features = ["MA10", "MA50", "RSI", "MACD"]
            feature_cols = [f for f in expected_features if f in df.columns]
            st.session_state.ml_features = feature_cols

            # ================= Metrics =================
            train_size = int(len(df) * (train_split / 100))
            test_df = df[train_size:]
            accuracy = precision = recall = f1 = auc = 0.0
            cm, y_true_bin = None, None
            total_samples = 0
            try:
                if not test_df.empty and 'Signal' in test_df and 'Prediction' in test_df:
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
                    from sklearn.preprocessing import label_binarize
                    y_true = test_df['Signal'].values
                    y_pred = test_df['Prediction'].values
                    accuracy = accuracy_score(y_true, y_pred)
                    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                    total_samples = len(y_true)
                    cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
            except Exception as e:
                logging.error(f"Metrics computation failed: {e}")

            # ================= Trade Signals =================
            st.markdown("### 📈 Trade Signals")
            chart_filename = f"{selected_symbol}_{ml_model}_Signals.png"
            fig = plot_trade_signals(df, selected_symbol, ml_model, chart_type=chart_mode)
            plotly_chart_unique(fig, "ml_signals")

            import plotly.io as pio
            chart_saved = False
            try:
                # Try Plotly export first
                pio.write_image(fig, chart_filename, engine="kaleido", width=800, height=600)
                chart_saved = True
            except Exception as e:
                logging.error(f"Plotly save failed: {e}. Falling back to Matplotlib.")
                # --- Fallback: Matplotlib ---
                try:
                    import matplotlib.pyplot as plt
                    fig_mat = plot_trade_signals(df, selected_symbol, ml_model, chart_type="matplotlib")
                    fig_mat.savefig(chart_filename, dpi=150, bbox_inches="tight")
                    plt.close(fig_mat)
                    chart_saved = True
                except Exception as e2:
                    logging.error(f"Matplotlib fallback also failed: {e2}")

            safe_download(
                label="📥 Download Signals Chart",
                filepath=chart_filename if chart_saved else None,
                default_name=chart_filename,
                mime_type="image/png",
                show_warning=False
            )

            # ================= Equity Curve =================
            st.markdown("### 💰 Equity Curve")
            equity_filename = f"{selected_symbol}_{ml_model}_Equity.png"
            # Risk metrics (daily frequency assumed)
            if equity_curve:
                equity_series = pd.Series(equity_curve)
                metrics = compute_risk_metrics(equity_series, periods_per_year=252)
                c1, c2, c3 = st.columns(3)
                c1.metric("Sharpe",  f"{metrics['sharpe']:.2f}")
                c2.metric("Sortino", f"{metrics['sortino']:.2f}")
                c3.metric("Max DD",  f"{metrics['max_dd']:.2%}")

            if equity_curve and len(equity_curve) > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["Date"][-len(equity_curve):], y=equity_curve, name="Equity"))
                plotly_chart_unique(fig, "ml_equity")
                try:
                    import plotly.io as pio
                    pio.write_image(fig, equity_filename, engine="kaleido", width=800, height=600)
                    equity_saved = True
                except:
                    equity_saved = False
                safe_download(
                    label="📥 Download Equity Curve",
                    filepath=equity_filename if equity_saved else None,
                    default_name=equity_filename,
                    mime_type="image/png",
                    show_warning=False
                )

            # ================= Performance Metrics =================
            st.markdown("### 📊 Performance Metrics")
            returns = (final_cash - initial_cash) / initial_cash * 100 if equity_curve else 0.0
            col1, col2, col3 = st.columns(3)
            col1.metric("Final Portfolio Value", f"₹{final_cash:,.2f}")
            col2.metric("Return (%)", f"{returns:.2f}%")
            col3.metric("Accuracy", f"{accuracy:.2f}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Precision", f"{precision:.2f}")
            col2.metric("Recall", f"{recall:.2f}")
            col3.metric("F1-Score", f"{f1:.2f}")
            st.metric("Total Samples", f"{total_samples}")

            # ================= Confusion Matrix =================
            st.markdown("### 📋 Confusion Matrix")
            if cm is not None:
                fig_cm = go.Figure(data=go.Heatmap(z=cm, x=["Sell","Hold","Buy"], y=["Sell","Hold","Buy"], colorscale="Viridis"))
                plotly_chart_unique(fig_cm, "ml_confmat")
                safe_download(
                    label="📥 Download Confusion Matrix",
                    data_bytes=fig_cm.to_json().encode("utf-8"),
                    default_name=f"{selected_symbol}_{ml_model}_ConfusionMatrix.json",
                    mime_type="application/json"
                )

            # ================= ROC Curve =================
            st.markdown("### 📈 ROC Curve")
            try:
                if hasattr(trained_model, "predict_proba") and not test_df.empty:
                    from sklearn.preprocessing import label_binarize
                    from sklearn.metrics import roc_curve, auc
                    X_test = test_df[feature_cols]
                    y_true = test_df['Signal'].values
                    y_pred_prob = trained_model.predict_proba(X_test)
                    y_true_bin = label_binarize(y_true, classes=[-1, 0, 1])
                    if y_true_bin.shape[1] == y_pred_prob.shape[1]:
                        fig_roc = go.Figure()
                        for i in range(y_true_bin.shape[1]):
                            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
                            roc_auc = auc(fpr, tpr)
                            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Class {[-1,0,1][i]} (AUC={roc_auc:.2f})"))
                        plotly_chart_unique(fig_roc, "ml_roc")
                        safe_download(
                            label="📥 Download ROC Curve",
                            data_bytes=fig_roc.to_json().encode("utf-8"),
                            default_name=f"{selected_symbol}_{ml_model}_ROC.json",
                            mime_type="application/json"
                        )
            except Exception as e:
                st.error(f"❌ ROC generation failed: {e}")

            # ================= Explainability =================
            if enable_explainability:
                st.markdown("### 🔍 Model Explainability")
                try:
                    results = generate_explainability(df, trained_model, ml_model)
                    st.session_state.explainability_results = results
                    if results.get("shap_bar"):
                        plotly_chart_unique(results["shap_bar"], "ml_shap_bar")
                        # stash PNG for report
                        try:
                            import plotly.io as pio
                            st.session_state["explain_shap_bar_png"] = pio.to_image(results["shap_bar"], format="png", scale=2, width=1000, height=600)
                        except Exception:
                            st.session_state["explain_shap_bar_png"] = None
                        safe_download(
                            label="📥 Download SHAP Bar",
                            data_bytes=results["shap_bar"].to_json().encode("utf-8"),
                            default_name=f"{selected_symbol}_{ml_model}_SHAP_Bar.json",
                            mime_type="application/json"
                        )
                    if results.get("shap_beeswarm"):
                        plotly_chart_unique(results["shap_beeswarm"], "ml_shap_beeswarm")
                        # stash PNG for report
                        try:
                            import plotly.io as pio
                            st.session_state["explain_shap_beeswarm_png"] = pio.to_image(results["shap_beeswarm"], format="png", scale=2, width=1000, height=600)
                        except Exception:
                            st.session_state["explain_shap_beeswarm_png"] = None
                        safe_download(
                            label="📥 Download SHAP Beeswarm",
                            data_bytes=results["shap_beeswarm"].to_json().encode("utf-8"),
                            default_name=f"{selected_symbol}_{ml_model}_SHAP_Beeswarm.json",
                            mime_type="application/json"
                        )
                    if results.get("lime_plot"):
                        plotly_chart_unique(results["lime_plot"], "ml_lime")
                        # stash PNG for report
                        try:
                            import plotly.io as pio
                            st.session_state["explain_lime_png"] = pio.to_image(results["lime_plot"], format="png", scale=2, width=1000, height=600)
                        except Exception:
                            st.session_state["explain_lime_png"] = None
                        safe_download(
                            label="📥 Download LIME",
                            data_bytes=results["lime_plot"].to_json().encode("utf-8"),
                            default_name=f"{selected_symbol}_{ml_model}_LIME.json",
                            mime_type="application/json"
                        )
                    if results.get("shap_force"):
                        st.markdown("#### SHAP Force Plot")
                        try:
                            import shap
                            # results["shap_force"] may be a SHAP object or already HTML
                            obj = results["shap_force"]
                            if hasattr(obj, "to_html"):
                                shap_html_body = obj.to_html()
                            elif isinstance(obj, str):
                                shap_html_body = obj
                            else:
                                # Last-resort: try legacy .html() if present
                                shap_html_body = getattr(obj, "html", lambda: "")() or ""

                            # Ensure SHAP JS is included so the force plot renders in Streamlit
                            shap_html_full = f"<html><head>{shap.getjs()}</head><body style='margin:0'>{shap_html_body}</body></html>"

                            components.html(shap_html_full, height=600, scrolling=True)

                            safe_download(
                                label="📥 Download SHAP Force (HTML)",
                                data_bytes=shap_html_full.encode("utf-8"),
                                default_name=f"{selected_symbol}_{ml_model}_SHAP_Force.html",
                                mime_type="text/html"
                            )
                        except Exception as e:
                            # Static fallback → try to export a PNG
                            import io, matplotlib.pyplot as plt
                            try:
                                fig = results.get("shap_force_matplot")
                                if fig is None:
                                    # If your generator stashed an Explanation, try to draw one
                                    exp = results.get("shap_explanation")
                                    if exp is not None:
                                        import shap as _shap
                                        _shap.plots.force(exp, matplotlib=True, show=False)
                                        fig = plt.gcf()
                                if fig is not None:
                                    buf = io.BytesIO()
                                    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                                    st.image(buf.getvalue(), caption="SHAP Force (static)")
                                    safe_download(
                                        label="📥 Download SHAP Force (PNG)",
                                        data_bytes=buf.getvalue(),
                                        default_name=f"{selected_symbol}_{ml_model}_SHAP_Force.png",
                                        mime_type="image/png"
                                    )
                                else:
                                    st.info("SHAP force plot unavailable for this model/selection.")
                            except Exception:
                                st.info("SHAP force plot unavailable for this model/selection.")

                except Exception as e:
                    st.error(f"❌ Explainability Error: {e}")
        except Exception as e:
                    st.error(f"❌ ML strategy Error: {e}")
        finally:
            # always un-pause auto-refresh
            st.session_state["busy"] = False

# --- Tab 3: DQN Strategy ---
with tab3:
    st.subheader("🎮 DQN Trading Strategy")

    total_timesteps = st.slider("Total Timesteps", 1000, 50000, 10000, step=500, key="dqn_timesteps")
    eval_eps = st.slider("Eval exploration (epsilon)", 0.0, 0.3, 0.10, 0.01,help="At evaluation, take a random action with this probability to avoid 'always hold'.")
    if st.button("Train & Run DQN Agent", key="run_dqn"):
        try:
            from rl_dqn_agent import (
                train_dqn_agent, simulate_trading, evaluate_dqn_performance,
                plot_dqn_reward_curve, plot_dqn_net_worth, plot_dqn_results
            )
            from ml_strategy import fetch_ohlcv_data

            # Clear previous DQN session state
            for key in ["dqn_df", "dqn_worth", "dqn_log"]:
                if key in st.session_state:
                    del st.session_state[key]

            # 1) Fetch data
            with st.spinner("📥 Fetching OHLCV data..."):
                logging.info(f"Fetching OHLCV data for {selected_symbol} from {start_date} to {end_date}")
                df = fetch_ohlcv_data(selected_symbol, start=start_date, end=end_date)
                if df.empty:
                    st.error(f"❌ No data fetched for {selected_symbol} from {start_date} to {end_date}.")
                    logging.error(f"No data fetched for {selected_symbol} from {start_date} to {end_date}")
                    st.stop()

                # Ensure Date column exists and is datetime (keep original order)
                if "Date" not in df.columns:
                    df["Date"] = df.index
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.dropna(subset=["Date"])

            # 2) Sanitize OHLCV (avoid zeros everywhere)
            needed = ["Open", "High", "Low", "Close", "Volume"]
            missing = [c for c in needed if c not in df.columns]
            if missing:
                st.error(f"DQN: Missing columns: {missing}")
                st.stop()

            df = df.copy()

            # Ensure numeric
            for c in needed:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            # Sort by Date just to be safe
            df = df.sort_values("Date").reset_index(drop=True)

            # Replace inf -> NaN; then forward/back fill
            df[needed] = df[needed].replace([np.inf, -np.inf], np.nan)
            df[needed] = df[needed].ffill().bfill()

            # If still NaNs, stop with message
            if df[needed].isna().any().any():
                st.error("DQN: NaNs remain in OHLCV after fill; widen date range or change symbol.")
                st.stop()

            # If Volume all zero, add epsilon to avoid degenerate obs
            if (df["Volume"] == 0).all():
                df["Volume"] = 1e-6

            # Quick preview so you can see non-zero data before training
            st.caption(
                f"DQN data preview → rows: {len(df)} | "
                f"Close[min/mean/max]: {df['Close'].min():.2f} / {df['Close'].mean():.2f} / {df['Close'].max():.2f}"
            )
            st.dataframe(df[["Date", "Open", "High", "Low", "Close", "Volume"]].tail(5), use_container_width=True)

            # 3) Train DQN Agent
            with st.spinner("🧠 Training DQN agent..."):
                dqn_model = train_dqn_agent(df, total_timesteps=total_timesteps)

            # 4) Simulate trading
            with st.spinner("📈 Simulating trades using trained agent..."):
                rewards, worths, trade_log = simulate_trading(df, dqn_model, eval_epsilon=eval_eps)

            # 5) Align lengths if needed
            if len(df) != len(worths):
                st.warning(f"⚠️ DQN DataFrame length ({len(df)}) does not match DQN worth length ({len(worths)}). Truncating.")
                logging.warning(f"DQN DataFrame length: {len(df)}, DQN worth length: {len(worths)}")
                min_len = min(len(df), len(worths))
                df = df.iloc[:min_len].reset_index(drop=True)
                worths = worths[:min_len]
                if not trade_log.empty:
                    trade_log = trade_log[trade_log["step"].apply(lambda x: x < min_len)]

            # 6) Store in session state
            st.session_state.dqn_df = df
            st.session_state.dqn_worth = worths
            st.session_state.dqn_log = trade_log

            # 7) UI outputs
            st.success("✅ DQN Strategy executed successfully!")
            st.write("### DQN Strategy Results")
            st.dataframe(trade_log.tail(10), use_container_width=True)

            # Plot trade chart
            fig = plot_dqn_results(df, worths, trade_log, title=f"{selected_symbol} – DQN Strategy")
            plotly_chart_unique(fig, "dqn_trades")
            # Save equity curve (optional)
            chart_filename = f"{selected_symbol}_DQN_Equity.png"
            saved = False
            try:
                # we already rendered the Plotly figure above; try to write a PNG
                import plotly.io as pio
                pio.write_image(fig, chart_filename, engine="kaleido", width=1000, height=600)
                saved = True
                st.caption("💡 Tip: You can also use the camera icon on the chart.")
            except Exception as e:
                logging.warning(f"Failed to save DQN equity curve: {e}")

            if saved and os.path.exists(chart_filename):
                with open(chart_filename, "rb") as f:
                    st.download_button(
                        label="📥 Download DQN Equity Curve (PNG)",
                        data=f,
                        file_name=chart_filename,
                        mime="image/png"
                    )


            # Evaluate strategy
            st.markdown("### 📊 Strategy Summary")
            perf = evaluate_dqn_performance(trade_log)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Final Net Worth", f"₹{perf['final_net_worth']:,.2f}")
                st.metric("Avg Profit/Trade", f"₹{perf['avg_profit']:,.2f}")
            with col2:
                st.metric("Total Trades", perf["total_trades"])
                st.metric("Win Rate", f"{perf['win_rate']}%")
            with col3:
                st.metric("Buy Trades", perf["buy_trades"])
                st.metric("Sell Trades", perf["sell_trades"])

            # Reward & Net Worth charts
            st.markdown("### 📉 DQN Reward & Net Worth Plots")
            col_a, col_b = st.columns(2)
            with col_a:
                fig_reward = plot_dqn_reward_curve(rewards, title="DQN Reward Curve")
                st.pyplot(fig_reward)
                buf_reward = fig_reward.canvas.print_to_buffer()[0]
                st.download_button("📥 Download Reward Plot", buf_reward, file_name="dqn_reward_curve.png", mime="image/png")
            with col_b:
                fig_worth = plot_dqn_net_worth(worths, title="DQN Net Worth")
                st.pyplot(fig_worth)
                buf_worth = fig_worth.canvas.print_to_buffer()[0]
                st.download_button("📥 Download Net Worth Plot", buf_worth, file_name="dqn_networth_curve.png", mime="image/png")

        except Exception as e:
            st.error(f"❌ Error in DQN Strategy: {e}")
            logging.error(f"DQN Strategy error: {e}")

# --- Tab 4: Strategy Comparison ---
with tab4:
    st.subheader("⚖️ ML vs. DQN Strategy Comparison")
    
    # Check if both strategies have been run
    required_keys = ["ml_df", "ml_equity", "ml_final_cash", "dqn_df", "dqn_worth", "dqn_log"]
    if not all(key in st.session_state for key in required_keys):
        st.warning("⚠️ Please run both ML Strategy (Tab 2) and DQN Strategy (Tab 3) before comparing.")
        logging.warning("Missing session state keys for comparison")
        st.stop()
    
    # Retrieve data
    ml_df = st.session_state.ml_df
    ml_equity = st.session_state.ml_equity
    ml_final_cash = st.session_state.ml_final_cash
    dqn_df = st.session_state.dqn_df
    dqn_worth = st.session_state.dqn_worth
    dqn_log = st.session_state.dqn_log

    # Ensure Date columns are datetime
    ml_df["Date"] = pd.to_datetime(ml_df["Date"], errors='coerce')
    dqn_df["Date"] = pd.to_datetime(dqn_df["Date"], errors='coerce')
    
    # Drop rows with invalid dates
    ml_df = ml_df.dropna(subset=["Date"])
    dqn_df = dqn_df.dropna(subset=["Date"])
    
    if ml_df.empty or dqn_df.empty:
        st.error("❌ One or both DataFrames have no valid dates after conversion.")
        logging.error(f"Empty DataFrame after date conversion: ml_df empty={ml_df.empty}, dqn_df empty={dqn_df.empty}")
        st.stop()

    # Align date ranges
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    start_date = max(ml_df["Date"].min(), dqn_df["Date"].min(), start_date)
    end_date = min(ml_df["Date"].max(), dqn_df["Date"].max(), end_date)
    
    mask_ml = (ml_df["Date"] >= start_date) & (ml_df["Date"] <= end_date)
    mask_dqn = (dqn_df["Date"] >= start_date) & (dqn_df["Date"] <= end_date)
    ml_df = ml_df[mask_ml]
    dqn_df = dqn_df[mask_dqn]
    
    # Align equity curves
    ml_equity = ml_equity[:len(ml_df)]
    dqn_worth = dqn_worth[:len(dqn_df)]
    
    # Log data details
    logging.info(f"ML df length: {len(ml_df)}, ML equity length: {len(ml_equity)}")
    logging.info(f"DQN df length: {len(dqn_df)}, DQN worth length: {len(dqn_worth)}")
    logging.info(f"Date range: {start_date} to {end_date}")

    # Plot comparison
    plot_type = st.selectbox("Select Plot Type", ["Plotly", "Matplotlib"], key="comparison_plot_type")
    
    cmp_png = None  # will hold an in-memory PNG for downloads + report

    if plot_type == "Plotly":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ml_df["Date"],
            y=ml_equity,
            name="ML Strategy",
            line=dict(color="blue")
        ))
        fig.add_trace(go.Scatter(
            x=dqn_df["Date"],
            y=dqn_worth,
            name="DQN Strategy",
            line=dict(color="green")
        ))
        fig.update_layout(
            title=f"{selected_symbol} – ML vs DQN Equity Curve",
            xaxis_title="Date",
            yaxis_title="Portfolio Value (₹)",
            template="plotly_white",
            height=500,
            xaxis=dict(range=[start_date, end_date])
        )
        plotly_chart_unique(fig, "compare_equity")

        # In-memory PNG (no filesystem/Kaleido needed)
        try:
            import plotly.io as pio
            cmp_png = pio.to_image(fig, format="png", scale=2, width=1200, height=500)
            st.download_button(
                label="📥 Download Equity Curve (PNG)",
                data=cmp_png,
                file_name=f"{selected_symbol}_Comparison_Equity.png",
                mime="image/png",
                key="dl_cmp_equity_png"
            )
        except Exception as e:
            logging.warning(f"Comparison chart bytes error: {e}")
            st.caption("💡 Tip: You can use the camera icon on the chart to export a PNG.")

    elif plot_type == "Matplotlib":
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(ml_df["Date"], ml_equity, label="ML Strategy", color="blue")
        ax.plot(dqn_df["Date"], dqn_worth, label="DQN Strategy", color="green")
        ax.set_title(f"{selected_symbol} – ML vs DQN Equity Curve")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value (₹)")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Optional: Matplotlib to PNG bytes for download
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            cmp_png = buf.getvalue()
            st.download_button(
                label="📥 Download Equity Curve (PNG)",
                data=cmp_png,
                file_name=f"{selected_symbol}_Comparison_Equity.png",
                mime="image/png",
                key="dl_cmp_equity_png_mat"
            )
        except Exception as e:
            logging.warning(f"Matplotlib comparison bytes error: {e}")

    # Display metrics
    st.markdown("### 📊 Performance Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("ML Final Portfolio Value", f"₹{ml_final_cash:,.2f}")
        ml_roi = ((ml_final_cash - 10000) / 10000) * 100
        st.metric("ML ROI", f"{ml_roi:.2f}%")
    with col2:
        dqn_final_worth = dqn_worth[-1] if dqn_worth else 10000
        st.metric("DQN Final Portfolio Value", f"₹{dqn_final_worth:,.2f}")
        dqn_roi = ((dqn_final_worth - 10000) / 10000) * 100
        st.metric("DQN ROI", f"{dqn_roi:.2f}%")

    # --- Generate Strategy Report (HTML everywhere, optional PDF) ---
    if st.button("Generate Strategy Report", key="generate_report"):
        try:
            import os, base64
            from report_generator import generate_html_report, html_to_pdf_bytes

            # Pull items you already keep in session/state
            model_name = st.session_state.get("ml_model", "Logistic Regression")
            acc = st.session_state.get("ml_accuracy", 0.0)  # if you store true accuracy
            shap_path = st.session_state.get("shap_plot_path", "")
            lime_path = st.session_state.get("lime_plot_path", "")
            chart_path = ""  # file-based chart may not exist on Cloud; we’ll embed bytes below

            # Tables for the report
            benchmark_df = pd.DataFrame({
                "Strategy": ["ML", "DQN"],
                "Final Value": [ml_final_cash, dqn_final_worth],
                "ROI (%)": [ml_roi, dqn_roi],
            })

            report_df = pd.DataFrame({
                "Date": ml_df["Date"].dt.strftime("%Y-%m-%d"),
                "ML Equity": ml_equity,
                "DQN Worth": [x if i < len(dqn_worth) else None for i, x in enumerate(dqn_worth)],
            }).dropna()

            # Optional: embed images (SHAP/LIME/equity chart) inline using base64 if files exist
            def _img_tag_if_exists(path, title):
                try:
                    if path and os.path.exists(path):
                        with open(path, "rb") as fh:
                            b64 = base64.b64encode(fh.read()).decode("ascii")
                        return f'<h3>{title}</h3><img src="data:image/png;base64,{b64}" style="max-width:100%;border:1px solid #eee;border-radius:6px;margin:6px 0;" />'
                except Exception:
                    pass
                return ""

            # NEW: embed comparison chart from bytes if available
            def _img_tag_from_bytes(png_bytes, title):
                if not png_bytes:
                    return ""
                b64 = base64.b64encode(png_bytes).decode("ascii")
                return f'<h3>{title}</h3><img src="data:image/png;base64,{b64}" style="max-width:100%;border:1px solid #eee;border-radius:6px;margin:6px 0;" />'

            visuals_html = ""
            visuals_html += _img_tag_if_exists(chart_path, "Equity Curve")
            # prefer in-memory PNGs captured in Tab 2 if present
            shap_bar_png = st.session_state.get("explain_shap_bar_png")
            shap_bee_png = st.session_state.get("explain_shap_beeswarm_png")
            lime_png     = st.session_state.get("explain_lime_png")
            if shap_bar_png:
                visuals_html += _img_tag_from_bytes(shap_bar_png, "SHAP Feature Importance (Bar)")
            if shap_bee_png:
                visuals_html += _img_tag_from_bytes(shap_bee_png, "SHAP Beeswarm")
            if lime_png:
                visuals_html += _img_tag_from_bytes(lime_png, "LIME Explanation")
            visuals_html += _img_tag_from_bytes(cmp_png, "ML vs DQN Equity (Comparison)")


            # Build the context for the report generator
            ctx = {
                "title": "Strategy Comparison Report",
                "symbol": selected_symbol,
                "period": f"{start_date} → {end_date}",
                "kpis": {
                    "Model": model_name,
                    "Accuracy": f"{acc:.2%}" if isinstance(acc, (int, float)) else acc,
                    "ML Final": f"₹{ml_final_cash:,.2f}",
                    "DQN Final": f"₹{dqn_final_worth:,.2f}",
                    "ML ROI": f"{ml_roi:.2f}%",
                    "DQN ROI": f"{dqn_roi:.2f}%",
                },
                "tables": {
                    "Benchmark Summary": benchmark_df,
                    "Equity vs Worth (sample)": report_df.head(100),  # keep light; full CSV is downloadable elsewhere
                    **({"Visuals": visuals_html} if visuals_html else {}),
                },
                "notes": (
                    "This report compares ML and DQN strategies over the selected period. "
                    "Visuals are included when available (equity curve / SHAP / LIME)."
                ),
            }

            # 1) Always create HTML (works on Streamlit Cloud)
            html_str = generate_html_report(ctx)

            # --- Ensure visuals appear in the HTML even if the report generator escapes table cells ---
            # We append our assembled <img> HTML right before </body>.
            try:
                if visuals_html:
                    html_str = html_str.replace(
                        "</body>",
                        f'<section style="margin-top:16px"><h2 style="font-family:Segoe UI, sans-serif;">Visuals</h2>{visuals_html}</section></body>'
                    )
            except Exception as _e:
                logging.warning(f"Could not inline visuals into HTML report: {_e}")

            st.download_button(
                label="📄 Download Strategy Report (HTML)",
                data=html_str,
                file_name=f"{selected_symbol}_Strategy_Comparison.html",
                mime="text/html",
                key="dl_report_html",
            )

            # 2) Try PDF (works locally if WeasyPrint is installed; returns None on Cloud)
            pdf_bytes = html_to_pdf_bytes(html_str)
            if pdf_bytes:
                st.download_button(
                    label="🧾 Download Strategy Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"{selected_symbol}_Strategy_Comparison.pdf",
                    mime="application/pdf",
                    key="dl_report_pdf",
                )
            else:
                st.info("PDF generation isn’t available in this environment. Use the HTML report (or run locally with WeasyPrint installed).")

        except Exception as e:
            st.error(f"❌ Failed to generate report: {e}")
            logging.error(f"Report generation error: {e}")


# --- Tab 5: News Sentiment ---
with tab5:
    st.subheader(f"📰 News Sentiment for {selected_symbol}")
    st.markdown("Analyze recent headlines for real-time sentiment using Google News RSS and VADER model.")

    max_headlines = st.slider("📰 Number of Headlines", 5, 25, 10)
    run_news_sentiment = st.button("🔍 Run Sentiment Analysis", key="run_news_btn")

    if run_news_sentiment:
        with st.spinner("🔎 Fetching latest news..."):
            news_df = fetch_google_news_sentiment(ticker=selected_symbol.replace(".NS", ""), max_items=max_headlines)

        if news_df.empty:
            st.warning("⚠️ No recent news found for this stock.")
        else:
            st.success(f"✅ Fetched {len(news_df)} headlines for {selected_symbol}")
            # --- Quick Sentiment Summary (counts + average score) ---
            avg_score = float(news_df['Score'].mean())
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric(label="Avg Compound Score", value=f"{avg_score:.3f}")
            with c2:
                counts = news_df['Sentiment'].value_counts().reindex(['Positive','Neutral','Negative']).fillna(0).astype(int)
                bar_fig = go.Figure(data=[go.Bar(x=counts.index.tolist(), y=counts.values.tolist())])
                bar_fig.update_layout(title=f"Headline Counts by Sentiment – {selected_symbol}", height=300, xaxis_title="Sentiment", yaxis_title="Count")
                plotly_chart_unique(bar_fig, "news_bar")
            # --- Daily average sentiment over time ---
            show_daily_line = st.checkbox('Show daily average sentiment line', value=True, key='news_daily_line')
            try:
                df_plot = news_df.copy()
                df_plot['Published'] = pd.to_datetime(df_plot['Published'], errors='coerce')
                df_plot.dropna(subset=['Published'], inplace=True)
                df_plot['date'] = df_plot['Published'].dt.date
                grouped = df_plot.groupby('date').agg(avg_score=('Score','mean'), count=('Score','size')).reset_index()
                if show_daily_line:
                    line_fig = go.Figure(data=[go.Scatter(
                        x=grouped['date'], y=grouped['avg_score'], mode='lines+markers',
                        customdata=grouped['count'],
                        hovertemplate='Date: %{x}<br>Avg: %{y:.3f}<br>Headlines: %{customdata}<extra></extra>'
                    )])
                    line_fig.update_layout(title=f"Daily Avg Sentiment – {selected_symbol}", height=300, xaxis_title="Date", yaxis_title="Avg Compound Score")
                    plotly_chart_unique(line_fig, "news_daily")
                # CSV download for daily averages
                daily_csv = grouped.rename(columns={'avg_score':'AvgScore','count':'Headlines'}).to_csv(index=False).encode()
                st.download_button('📥 Download Daily Averages (CSV)', data=daily_csv, file_name=f'{selected_symbol}_daily_sentiment.csv')
            except Exception as e:
                logging.warning(f"Daily sentiment plot failed: {e}")

            # Tabs for sentiment breakdown
            tab_pos, tab_neu, tab_neg, tab_all = st.tabs(["🟢 Positive", "🟡 Neutral", "🔴 Negative", "📋 All News"])

            def render_news(df, label):
                icon = "✅" if label == "Positive" else "⚠️" if label == "Neutral" else "❌"
                for _, row in df.iterrows():
                    with st.expander(f"{icon} {row['Headline']} ({row['Score']:+.2f})"):
                        st.markdown(f"**Published**: {row['Published']}")
                        st.markdown(f"**Sentiment**: {row['Sentiment']}")
                        st.markdown(f"[🔗 Read Full Article]({row['Link']})")

            with tab_pos:
                pos_df = news_df[news_df["Sentiment"] == "Positive"]
                render_news(pos_df, "Positive")

            with tab_neu:
                neu_df = news_df[news_df["Sentiment"] == "Neutral"]
                render_news(neu_df, "Neutral")

            with tab_neg:
                neg_df = news_df[news_df["Sentiment"] == "Negative"]
                render_news(neg_df, "Negative")

            with tab_all:
                st.dataframe(news_df[["Published", "Headline", "Sentiment", "Score"]], use_container_width=True)

            # Pie Chart
            st.markdown("### 🥧 Sentiment Breakdown")
            sentiment_counts = news_df["Sentiment"].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                hole=0.3,
                marker=dict(colors=["#6BCB77", "#FFD93D", "#FF6B6B"])
            )])
            fig.update_layout(title=f"Sentiment Pie – {selected_symbol}", height=400)
            plotly_chart_unique(fig, "news_pie")

            # CSV download
            csv = news_df.to_csv(index=False).encode()
            st.download_button("📥 Download Full News Sentiment CSV", data=csv, file_name=f"{selected_symbol}_news_sentiment.csv")

# --- Tab 6: Live Feed ---
with tab6:
    st.subheader("📡 Live Market Feed & Price Monitoring")
    st.caption("If you see this, the tab rendered successfully ✅")
    st_autorefresh(interval=60_000, key="alert_refresh")

    # Init dedup state for alert demo
    if "last_alert_text" not in st.session_state:
        st.session_state["last_alert_text"] = None
    if "last_alert_time" not in st.session_state:
        st.session_state["last_alert_time"] = None

    logger = logging.getLogger(__name__)
    refresh_rate = st.session_state.get("refresh_rate", 60)

    # ─────────────────────────────────────────────
    # Trading session masking (rangebreaks toggle)
    # ─────────────────────────────────────────────
    st.markdown("#### 🕒 Trading Session View")
    hide_non_trading = st.checkbox(
        "Hide non-trading time (weekends & off-hours)",
        value=True,
        key="hide_non_trading",
    )

    def _rangebreaks_trading_hours():
        # NSE 09:15–15:30 IST, Mon–Fri
        return [
            dict(bounds=["sat", "mon"]),             # weekends
            dict(bounds=[0, 9.25],  pattern="hour"), # 00:00–09:15
            dict(bounds=[15.5, 24], pattern="hour"), # 15:30–24:00
        ]

    # =========================
    # 1) Unified Candlestick
    # =========================
    st.markdown("### 🕯️ Candlestick (multi-timeframe)")
    tf = st.selectbox("Timeframe", ["5m", "10m", "15m", "1h", "1d"], index=0, key="candle_tf_unified")
    show_bbands = st.checkbox("Overlay Bollinger Bands (20,2)", value=False, key="candle_bb_unified")

    # Any date range: default 30d (1d TF) or 5d (intraday TFs)
    today = pd.Timestamp.today().normalize()
    default_days = 30 if tf == "1d" else 5
    default_start = today - pd.Timedelta(days=default_days)
    start_date, end_date = st.date_input(
        "Date range",
        value=(default_start.date(), today.date()),
        max_value=today.date(),
        key="candle_date_range_unified",
    )

    try:
        start_ts = pd.to_datetime(start_date)
        end_ts_exclusive = pd.to_datetime(end_date) + pd.Timedelta(days=1)

        # Fetch OHLCV
        if tf == "10m":
            # Pull 5m then resample → 10m (clean raw first)
            raw = yf.Ticker(selected_symbol).history(start=start_ts, end=end_ts_exclusive, interval="5m")
            if not raw.empty:
                try:
                    raw.index = raw.index.tz_convert("Asia/Kolkata")
                except Exception:
                    raw.index = raw.index.tz_localize("Asia/Kolkata")
                raw = raw[~raw.index.duplicated(keep="last")].sort_index()
                df_c = (
                    raw.resample("10T")
                    .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
                    .dropna()
                )
            else:
                df_c = raw
        else:
            itv = {"5m": "5m", "15m": "15m", "1h": "60m", "1d": "1d"}[tf]
            df_c = yf.Ticker(selected_symbol).history(start=start_ts, end=end_ts_exclusive, interval=itv)

        if df_c.empty:
            tip = " • Tip: For intraday, narrow the date range or use 1d." if tf in ("5m","10m","15m","1h") else ""
            st.info(f"No data for {selected_symbol} in the selected range at {tf}.{tip}")
        else:
            # Clean timezone & duplicates
            try:
                df_c.index = df_c.index.tz_convert("Asia/Kolkata")
            except Exception:
                df_c.index = df_c.index.tz_localize("Asia/Kolkata")
            df_c = df_c[~df_c.index.duplicated(keep="last")].sort_index()

            # Drop any rows with zero OHLC that would drag y-axis to 0
            if set(["Open","High","Low","Close"]).issubset(df_c.columns):
                df_c = df_c[(df_c[["Open","High","Low","Close"]] > 0).all(axis=1)]

            # Guard: still have data?
            if df_c.empty:
                st.info("All rows were invalid/zero; no candles to plot in this range.")
            else:
                # --- Candlestick ---
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df_c.index,
                    open=df_c["Open"], high=df_c["High"], low=df_c["Low"], close=df_c["Close"],
                    name="Price",
                    increasing_line_color="green", decreasing_line_color="red",
                    increasing_fillcolor="green", decreasing_fillcolor="red"
                ))

                # Optional BBANDS — your ml_strategy.compute_bollinger_bands now returns NaNs (no zeros)
                if show_bbands and len(df_c) >= 20:
                    from ml_strategy import compute_bollinger_bands
                    mid, up, lo = compute_bollinger_bands(df_c["Close"], window=20, num_std=2)
                    bb = pd.DataFrame({"mid": mid, "up": up, "lo": lo}).dropna()
                    if not bb.empty:
                        fig.add_trace(go.Scatter(x=bb.index, y=bb["up"], name="BB Upper", mode="lines", opacity=0.6))
                        fig.add_trace(go.Scatter(x=bb.index, y=bb["mid"], name="BB Mid",   mode="lines", opacity=0.4))
                        fig.add_trace(go.Scatter(x=bb.index, y=bb["lo"], name="BB Lower",  mode="lines", opacity=0.6))

                # Tight y-range around actual highs/lows (no zero anchor)
                ymin = float(df_c["Low"].min())
                ymax = float(df_c["High"].max())
                pad = max((ymax - ymin) * 0.05, 0.5)  # 5% or at least 0.5
                fig.update_layout(
                    height=420,
                    template="plotly_white",
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=20, r=20, t=30, b=20),
                    title=f"{selected_symbol} — {tf} candlesticks",
                )
                fig.update_yaxes(range=[ymin - pad, ymax + pad])
                if hide_non_trading and tf != "1d":
                    fig.update_xaxes(rangebreaks=_rangebreaks_trading_hours())

                plotly_chart_unique(fig, "live_candles")

                # Export buttons (PNG via in-memory bytes)
                col_csv, col_png = st.columns(2)
                with col_csv:
                    st.download_button(
                        "📥 Download CSV",
                        data=df_c.reset_index().to_csv(index=False).encode(),
                        file_name=f"{selected_symbol}_{tf}_{start_date}_to_{end_date}.csv",
                        mime="text/csv",
                        key="candle_csv_unified",
                    )
                with col_png:
                    if st.button("📸 Export PNG", key="candle_export_png_unified"):
                        try:
                            import plotly.io as pio
                            png_bytes = pio.to_image(fig, format="png", scale=2, width=1100, height=420)
                            st.download_button(
                                "📥 Download PNG",
                                data=png_bytes,
                                file_name=f"{selected_symbol}_{tf}_{start_date}_to_{end_date}.png",
                                mime="image/png",
                                key="candle_png_dl_unified"
                            )
                        except Exception as img_err:
                            st.warning(f"PNG export failed: {img_err}")

                # 20-min Volatility Snapshot (from same cleaned df)
                if len(df_c) > 20:
                    st.markdown("### ⚡ 20-Min Volatility Snapshot")
                    vol_fig = go.Figure()
                    vol_fig.add_trace(go.Scatter(
                        x=df_c.index[-20:], y=df_c["Close"].iloc[-20:],
                        mode="lines+markers", marker=dict(size=5),
                        name="Close Price"
                    ))
                    # Tighten y range for the snapshot too
                    vmin = float(df_c["Close"].iloc[-20:].min())
                    vmax = float(df_c["Close"].iloc[-20:].max())
                    vpad = max((vmax - vmin) * 0.05, 0.25)
                    vol_fig.update_layout(
                        height=250,
                        margin=dict(l=10, r=10, t=20, b=10),
                        template="plotly_dark",
                        showlegend=False,
                        xaxis_title="",
                        yaxis_title="Price",
                    )
                    vol_fig.update_yaxes(range=[vmin - vpad, vmax + vpad])
                    plotly_chart_unique(vol_fig, "live_volatility")

    except Exception as e:
        st.warning(f"Candlestick error: {e}")

    # =========================
    # 2) KPI (single, no extra candle)
    # =========================
    try:
        # Use prev close for a meaningful % even after-hours
        current_price, open_price = get_live_price(selected_symbol)  # keep your function call
        used_fallback = False

        # Derive prev_close (2 most recent daily bars)
        prev_close = None
        try:
            d2 = yf.Ticker(selected_symbol).history(period="2d", interval="1d")
            if len(d2) >= 2:
                prev_close = float(d2["Close"].iloc[-2])
            elif len(d2) == 1:
                prev_close = float(d2["Close"].iloc[-1])
        except Exception:
            prev_close = None

        if current_price is None or np.isnan(current_price) or prev_close is None or np.isnan(prev_close):
            daily_fb = yf.Ticker(selected_symbol).history(period="5d", interval="1d")
            if not daily_fb.empty:
                current_price = float(daily_fb["Close"].iloc[-1])
                prev_close   = float(daily_fb["Close"].iloc[-2]) if len(daily_fb) > 1 else float(daily_fb["Close"].iloc[-1])
                used_fallback = True

        if current_price is not None and prev_close is not None:
            pct = ((current_price - prev_close) / prev_close) * 100 if prev_close else 0.0
            arrow = "↑" if pct >= 0 else "↓"
            st.metric(
                label=f"📘 {selected_symbol} Live Price" + (" (daily)" if used_fallback else ""),
                value=f"₹{current_price:.2f}",
                delta=f"{arrow} {abs(pct):.2f}%",
                delta_color="normal"  # green for positive, red for negative
            )
            st.caption(("📅 Using last daily data (market closed). " if used_fallback else "") + f"Auto-refresh every {refresh_rate} sec.")
        else:
            st.warning("⚠️ Could not fetch price for KPI.")

    except Exception as e:
        st.warning(f"KPI error: {e}")

    # =========================
    # 3) Telegram alert demo
    # =========================
    with st.expander("📣 Telegram alerts", expanded=True):
        st.caption("Configure simple alert rules. Click **Send Test Alert** to push a real Telegram message")
        now_ist_str = dt.datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S IST")

        colA, colB, colC = st.columns(3)
        with colA:
            pct_move_thresh = st.number_input("Price move threshold (%)", min_value=0.1, value=1.0, step=0.1)
            vol_spike_mult  = st.number_input("Volume spike × average", min_value=1.0, value=1.5, step=0.1)
        with colB:
            rsi_buy_level   = st.number_input("RSI oversold → buy (≤)", min_value=5, max_value=50, value=30, step=1)
            rsi_sell_level  = st.number_input("RSI overbought → sell (≥)", min_value=50, max_value=95, value=70, step=1)
        with colC:
            adx_trend_level = st.number_input("ADX trend pickup (≥)", min_value=5, max_value=60, value=25, step=1)
            show_preview_only = st.checkbox("Preview only (don’t send)", value=False)

        try:
            live = yf.Ticker(selected_symbol).history(period="1d", interval="1m")
            if live.empty:
                live = yf.Ticker(selected_symbol).history(period="5d", interval="5m")

            if not live.empty:
                last_close = float(live["Close"].iloc[-1])
                prev_close_intra = float(live["Close"].iloc[-2]) if len(live) > 1 else last_close
                day_open   = float(live["Open"].iloc[0])
                day_high   = float(live["High"].max())
                day_low    = float(live["Low"].min())
                pct_move   = ((last_close - prev_close_intra) / prev_close_intra) * 100 if prev_close_intra else 0.0

                vol_now  = float(live["Volume"].iloc[-1]) if "Volume" in live.columns else 0.0
                vol_avg  = float(live["Volume"].rolling(20).mean().iloc[-1]) if "Volume" in live.columns else 0.0
                spike_ok = (vol_avg > 0 and vol_now >= vol_spike_mult * vol_avg)

                def _rsi(s, n=14):
                    d = s.diff()
                    up = d.clip(lower=0).rolling(n).mean()
                    dn = (-d.clip(upper=0)).rolling(n).mean()
                    rs = up / dn.replace(0, np.nan)
                    return (100 - (100 / (1 + rs))).fillna(50)

                def _adx(h, l, c, n=14):
                    df_ = pd.DataFrame({"h": h, "l": l, "c": c})
                    tr  = np.maximum(
                        df_["h"] - df_["l"],
                        np.maximum((df_["h"] - df_["c"].shift(1)).abs(), (df_["l"] - df_["c"].shift(1)).abs())
                    )
                    pdm = np.where(
                        (df_["h"] - df_["h"].shift(1)) > (df_["l"].shift(1) - df_["l"]),
                        np.maximum(df_["h"] - df_["h"].shift(1), 0),
                        0
                    )
                    ndm = np.where(
                        (df_["l"].shift(1) - df_["l"]) > (df_["h"] - df_["h"].shift(1)),
                        np.maximum(df_["l"].shift(1) - df_["l"], 0),
                        0
                    )
                    trn  = pd.Series(tr).rolling(n).sum()
                    pdmn = pd.Series(pdm).rolling(n).sum()
                    ndmn = pd.Series(ndm).rolling(n).sum()
                    pdi = 100 * (pdmn / trn).replace({0: np.nan})
                    ndi = 100 * (ndmn / trn).replace({0: np.nan})
                    dx  = (100 * (abs(pdi - ndi) / (pdi + ndi))).replace({np.inf: np.nan})
                    return dx.rolling(n).mean().fillna(20)

                look = min(len(live), 200)
                rsi_now = float(_rsi(live["Close"].iloc[-look:]).iloc[-1]) if look >= 15 else 50.0
                adx_now = float(_adx(live["High"].iloc[-look:], live["Low"].iloc[-look:], live["Close"].iloc[-look:]).iloc[-1]) if look >= 30 else 20.0

                mid = live["Close"].rolling(20).mean()
                std = live["Close"].rolling(20).std(ddof=0)
                bb_up = (mid + 2 * std).iloc[-1] if not mid.isna().iloc[-1] else None
                bb_lo = (mid - 2 * std).iloc[-1] if not mid.isna().iloc[-1] else None
                bb_break = "↑ upper" if (bb_up and last_close > bb_up) else ("↓ lower" if (bb_lo and last_close < bb_lo) else None)

                reasons = []
                if abs(pct_move) >= pct_move_thresh: reasons.append(f"Price {pct_move:+.2f}% vs prev")
                if spike_ok: reasons.append(f"Vol spike ×{(vol_now / vol_avg):.1f}" if vol_avg else "Vol spike")
                if rsi_now <= rsi_buy_level: reasons.append(f"RSI {rsi_now:.0f} (oversold)")
                if rsi_now >= rsi_sell_level: reasons.append(f"RSI {rsi_now:.0f} (overbought)")
                if adx_now >= adx_trend_level: reasons.append(f"ADX {adx_now:.0f} (trend)")
                if bb_break: reasons.append(f"BB break ({bb_break})")
                triggered = " | ".join(reasons) if reasons else "No rule triggered"

                emoji = "🚨" if reasons else "ℹ️"
                

                msg = (
                    f"{emoji} {selected_symbol}\n"
                    f"Last: ₹{last_close:,.2f}  (Δ {pct_move:+.2f}%)\n"
                    f"Day: H {day_high:,.2f}  L {day_low:,.2f}  O {day_open:,.2f}\n"
                    f"RSI: {rsi_now:.0f}   ADX: {adx_now:.0f}\n"
                    f"Rules: {triggered}\n"
                    f"{now_ist_str}"
                )
            else:
                # Fallback daily (for demo when intraday unavailable)
                daily = yf.Ticker(selected_symbol).history(period="5d", interval="1d")
                if daily.empty:
                    msg = f"ℹ️ {selected_symbol}\nNo recent data available."
                    last_close = rsi_now = adx_now = 0.0
                else:
                    last_close = float(daily["Close"].iloc[-1])
                    prev_close = float(daily["Close"].iloc[-2]) if len(daily) > 1 else last_close
                    day_open   = float(daily["Open"].iloc[-1])
                    day_high   = float(daily["High"].iloc[-1])
                    day_low    = float(daily["Low"].iloc[-1])
                    pct_move   = ((last_close - prev_close) / prev_close) * 100 if prev_close else 0.0
                    rsi_now, adx_now = 50.0, 20.0
                    msg = (
                        f"ℹ️ {selected_symbol}\n"
                        f"Last: ₹{last_close:,.2f}  (Δ {pct_move:+.2f}%)\n"
                        f"Day: H {day_high:,.2f}  L {day_low:,.2f}  O {day_open:,.2f}\n"
                        f"RSI: {rsi_now:.0f}   ADX: {adx_now:.0f}\n"
                        f"Rules: Market closed (daily data)\n"
                        f"{now_ist_str}"
                    )

            # Message preview + cooldown
            st.text_area("Telegram message preview", msg, height=140)
            cooldown_min = st.number_input("Cooldown (minutes)", min_value=0, value=5, step=1)

            # ---- AUTO SEND on refresh if rules triggered ----
            if 'reasons' in locals() and reasons:
                now = dt.datetime.now(pytz.timezone("Asia/Kolkata"))
                last_text = st.session_state.get("last_alert_text")
                last_time = st.session_state.get("last_alert_time")
                within_cooldown = bool(last_time) and ((now - last_time).total_seconds() < cooldown_min * 60)
                if (msg != last_text) or not within_cooldown:
                    if not show_preview_only:
                        try:
                            send_telegram_alert(msg)
                            st.session_state["last_alert_text"] = msg
                            st.session_state["last_alert_time"] = now
                            st.info("📨 Auto alert sent (refresh driven).")
                        except Exception as e:
                            st.warning(f"Auto-send failed: {e}")

            if st.button("📤 Send Test Alert", key="send_tele_demo"):
                now = dt.datetime.now(pytz.timezone("Asia/Kolkata"))
                last_text = st.session_state.get("last_alert_text")
                last_time = st.session_state.get("last_alert_time")
                is_duplicate = (last_text == msg)
                within_cooldown = bool(last_time) and ((now - last_time).total_seconds() < cooldown_min * 60)

                if is_duplicate and within_cooldown:
                    st.info(f"⏱️ Duplicate message within cool-down ({cooldown_min} min). Not sending.")
                else:
                    if show_preview_only:
                        st.info("Preview-only mode: not sending.")
                    else:
                        try:
                            send_telegram_alert(msg)  # guarded sender
                            st.success("Sent (or skipped if secrets not set). Check logs.")
                            st.session_state["last_alert_text"] = msg
                            st.session_state["last_alert_time"] = now
                        except Exception as e:
                            st.warning(f"Send failed: {e}")

            st.write(f"Last close ₹{last_close:,.2f}  | RSI {rsi_now:.0f}  | ADX {adx_now:.0f}")
            if st.session_state.get("last_alert_time"):
                ts = st.session_state["last_alert_time"].astimezone(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S IST")
                st.caption(f"Last alert at **{ts}**")
        except Exception as e:
            st.warning(f"Alert demo error: {e}")
