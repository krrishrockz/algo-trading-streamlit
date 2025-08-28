# utils_live_sentiment.py
import pandas as pd
import streamlit as st
from news_sentiment_utils import fetch_google_news_sentiment

@st.cache_data(ttl=180)  # refresh every 3 minutes
def get_live_daily_sentiment(ticker: str) -> pd.Series:
    """
    Pull latest Google News headlines, score with VADER (already in your util),
    and return a daily Series of average sentiment (index normalized to date).
    """
    df = fetch_google_news_sentiment(ticker=ticker.replace(".NS", ""))
    if df is None or df.empty:
        return pd.Series(dtype=float)

    # Ensure datetime + normalize to dates
    pub = pd.to_datetime(df["Published"], errors="coerce").dt.normalize()
    scores = pd.to_numeric(df["Score"], errors="coerce")
    daily = (
        pd.DataFrame({"Published": pub, "Score": scores})
        .dropna()
        .groupby("Published")["Score"].mean()
        .sort_index()
    )
    return daily

def align_sentiment_to_index(sent_daily: pd.Series, price_index: pd.DatetimeIndex) -> pd.Series:
    """
    Forward-fill daily sentiment onto your price index (e.g., business-day close series).
    Returns a float Series aligned 1:1 to price_index; neutral (0.0) if no news.
    """
    if sent_daily is None or sent_daily.empty:
        return pd.Series(0.0, index=price_index)

    # Normalize the price index to dates, then forward fill sentiment
    idx = pd.DatetimeIndex(price_index).normalize()
    s = sent_daily.reindex(idx).ffill()
    return s.fillna(0.0)
