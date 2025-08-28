# utils_live_sentiment.py
import pandas as pd
import streamlit as st
from news_sentiment_utils import fetch_google_news_sentiment

@st.cache_data(ttl=180)  # refresh every 3 minutes
def get_live_daily_sentiment(ticker: str) -> pd.Series:
    """
    Returns a DAILY sentiment signal (float) indexed by date, with real variation.
    1) Pull latest headlines (e.g., Google News / RSS)
    2) Score with VADER compound
    3) Aggregate by date
    4) Z-score normalize & clip to avoid flat/zero variance series
    """
    import pandas as pd
    import numpy as np
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import feedparser
    from datetime import datetime

    # 1) Fetch headlines (adjust your feed / query here)
    # Example: Google News RSS search
    query = ticker.replace(" ", "+")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)

    rows = []
    for entry in feed.entries[:50]:
        # published_parsed may be missing; guard it
        try:
            dt_pub = datetime(*entry.published_parsed[:6])
        except Exception:
            continue
        rows.append({"published": pd.Timestamp(dt_pub).normalize(), "title": entry.title or ""})

    if not rows:
        return pd.Series(dtype=float)

    df = pd.DataFrame(rows)
    sia = SentimentIntensityAnalyzer()
    df["compound"] = df["title"].astype(str).apply(lambda x: sia.polarity_scores(x)["compound"])

    # 2) Aggregate by day
    daily = df.groupby("published")["compound"].mean().sort_index()

    # 3) Normalize (z-score) to ensure non-flat signal
    if daily.std(ddof=0) > 1e-6:
        z = (daily - daily.mean()) / (daily.std(ddof=0) + 1e-9)
        daily = z.clip(-3, 3)
    else:
        # If still near-constant, return empty to signal "no usable exog"
        return pd.Series(dtype=float)

    daily.index.name = "Date"
    return daily


def align_sentiment_to_index(sent_daily: pd.Series, price_index: pd.DatetimeIndex) -> pd.Series:
    """
    Align daily sentiment to your price index:
    - Reindex to DAILY dates covering price range
    - FFill to cover missing days
    - Reindex again to price_index (keeps business days)
    - Return standardized (z-score) again on the aligned slice
    """
    import pandas as pd
    import numpy as np

    if sent_daily is None or len(sent_daily) == 0:
        return pd.Series(dtype=float, index=price_index)

    # Cover the price index span
    full_daily = sent_daily.reindex(
        pd.date_range(price_index.min().normalize(), price_index.max().normalize(), freq="D")
    ).ffill()

    aligned = full_daily.reindex(price_index, method="ffill")

    # Standardize on the aligned window to avoid near-flat scaling after alignment
    if aligned.std(ddof=0) <= 1e-6 or aligned.isna().all():
        return pd.Series(dtype=float, index=price_index)

    z = (aligned - aligned.mean()) / (aligned.std(ddof=0) + 1e-9)
    return z.clip(-3, 3)

