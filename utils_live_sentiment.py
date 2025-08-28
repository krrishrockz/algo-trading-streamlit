# utils_live_sentiment.py
# ------------------------------------------------------------
# Live news sentiment utilities for SARIMAX exogenous input
# ------------------------------------------------------------

import os
import time
import datetime as dt
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import feedparser
import pytz
import nltk

# We import SentimentIntensityAnalyzer lazily AFTER ensuring the lexicon exists.


# ------------------------------------------------------------
# VADER availability: prefer repo copy, fallback to nltk, then download
# ------------------------------------------------------------
def _repo_nltk_path() -> str:
    """
    Return the absolute path to ./nltk_data inside this repo.
    """
    return os.path.join(os.path.dirname(__file__), "nltk_data")


def ensure_vader_lexicon() -> bool:
    """
    Ensure the VADER lexicon is available.

    Priority:
      1) repo's ./nltk_data/sentiment/vader_lexicon.zip  (no network)
      2) already available in any existing nltk.data.path
      3) download via nltk (last resort)
    """
    try:
        # 1) Prefer the repo copy (zip)
        repo_sentiment_zip = os.path.join(
            _repo_nltk_path(), "sentiment", "vader_lexicon.zip"
        )
        repo_sentiment_dir = os.path.join(
            _repo_nltk_path(), "sentiment", "vader_lexicon"
        )

        if os.path.exists(repo_sentiment_zip) or os.path.exists(repo_sentiment_dir):
            # Ensure nltk looks into ./nltk_data first
            nltk.data.path.insert(0, _repo_nltk_path())

        # 2) Try to locate the resource without downloading
        #    We check both the zip and extracted dir variants.
        try:
            nltk.data.find(
                "sentiment/vader_lexicon.zip/vader_lexicon/vader_lexicon.txt"
            )
            return True
        except LookupError:
            pass

        try:
            nltk.data.find("sentiment/vader_lexicon/vader_lexicon.txt")
            return True
        except LookupError:
            pass

        # 3) Last resort: try to download (may fail on restricted hosts)
        nltk.download("vader_lexicon")
        # Verify after download:
        nltk.data.find("sentiment/vader_lexicon/vader_lexicon.txt")
        return True

    except Exception as e:
        print(f"⚠️ ensure_vader_lexicon() failed: {e}")
        return False


def _get_analyzer():
    """
    Return a VADER SentimentIntensityAnalyzer after ensuring lexicon availability.
    """
    if not ensure_vader_lexicon():
        raise RuntimeError("VADER lexicon could not be loaded.")
    from nltk.sentiment.vader import SentimentIntensityAnalyzer  # lazy import
    return SentimentIntensityAnalyzer()


# ------------------------------------------------------------
# Headline fetching (Google News RSS)
# ------------------------------------------------------------
def _google_news_feed_url(query: str) -> str:
    """
    Build a Google News RSS search URL tuned for India/English feeds.
    """
    # You can tweak the query if you want broader coverage (e.g. add 'stock OR shares').
    # We also add when:7d via "q={query} when:7d" pattern by using 'q=' param directly.
    q = f"{query} stock when:7d"
    return f"https://news.google.com/rss/search?q={nltk.re.sub('[ ]+', '%20', q)}&hl=en-IN&gl=IN&ceid=IN:en"


def fetch_headlines_google_news(
    symbol: str, max_items: int = 50, timeout: int = 8
) -> pd.DataFrame:
    """
    Fetch recent headlines for a symbol from Google News RSS.

    Returns a DataFrame with columns: ['published', 'title', 'link'] in UTC.
    Empty DataFrame if nothing is found or on error.
    """
    try:
        url = _google_news_feed_url(symbol)
        # feedparser is robust; it handles redirects and tls automatically.
        feed = feedparser.parse(url)
        rows = []
        for entry in feed.get("entries", [])[:max_items]:
            t = entry.get("title", "").strip()
            lnk = entry.get("link", "").strip()
            # published_parsed is a time.struct_time (UTC)
            p = entry.get("published_parsed")
            if p is None:
                # try 'updated_parsed' fallback
                p = entry.get("updated_parsed")
            if p:
                pub_dt_utc = dt.datetime.fromtimestamp(time.mktime(p), tz=dt.timezone.utc)
            else:
                # If no time available, skip
                continue
            rows.append({"published": pub_dt_utc, "title": t, "link": lnk})

        if not rows:
            return pd.DataFrame(columns=["published", "title", "link"])

        df = pd.DataFrame(rows)
        return df

    except Exception as e:
        print(f"⚠️ fetch_headlines_google_news error: {e}")
        return pd.DataFrame(columns=["published", "title", "link"])


# ------------------------------------------------------------
# Sentiment scoring + daily aggregation (IST)
# ------------------------------------------------------------
def score_headlines_vader(titles: List[str]) -> List[float]:
    """
    Score a list of titles with VADER and return the 'compound' scores.
    """
    sid = _get_analyzer()
    scores = []
    for t in titles:
        try:
            s = sid.polarity_scores(t or "")
            scores.append(float(s.get("compound", 0.0)))
        except Exception:
            scores.append(0.0)
    return scores


def get_live_daily_sentiment(
    symbol: str, max_items: int = 50, lookback_days: int = 14
) -> pd.Series:
    """
    Fetch live headlines for `symbol`, score via VADER, and return a
    daily-mean sentiment Series (index = IST dates, dtype=float).

    If no headlines are found, returns an empty Series.
    """
    df = fetch_headlines_google_news(symbol, max_items=max_items)

    if df.empty:
        return pd.Series(dtype=float)

    # Score headlines
    df["score"] = score_headlines_vader(df["title"].astype(str).tolist())

    # Convert published times from UTC to IST and round down to date
    ist = pytz.timezone("Asia/Kolkata")
    df["published_ist"] = (
        pd.to_datetime(df["published"], utc=True)
        .dt.tz_convert(ist)
        .dt.tz_localize(None)  # make timezone-naive after conversion
    )
    df["date_ist"] = df["published_ist"].dt.date

    # Filter to lookback_days window (optional)
    if lookback_days and lookback_days > 0:
        cutoff = (dt.datetime.now(ist) - dt.timedelta(days=lookback_days)).date()
        df = df[df["date_ist"] >= cutoff]

    # Daily mean
    daily = (
        df.groupby("date_ist")["score"]
        .mean()
        .sort_index()
    )
    daily.index = pd.to_datetime(daily.index)  # ensure DatetimeIndex
    return daily


# ------------------------------------------------------------
# Alignment helper for SARIMAX exogenous array
# ------------------------------------------------------------
def align_sentiment_to_index(sent_daily: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """
    Align a daily sentiment Series to a price DatetimeIndex.

    - Normalizes both to dates
    - Reindexes to all target dates
    - Forward-fills gaps; fills remaining NaN with 0.0
    """
    if sent_daily is None or len(sent_daily) == 0:
        # Return a neutral (0.0) series aligned to target_index
        return pd.Series(np.zeros(len(target_index), dtype=float), index=target_index)

    # Normalize both to dates
    target_dates = pd.to_datetime(target_index).normalize()
    sent = sent_daily.copy()
    sent.index = pd.to_datetime(sent.index).normalize()

    # Reindex and fill forward, then zeros for any leading NaNs
    aligned = sent.reindex(target_dates).ffill().fillna(0.0)
    aligned.index = target_index  # restore original index
    return aligned
