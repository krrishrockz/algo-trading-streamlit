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
from urllib.parse import quote_plus
import yfinance as yf

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
        # 1) Prefer the repo copy (zip or extracted dir)
        repo_sentiment_zip = os.path.join(
            _repo_nltk_path(), "sentiment", "vader_lexicon.zip"
        )
        repo_sentiment_dir = os.path.join(
            _repo_nltk_path(), "sentiment", "vader_lexicon"
        )

        if os.path.exists(repo_sentiment_zip) or os.path.exists(repo_sentiment_dir):
            # Ensure nltk looks into ./nltk_data first
            if _repo_nltk_path() not in nltk.data.path:
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
    # Lazy import after we know the data path is correct
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


# ✅ NEW helper: keep your existing call sites intact
def _ensure_vader():
    """
    Back-compat shim for code that calls _ensure_vader().
    Returns a ready-to-use SentimentIntensityAnalyzer.
    """
    return _get_analyzer()


# ------------------------------------------------------------
# Headline fetching (Google News RSS)
# ------------------------------------------------------------
from urllib.parse import quote_plus

def _google_news_feed_url(query: str) -> str:
    q = f'{query} (stock OR shares) when:7d'
    return f"https://news.google.com/rss/search?q={quote_plus(q)}&hl=en-IN&gl=IN&ceid=IN:en"


def fetch_headlines_google_news(symbol: str, max_items: int = 50, timeout: int = 8) -> pd.DataFrame:
    """
    Broaden query to include company name; use title+summary for sentiment.
    """
    try:
        base = symbol.replace(".NS", "").replace(".BO", "").strip()
        company = None
        try:
            info = yf.Ticker(symbol).info or {}
            company = info.get("shortName") or info.get("longName")
        except Exception:
            company = None

        q = f'"{base}"'
        if company:
            q = f'("{base}" OR "{company}")'

        url = _google_news_feed_url(q)
        feed = feedparser.parse(url)

        rows = []
        for entry in feed.get("entries", [])[:max_items]:
            title = (entry.get("title") or "").strip()
            summary = (entry.get("summary") or entry.get("description") or "").strip()
            text = (title + " " + summary).strip()
            link = (entry.get("link") or "").strip()
            p = entry.get("published_parsed") or entry.get("updated_parsed")
            if not p:
                continue
            pub_dt_utc = dt.datetime.fromtimestamp(time.mktime(p), tz=dt.timezone.utc)
            rows.append({"published": pub_dt_utc, "text": text, "link": link})

        return pd.DataFrame(rows, columns=["published", "text", "link"]) if rows else \
               pd.DataFrame(columns=["published", "text", "link"])
    except Exception as e:
        print(f"⚠️ fetch_headlines_google_news error: {e}")
        return pd.DataFrame(columns=["published", "text", "link"])


def fetch_headlines_yf_news(symbol: str, max_items: int = 50) -> pd.DataFrame:
    """
    Fallback source: yfinance Ticker.news
    Returns columns ['published','text','link'] in UTC.
    """
    try:
        rows = []
        news = yf.Ticker(symbol).news or []
        for item in news[:max_items]:
            title = (item.get("title") or "").strip()
            provider = (item.get("publisher") or "").strip()
            text = f"{title} {provider}".strip()
            ts = item.get("providerPublishTime") or item.get("published") or None
            if not ts:
                continue
            pub_dt_utc = dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc)
            rows.append({"published": pub_dt_utc, "text": text, "link": item.get("link", "")})
        return pd.DataFrame(rows, columns=["published","text","link"])
    except Exception:
        return pd.DataFrame(columns=["published","text","link"])


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


def compute_daily_compound(df_news: pd.DataFrame) -> pd.Series:
    """
    Expect a 'text' column (title+summary). Group by local date and average VADER compound.
    """
    if df_news is None or df_news.empty:
        return pd.Series(dtype=float)

    sid = _ensure_vader()  # your existing getter/auto-download (now defined)
    df = df_news.copy()
    df["published"] = pd.to_datetime(df["published"], utc=True, errors="coerce")
    df.dropna(subset=["published"], inplace=True)
    df["date_local"] = df["published"].dt.tz_convert("Asia/Kolkata").dt.date
    df["compound"] = df["text"].fillna("").apply(lambda t: sid.polarity_scores(t)["compound"])
    out = df.groupby("date_local")["compound"].mean()
    out.index = pd.to_datetime(out.index)
    out.index = out.index.tz_localize("Asia/Kolkata").tz_convert("UTC").tz_localize(None)
    return out.sort_index()


def get_live_daily_sentiment(symbol: str, max_items: int = 80, lookback_days: int = 21):
    """
    1) Google News (title+summary)
    2) Fallback: Yahoo Finance news
    3) Final fallback: price-return proxy (z-scored)
    Returns (series, source_label)
    """
    # 1) Google
    df_g = fetch_headlines_google_news(symbol, max_items=max_items)
    s_g = compute_daily_compound(df_g)
    if not s_g.empty and s_g.std() > 1e-6:
        return s_g.tail(lookback_days), "Google News"

    # 2) Yahoo Finance fallback
    df_y = fetch_headlines_yf_news(symbol, max_items=max_items)
    s_y = compute_daily_compound(df_y)
    if not s_y.empty and s_y.std() > 1e-6:
        return s_y.tail(lookback_days), "Yahoo Finance News"

    # 3) Price return proxy fallback
    try:
        hist = yf.Ticker(symbol).history(period=f"{max(lookback_days+5, 30)}d", interval="1d")
        if not hist.empty and "Close" in hist:
            ret = hist["Close"].pct_change().dropna()
            # z-score
            z = (ret - ret.mean()) / (ret.std() if ret.std() else 1.0)
            z.name = "compound_proxy"
            return z.tail(lookback_days), "Price Return Proxy"
    except Exception:
        pass

    return pd.Series(dtype=float), "None"


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
