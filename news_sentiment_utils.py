# news_sentiment_utils.py

import feedparser
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def fetch_google_news_sentiment(ticker="RELIANCE.NS", max_items=10):
    try:
        query = ticker.replace(".NS", "")
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        print(f"[🔍] Fetching RSS for: {rss_url}")
        feed = feedparser.parse(rss_url)

        if not feed.entries:
            print("[⚠️] No entries found. Trying fallback...")
            fallback_url = f"https://news.google.com/rss/search?q={query}+stock&hl=en-IN&gl=IN&ceid=IN:en"
            feed = feedparser.parse(fallback_url)

        analyzer = SentimentIntensityAnalyzer()
        news_data = []

        for entry in feed.entries[:max_items]:
            headline = entry.title
            link = entry.link
            published = entry.get("published", "N/A")

            score = analyzer.polarity_scores(headline)["compound"]
            sentiment = "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"

            news_data.append({
                "Published": published,
                "Headline": headline,
                "Sentiment": sentiment,
                "Score": score,
                "Link": link
            })

        df = pd.DataFrame(news_data)
        print(f"[✅] Parsed {len(df)} headlines")
        return df

    except Exception as e:
        print(f"[❌ ERROR] Failed to fetch sentiment: {e}")
        return pd.DataFrame()


# Test block
if __name__ == "__main__":
    ticker = "RELIANCE"  # or "TCS", "HDFCBANK", etc.
    df = fetch_google_news_sentiment(ticker, max_items=10)
    if df.empty:
        print("❌ No news found.")
    else:
        print(df[["Published", "Headline", "Sentiment", "Score"]])
