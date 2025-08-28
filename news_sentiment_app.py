# news_sentiment_app.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from news_sentiment_utils import fetch_google_news_sentiment  # ✅ Make sure this is correct

st.set_page_config(page_title="News Sentiment", layout="wide")
st.title("📰 News Sentiment Analysis (Standalone Tab 5 Test)")

# --- Sidebar: Stock Input ---
st.sidebar.header("📘 Settings")
stock_input = st.sidebar.text_input("Enter NSE Symbol (e.g., RELIANCE)", value="RELIANCE")
max_headlines = st.sidebar.slider("Number of Headlines", 5, 25, 10)
run_sentiment = st.sidebar.button("🔍 Analyze Sentiment")

if run_sentiment:
    st.markdown(f"### 🔎 Analyzing: {stock_input}")
    search_ticker = stock_input.strip().upper()
    try:
        # ✅ Fetch news
        news_df = fetch_google_news_sentiment(ticker=search_ticker, max_items=max_headlines)
        
        if news_df.empty:
            st.warning("⚠️ No recent news found. Try another symbol.")
        else:
            st.success(f"✅ Fetched {len(news_df)} headlines.")
            st.dataframe(news_df[["Published", "Headline", "Sentiment", "Score"]], use_container_width=True)

            # 📊 Pie chart
            st.markdown("### 🥧 Sentiment Distribution")
            sentiment_counts = news_df["Sentiment"].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                hole=0.3,
                marker=dict(colors=["#6BCB77", "#FFD93D", "#FF6B6B"])
            )])
            fig.update_layout(title="Sentiment Pie Chart", height=400)
            st.plotly_chart(fig, use_container_width=True)

            # 📥 Download
            csv = news_df.to_csv(index=False).encode()
            st.download_button("📥 Download News Sentiment CSV", data=csv, file_name=f"{search_ticker}_news_sentiment.csv")
    except Exception as e:
        st.error(f"❌ Error fetching sentiment: {e}")
