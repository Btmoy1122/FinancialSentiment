# app.py

import streamlit as st
from compare_sentiment import compare_sentiment
from yahoo_sentiment import get_yahoo_sentiment
from reddit_sentiment import get_reddit_sentiment
from transformers import pipeline
import os
import sys
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg at the start of your app

# Handle torch module edge case
if 'torch' in sys.modules:
    import types
    import torch
    torch.classes = types.SimpleNamespace()

os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

@st.cache_resource
def load_pipeline():
    return pipeline("text-classification", model="ProsusAI/finbert", truncation=True)

pipe = load_pipeline()

st.title('📈 Financial Sentiment Analysis')

with st.sidebar:
    st.header("🔧 Input Settings")
    tickers_input = st.text_input('Enter Stock Tickers (comma-separated)', 'META, AAPL, TSLA')
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()]

    custom_subreddits = {}
    for ticker in tickers:
        subs = st.text_input(f'Subreddits for {ticker}', '')
        if subs:
            custom_subreddits[ticker] = [sub.strip() for sub in subs.split(',') if sub.strip()]

    analyze_button = st.button('Analyze Sentiment')

# Default fallback subreddits if user leaves blank
default_subreddits = {
    'META': ['technology', 'stocks'],
    'AAPL': ['apple', 'stocks'],
    'TSLA': ['teslamotors', 'stocks']
}
default_keywords = {
    'META': 'meta',
    'AAPL': 'apple',
    'TSLA': 'tesla'
}

def get_sentiment_for_tickers(tickers, custom_subreddits):
    reddit_scores = []
    yahoo_scores = []
    detailed_results = []

    for ticker in tickers:
        keyword = default_keywords.get(ticker, ticker)
        subreddits = custom_subreddits.get(ticker) or default_subreddits.get(ticker, [ticker])

        st.subheader(f"Sentiment for {ticker}")
        yahoo_score, yahoo_articles = get_yahoo_sentiment(ticker, keyword, pipe=pipe, verbose=False, return_articles=True)
        reddit_score, reddit_posts = get_reddit_sentiment(keyword, subreddits, pipe=pipe, verbose=False, return_posts=True)

        reddit_scores.append(reddit_score)
        yahoo_scores.append(yahoo_score)

        detailed_results.append({
            "ticker": ticker,
            "yahoo_articles": yahoo_articles,
            "reddit_posts": reddit_posts
        })

    return reddit_scores, yahoo_scores, detailed_results

if analyze_button and tickers:
    st.success(f"Analyzing sentiment for {', '.join(tickers)}...")

    reddit_scores, yahoo_scores, detailed_results = get_sentiment_for_tickers(tickers, custom_subreddits)

    tabs = st.tabs(["📊 Sentiment Graph", "📰 Detailed Results"])

    with tabs[0]:  # Graph Tab
        import matplotlib.pyplot as plt

        plt.switch_backend('Agg')
        fig, ax = plt.subplots()
        bar_width = 0.35
        x = range(len(tickers))

        bars1 = ax.bar([i - bar_width/2 for i in x], reddit_scores, width=bar_width, label='Reddit')
        bars2 = ax.bar([i + bar_width/2 for i in x], yahoo_scores, width=bar_width, label='Yahoo')

        ax.set_ylabel('Sentiment Score')
        ax.set_title('Sentiment Comparison by Ticker')
        ax.set_xticks(list(x))
        ax.set_xticklabels(tickers)
        ax.legend()
        ax.axhline(0, color='black', linewidth=0.8)

        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

        plt.tight_layout()
        st.pyplot(fig)

    with tabs[1]:  # Detailed Results Tab
        for result in detailed_results:
            ticker = result["ticker"]
            st.subheader(f"Details for {ticker}")

            st.write("**Yahoo Finance Articles:**")
            if not result["yahoo_articles"].empty:
                st.dataframe(result["yahoo_articles"])
            else:
                st.info("No matching Yahoo Finance articles found.")

            # …

            if not result["reddit_posts"].empty:
                st.dataframe(result["reddit_posts"])
            else:
                st.info("No matching Reddit posts found.")


else:
    st.info("Enter tickers and click 'Analyze Sentiment' to start.")
