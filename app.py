import streamlit as st
from compare_sentiment import compare_sentiment
from yahoo_sentiment import get_yahoo_sentiment
from reddit_sentiment import get_reddit_sentiment
from transformers import pipeline
import os
import sys
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg at the start of your app

if 'torch' in sys.modules:
    import types
    import torch
    torch.classes = types.SimpleNamespace()

os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

@st.cache_resource  # Cache the pipeline to avoid reloading
def load_pipeline():
    return pipeline("text-classification", model="ProsusAI/finbert", truncation=True)

pipe = load_pipeline()

def get_sentiment_for_tickers(tickers):
    subreddits_map = {
        'META': ['technology', 'stocks'],
        'AAPL': ['apple', 'stocks'],
        'TSLA': ['teslamotors', 'stocks']
    }

    keywords_map = {
        'META': 'meta',
        'AAPL': 'apple',
        'TSLA': 'tesla'
    }
    
    compare_sentiment(tickers, subreddits_map, keywords_map)

st.title('Financial Sentiment Analysis')
tickers_input = st.text_input('Enter Stock Tickers (comma-separated)', 'META, AAPL, TSLA')
tickers = [ticker.strip() for ticker in tickers_input.split(',')]

if st.button('Analyze Sentiment'):
    if tickers:
        st.write(f"Analyzing sentiment for tickers: {', '.join(tickers)}")
        get_sentiment_for_tickers(tickers)
    else:
        st.warning('Please enter at least one ticker to analyze.')