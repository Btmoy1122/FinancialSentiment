import streamlit as st
import pandas as pd
from datetime import date, timedelta
from transformers import pipeline
import plotly.graph_objects as go
import yfinance as yf

from yahoo_sentiment import get_yahoo_sentiment
from reddit_sentiment import get_reddit_sentiment

# —————————————————————————————
# Page config & branding
# —————————————————————————————
st.set_page_config(page_title="FinSent Dashboard", page_icon="💹", layout="wide")
st.image("robot.png", width=120)
st.title("Financial Sentiment Analysis")

# —————————————————————————————
# Session state: advanced settings
# —————————————————————————————
if 'config' not in st.session_state:
    st.session_state.config = {
        'custom_subreddits': {},
        'num_posts': 10,
        'yahoo_count': 10,
        'start_date': date.today() - timedelta(days=180),
        'end_date': date.today()
    }
config = st.session_state.config

# —————————————————————————————
# Sidebar: advanced settings in expander
# —————————————————————————————
st.sidebar.header("⚙️ Advanced Settings")
with st.sidebar.expander("Show advanced controls", expanded=False):
    config['num_posts'] = st.number_input(
        "Reddit posts per subreddit",
        min_value=1, max_value=100,
        value=config['num_posts'], step=1
    )
    config['yahoo_count'] = st.number_input(
        "Yahoo articles per ticker",
        min_value=1, max_value=50,
        value=config['yahoo_count'], step=1
    )
    config['start_date'] = st.date_input(
        "Start date (for correlation)",
        value=config['start_date']
    )
    config['end_date'] = st.date_input(
        "End date (for correlation)",
        value=config['end_date']
    )
    st.markdown("#### Custom Subreddits")
    tickers_temp = st.session_state.get('tickers', ['META','AAPL','TSLA'])
    custom = {}
    for t in tickers_temp:
        subs = st.text_input(
            f"{t} subreddit(s) (comma-separated)",
            ', '.join(config['custom_subreddits'].get(t, []))
        )
        if subs:
            custom[t] = [s.strip() for s in subs.split(',') if s.strip()]
    config['custom_subreddits'] = custom

# —————————————————————————————
# Main UI: ticker input & run button
# —————————————————————————————
tickers_input = st.text_input(
    "Stock Tickers (comma-separated)",
    "META, AAPL, TSLA"
)
tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
st.session_state['tickers'] = tickers

run = st.button("Run Analysis")

# —————————————————————————————
# Helpers: cache pipeline & price loader
# —————————————————————————————
@st.cache_resource
def load_pipeline():
    return pipeline("text-classification", model="ProsusAI/finbert", truncation=True)

@st.cache_data(ttl=3600)
def load_price_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, interval="1d")
    if isinstance(df.columns, pd.MultiIndex):
        price_df = df.get("Adj Close", df.get("Close"))
    else:
        price_df = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    return price_df

pipe = load_pipeline()

def get_sentiment_for_tickers(tickers, custom_subreddits, num_posts, yahoo_count):
    reddit_scores, yahoo_scores, detailed = [], [], []
    default_subreddits = {
        'META': ['technology','stocks'],
        'AAPL': ['apple','stocks'],
        'TSLA': ['teslamotors','stocks']
    }
    default_keywords = {
        'META': 'meta',
        'AAPL': 'apple',
        'TSLA': 'tesla'
    }
    for t in tickers:
        kw = default_keywords.get(t, t)
        subs = custom_subreddits.get(t) or default_subreddits.get(t, [t])

        y_score, y_df = get_yahoo_sentiment(
            t, kw, pipe=pipe, verbose=False,
            return_articles=True, max_articles=yahoo_count
        )
        r_score, r_df = get_reddit_sentiment(
            kw, subs, num_posts=num_posts, pipe=pipe,
            verbose=False, return_posts=True
        )

        reddit_scores.append(r_score)
        yahoo_scores.append(y_score)
        detailed.append({
            'ticker': t,
            'yahoo_articles': y_df,
            'reddit_posts': r_df
        })
    return reddit_scores, yahoo_scores, detailed


def plot_sentiment_chart(tickers, reddit_scores, yahoo_scores):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=tickers, y=reddit_scores, name='Reddit', marker_color='indianred'))
    fig.add_trace(go.Bar(x=tickers, y=yahoo_scores, name='Yahoo', marker_color='lightslategray'))
    fig.update_layout(
        title="Sentiment Comparison by Ticker",
        xaxis_title="Ticker",
        yaxis_title="Avg Sentiment",
        plot_bgcolor=None,
        paper_bgcolor=None
    )
    st.plotly_chart(fig, use_container_width=True)

# —————————————————————————————
# Run analysis & display
# —————————————————————————————
if run and tickers:
    with st.spinner("Fetching sentiment and price data…"):
        rs, ys, details = get_sentiment_for_tickers(
            tickers,
            config['custom_subreddits'],
            config['num_posts'],
            config['yahoo_count']
        )
        prices = load_price_data(
            tickers,
            config['start_date'],
            config['end_date']
        )
    st.success("Analysis complete!")

    # KPI metrics
    st.markdown("## Overall Sentiment Scores")
    metric_cols = st.columns(len(tickers))
    for idx, t in enumerate(tickers):
        avg_score = (ys[idx] + rs[idx]) / 2
        delta = ys[idx] - rs[idx]
        metric_cols[idx].metric(label=t, value=f"{avg_score:.2f}", delta=f"{delta:+.2f}")

    # Tabs: Graph & Details
    tabs = st.tabs(["📊 Graph","📰 Details"])    
    with tabs[0]:
        plot_sentiment_chart(tickers, rs, ys)
    with tabs[1]:
        for res in details:
            st.subheader(f"Details for {res['ticker']}")
            st.write("**Yahoo Articles:**")
            if not res['yahoo_articles'].empty:
                st.dataframe(res['yahoo_articles'])
            else:
                st.info("No Yahoo articles found.")
            st.write("**Reddit Posts:**")
            if not res['reddit_posts'].empty:
                st.dataframe(res['reddit_posts'])
            else:
                st.info("No Reddit posts found.")

elif not run:
    st.info("Enter tickers and click Run Analysis to start.")
elif not tickers:
    st.warning("Please enter at least one ticker.")
