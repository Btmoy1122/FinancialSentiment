import matplotlib.pyplot as plt
from reddit_sentiment import get_reddit_sentiment
from yahoo_sentiment import get_yahoo_sentiment
import argparse

def compare_sentiment(tickers, subreddits_map, keyword_map):
    reddit_scores = []
    yahoo_scores = []

    for ticker in tickers:
        keyword = keyword_map.get(ticker, ticker)
        subreddits = subreddits_map.get(ticker, [ticker])

        print(f"\n===== {ticker} =====")
        yahoo_score = get_yahoo_sentiment(ticker, keyword)
        reddit_score = get_reddit_sentiment(keyword, subreddits)

        yahoo_scores.append(yahoo_score)
        reddit_scores.append(reddit_score)

    plot_sentiment_comparison(tickers, reddit_scores, yahoo_scores)

def plot_sentiment_comparison(tickers, reddit_scores, yahoo_scores):
    x = range(len(tickers))
    bar_width = 0.35

    fig, ax = plt.subplots()
    bars1 = ax.bar([i - bar_width/2 for i in x], reddit_scores, width=bar_width, label='Reddit')
    bars2 = ax.bar([i + bar_width/2 for i in x], yahoo_scores, width=bar_width, label='Yahoo')

    ax.set_ylabel('Sentiment Score')
    ax.set_title('Sentiment Comparison by Ticker')
    ax.set_xticks(x)
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
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare Reddit and Yahoo sentiment for given tickers.')
    parser.add_argument('--tickers', nargs='+', required=True, help='List of stock tickers to analyze')
    args = parser.parse_args()

    # You can customize subreddit and keyword mapping here
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

    compare_sentiment(args.tickers, default_subreddits, default_keywords)
