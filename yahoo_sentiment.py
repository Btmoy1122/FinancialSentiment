import feedparser
from transformers import pipeline
import pandas as pd

def get_yahoo_sentiment(ticker, keyword, pipe=None, verbose=True, return_articles=False):
    if pipe is None:
        pipe = pipeline("text-classification", model="ProsusAI/finbert")

    rss_url = f'https://finance.yahoo.com/rss/headline?s={ticker}'
    feed = feedparser.parse(rss_url)

    total_score = 0
    num_articles = 0
    positive_count = 0
    negative_count = 0
    articles_data = []

    if verbose:
        print(f"\nAnalyzing Yahoo Finance sentiment for keyword: {keyword}\n")

    for entry in feed.entries:
        summary = entry.get('summary', '')
        title = entry.get('title', '')

        if not summary or keyword.lower() not in (summary + title).lower():
            continue

        try:
            sentiment = pipe(summary)[0]
        except Exception as e:
            if verbose:
                print(f"Error processing entry: {e}")
            continue

        score_adjustment = sentiment['score'] if sentiment['label'] == 'positive' else -sentiment['score']
        total_score += score_adjustment
        num_articles += 1
        if sentiment['label'] == 'positive':
            positive_count += 1
        elif sentiment['label'] == 'negative':
            negative_count += 1

        articles_data.append({
            "Title": title,
            "Link": entry.get('link', ''),
            "Published": entry.get('published', ''),
            "Sentiment": sentiment['label'],
            "Score": round(sentiment['score'], 4)
        })

    final_score = (total_score / num_articles) if num_articles > 0 else 0.0

    if verbose:
        print(f"\nYahoo Sentiment: {'Positive' if final_score >= 0.15 else 'Negative' if final_score <= -0.15 else 'Neutral'} ({final_score:.4f})")
        print(f"Total Positive Articles: {positive_count}")
        print(f"Total Negative Articles: {negative_count}")

    if return_articles:
        return final_score, pd.DataFrame(articles_data)
    else:
        return final_score
