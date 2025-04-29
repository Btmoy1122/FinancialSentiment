import feedparser
from transformers import pipeline
import pandas as pd

def get_yahoo_sentiment(
    ticker,
    keyword,
    pipe=None,
    verbose=True,
    return_articles=False,
    max_articles=None,          # <-- add this parameter
):
    """
    max_articles: if set, only process up to that many matching entries
    """
    if pipe is None:
        pipe = pipeline("text-classification", model="ProsusAI/finbert")

    rss_url = f'https://finance.yahoo.com/rss/headline?s={ticker}'
    feed = feedparser.parse(rss_url)

    total_score = 0
    num_articles = 0
    positive_count = 0
    negative_count = 0
    articles_data = []
    processed = 0               # <-- counter for matched entries

    if verbose:
        print(f"\nAnalyzing Yahoo Finance sentiment for keyword: {keyword}\n")

    for entry in feed.entries:
        if max_articles is not None and processed >= max_articles:
            break

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

        score_adjustment = sentiment['score'] if sentiment['label']=='positive' else -sentiment['score']
        total_score += score_adjustment
        num_articles += 1
        if sentiment['label']=='positive':
            positive_count += 1
        elif sentiment['label']=='negative':
            negative_count += 1

        articles_data.append({
            "Title": title,
            "Link": entry.get('link', ''),
            "Published": entry.get('published', ''),
            "Sentiment": sentiment['label'],
            "Score": round(sentiment['score'], 4)
        })
        processed += 1           # <-- increment on each matched entry

    final_score = (total_score / num_articles) if num_articles>0 else 0.0

    if verbose:
        print(f"\nYahoo Sentiment: {final_score:.4f} ({'Positive' if final_score>=0.15 else 'Negative' if final_score<=-0.15 else 'Neutral'})")
        print(f"Total Positive: {positive_count}, Total Negative: {negative_count}")

    if return_articles:
        return final_score, pd.DataFrame(articles_data)
    else:
        return final_score
