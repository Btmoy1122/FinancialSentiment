import feedparser
from transformers import pipeline

def get_yahoo_sentiment(ticker, keyword, verbose=True):
    pipe = pipeline("text-classification", model="ProsusAI/finbert")
    rss_url = f'https://finance.yahoo.com/rss/headline?s={ticker}'
    feed = feedparser.parse(rss_url)

    total_score = 0
    num_articles = 0
    positive_count = 0
    negative_count = 0

    if verbose:
        print(f"\nAnalyzing Yahoo Finance sentiment for keyword: {keyword}\n")

    for i, entry in enumerate(feed.entries):
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

        #if verbose:
            #print(f"Title: {title}")
            #print(f"Link: {entry.get('link', 'N/A')}")
            #print(f"Published: {entry.get('published', 'N/A')}")
            #print(f"Summary: {summary}")
            #print(f"Sentiment: {sentiment['label']}, Score: {sentiment['score']:.4f}")
            #print('-'*40)

        if sentiment['label'] == 'positive':
            total_score += sentiment['score']
            num_articles += 1
            positive_count += 1
        elif sentiment['label'] == 'negative':
            total_score -= sentiment['score']
            num_articles += 1
            negative_count += 1

    if num_articles > 0:
        final_score = total_score / num_articles
        if verbose:
            print(f"\nYahoo Sentiment: {'Positive' if final_score >= 0.15 else 'Negative' if final_score <= -0.15 else 'Neutral'} ({final_score:.4f})")
            print(f"Total Positive Articles: {positive_count}")
            print(f"Total Negative Articles: {negative_count}")
        return final_score
    else:
        if verbose:
            print("\nNo matching articles found for sentiment analysis.")
        return 0.0
