import praw
from transformers import pipeline
import time
import pandas as pd

def get_reddit_sentiment(keyword, subreddits, num_posts=10, time_filter='day', verbose=True, sleep_time=1, pipe=None, return_posts=False):
    if pipe is None:
        pipe = pipeline("text-classification", model="ProsusAI/finbert")

    reddit = praw.Reddit(
        client_id="jQ_STKV5q9d3REJqfqAzxQ",
        client_secret=None,
        user_agent="financial sentiment analyzer by u/Time-Technician-3141"
    )

    total_score = 0
    num_analyzed = 0
    positive_count = 0
    negative_count = 0
    posts_data = []

    if verbose:
        print(f"\nAnalyzing Reddit sentiment for keyword: {keyword}\n")

    for sub in subreddits:
        if verbose:
            print(f"--- Searching r/{sub} ---")
        subreddit = reddit.subreddit(sub)

        try:
            for post in subreddit.search(keyword, sort='new', time_filter=time_filter, limit=num_posts):
                content = f"{post.title} {post.selftext}".strip()

                if not content or keyword.lower() not in content.lower():
                    continue

                sentiment = pipe(content)[0]
                score_adjustment = sentiment['score'] if sentiment['label'] == 'positive' else -sentiment['score']
                total_score += score_adjustment
                num_analyzed += 1
                if sentiment['label'] == 'positive':
                    positive_count += 1
                elif sentiment['label'] == 'negative':
                    negative_count += 1

                posts_data.append({
                    "Title": post.title,
                    "Link": post.url,
                    "Sentiment": sentiment['label'],
                    "Score": round(sentiment['score'], 4)
                })

                time.sleep(sleep_time)

        except Exception as e:
            if verbose:
                print(f"Error processing subreddit {sub}: {e}")

    final_score = (total_score / num_analyzed) if num_analyzed > 0 else 0.0

    if verbose:
        print(f"\nReddit Sentiment: {'Positive' if final_score >= 0.15 else 'Negative' if final_score <= -0.15 else 'Neutral'} ({final_score:.4f})")
        print(f"Total Positive Posts: {positive_count}")
        print(f"Total Negative Posts: {negative_count}")

    if return_posts:
        return final_score, pd.DataFrame(posts_data)
    else:
        return final_score
