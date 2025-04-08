import praw
from transformers import pipeline
import time
    

def get_reddit_sentiment(keyword, subreddits, num_posts=10, time_filter='day', verbose=True, sleep_time=1):
    
    
    pipe = pipeline("text-classification", model="ProsusAI/finbert")

    # Initialize Reddit API (replace with your credentials)
    reddit = praw.Reddit(
        client_id="jQ_STKV5q9d3REJqfqAzxQ",
        client_secret=None,
        user_agent="financial sentiment analyzer by u/Time-Technician-3141"
    )

    total_score = 0
    num_analyzed = 0
    positive_count = 0
    negative_count = 0

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

                #if verbose:
                    #print(f"Title: {post.title}")
                    #print(f"Sentiment: {sentiment['label']} (Score: {sentiment['score']:.4f})")
                    #print(f"Link: {post.url}")
                    #print("-" * 50)

                if sentiment['label'] == 'positive':
                    total_score += sentiment['score']
                    num_analyzed += 1
                    positive_count += 1
                elif sentiment['label'] == 'negative':
                    total_score -= sentiment['score']
                    num_analyzed += 1
                    negative_count += 1

                time.sleep(sleep_time)  # To avoid rate-limiting

        except Exception as e:
            if verbose:
                print(f"Error processing subreddit {sub}: {e}")

    if num_analyzed > 0:
        final_score = total_score / num_analyzed
        if verbose:
            print(f"\nReddit Sentiment: {'Positive' if final_score >= 0.15 else 'Negative' if final_score <= -0.15 else 'Neutral'} ({final_score:.4f})")
            print(f"Total Positive Posts: {positive_count}")
            print(f"Total Negative Posts: {negative_count}")
        return final_score
    else:
        if verbose:
            print("\nNo matching posts found for sentiment analysis.")
        return 0.0
