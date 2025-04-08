import requests

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification", model="ProsusAI/finbert")


keyword = 'meta'
date = '2024-08-18'

API_KEY = open('API_KEY').read().strip()
headers = {
    'x-api-key': API_KEY
}

url = (
    'https://newsapi.org/v2/everything?' \
    f'q={keyword}&' +
    f'from={date}&' +
    'sortBy=popularity'
)
print(url)


response = requests.get(url, headers=headers)

data = response.json()
if 'articles' not in data:
    print("API Error:", data)
    exit()

articles = data['articles']

articles = response.json()['articles']
articles = [article for article in articles if keyword.lower() in article['title'].lower() or keyword.lower() in article['description'].lower()]

total_score = 0
num_articles = 0

for i, article in enumerate(articles):

    print(f'Title: {article["title"]}')
    print(f'Link: {article["url"]}')
    print(f'Published: {article["description"]}')
    

    sentiment = pipe(article['content'])[0]

    print(f'Sentiment {sentiment["label"]}, Score: {sentiment["score"]}')
    print('-'*40)

    if sentiment['label'] == 'positive':
        total_score += sentiment['score']
        num_articles+=1
    elif sentiment['label'] == 'negative':
        total_score -= sentiment['score']
        num_articles+=1

final_score = total_score/num_articles
print(f'Overall Sentiment: {"Positive" if final_score>=0.15 else "Negative" if final_score <=-0.15 else "Neutral"} {final_score}')

