from transformers import pipeline
import yfinance as yf
from goose3 import Goose 
from requests import get
import pandas as pd
import os

class SentimentAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.news_list = self.get_ticker_news()
        self.extractor = Goose()
        self.pipe = pipeline("text-classification", model="ProsusAI/finbert")
        self.sentiment_data = self.get_sentiment_data()

    def get_ticker_news(self):
        ticker_news = yf.Ticker(self.ticker)
        return ticker_news.get_news()

    def get_sentiment_data(self):
        data = []
        for news in self.news_list:
            title = news['title']
            response = get(news['link'])
            article = self.extractor.extract(raw_html=response.content)
            text = article.cleaned_text
            date = article.publish_date
            publisher = news['publisher']
            link = news['link']
            if len(text) > 512:
                data.append({
                    'Date': f'{date}',
                    'Article sentiment': 'Nan too long'
                })
            else:
                result = self.pipe(text)
                data.append({
                    'Date': f'{date}',
                    'Article title': f'{title}',
                    'link': f'{link}',
                    'publisher': f'{publisher}',
                    'Article sentiment': result[0]['label']
                })
        df = pd.DataFrame(data)
        return df

    def generate_csv_and_json(self):
        # Create the 'out' directory if it doesn't exist
        if not os.path.exists('out'):
            os.makedirs('out')
        
        # Save DataFrame to CSV
        self.sentiment_data.to_csv(f'out/{self.ticker}_sentiment.csv', index=False)
        
        # Save DataFrame to JSON
        self.sentiment_data.to_json(f'out/{self.ticker}_sentiment.json', orient='records')
