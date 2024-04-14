from transformers import pipeline
import yfinance as yf
from goose3 import Goose 
from requests import get
import pandas as pd
import csv
import os
import json

def get_ticker_news_sentiment(ticker):
   
    ticker_news = yf.Ticker(ticker)
    news_list = ticker_news.get_news()
    
    extractor = Goose()
    pipe = pipeline("text-classification", model="ProsusAI/finbert")
    
    data = []
    for dict in news_list:
        title = dict['title']
        response = get(dict['link'])
        article = extractor.extract(raw_html=response.content)
        text = article.cleaned_text
        date = article.publish_date
        #thumbnail = dict['thumbnail']['resolutions'][1]
        publisher = dict['publisher']
        link = dict['link']
        if len(text) > 512:
            data.append({
                'Date':f'{date}',
                'Article sentiment':'Nan too long'
            })
            
        else:
            result = pipe(text)
            print(result)
            data.append({
                'Date':f'{date}',
                'Article title':f'{title}',
                'link':f'{link}',
                #'thumbnail':f'{thumbnail}',
                'publisher':f'{publisher}',
                'Article sentiment':result[0]['label']
            })
    #print(data)       
    df = pd.DataFrame(data)
    print(df)
    return df

def generate_csv_and_json(ticker):
    # Create the 'out' directory if it doesn't exist
    if not os.path.exists('out'):
        os.makedirs('out')
    
    # Get sentiment analysis results
    df = get_ticker_news_sentiment(ticker)
    
    # Save DataFrame to CSV
    df.to_csv(f'out/{ticker}.csv', index=False)
    
    # Save DataFrame to JSON
    df.to_json(f'out/{ticker}.json', orient='records')

if __name__ == '__main__':
    generate_csv_and_json('LMT')
    generate_csv_and_json('AAPL')
    
