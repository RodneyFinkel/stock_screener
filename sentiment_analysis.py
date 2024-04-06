from transformers import pipeline
import yfinance as yf
from goose3 import Goose 
from requests import get
import pandas as pd
import csv
import os

def get_ticker_news_sentiment(ticker):
    """
    Returns a pandas dataframe for the given ticker's most recent news 
    article headlines with the over sentiment of each article

    Args:
        ticker (string): _description_
        
    Returns: 
    pd.DataFrame: {'Date', 'Article title', 'Article sentiment'}
    """
    
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
        if len(text) > 512:
            data.append({
                'Date':f'{date}',
                'Article sentiment':'Nan too long'
            })
            
        else:
            result = pipe(text)
            # print(results)
            data.append({
                'Date':f'{date}',
                'Article title':f'{title}',
                'Article sentiment':result[0]['label']
            })
    print(data)       
    df = pd.DataFrame(data)
    print(df)
    return df

# def generate_csv(ticker):
#     pass

if __name__ == '__main__':
    get_ticker_news_sentiment('APPL')