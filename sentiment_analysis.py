from finvizfinance.screener.overview import Overview
from transformers import pipeline
import yfinance as yf
from goose3 import Goose 
from requests import get
import pandas as pd
import csv
import os
import json


def get_undervalued_stocks():
    foverview = Overview()
    filters_dict = {'Debt/Equity':'Under 1', 
                    'PEG':'Low (<1)', 
                    'Operating Margin':'Positive (>0%)', 
                    'P/B':'Low (<1)',
                    'P/E':'Low (<15)',
                    'InsiderTransactions':'Positive (>0%)'}
    parameters = ['Exchange', 'Index', 'Sector', 'Industry', 'Country', 'Market Cap.',
        'P/E', 'Forward P/E', 'PEG', 'P/S', 'P/B', 'Price/Cash', 'Price/Free Cash Flow',
        'EPS growththis year', 'EPS growthnext year', 'EPS growthpast 5 years', 'EPS growthnext 5 years',
        'Sales growthpast 5 years', 'EPS growthqtr over qtr', 'Sales growthqtr over qtr',
        'Dividend Yield', 'Return on Assets', 'Return on Equity', 'Return on Investment',
        'Current Ratio', 'Quick Ratio', 'LT Debt/Equity', 'Debt/Equity', 'Gross Margin',
        'Operating Margin', 'Net Profit Margin', 'Payout Ratio', 'InsiderOwnership', 'InsiderTransactions',
        'InstitutionalOwnership', 'InstitutionalTransactions', 'Float Short', 'Analyst Recom.',
        'Option/Short', 'Earnings Date', 'Performance', 'Performance 2', 'Volatility', 'RSI (14)',
        'Gap', '20-Day Simple Moving Average', '50-Day Simple Moving Average',
        '200-Day Simple Moving Average', 'Change', 'Change from Open', '20-Day High/Low',
        '50-Day High/Low', '52-Week High/Low', 'Pattern', 'Candlestick', 'Beta',
        'Average True Range', 'Average Volume', 'Relative Volume', 'Current Volume',
        'Price', 'Target Price', 'IPO Date', 'Shares Outstanding', 'Float']
    
    foverview.set_filter(filters_dict=filters_dict)
    df_overview = foverview.screener_view()
    if not os.path.exists('overview_out'):
        os.makedirs('overview_out')
    df_overview.to_csv('overview_out/Overview.csv', index=False)
    tickers = df_overview['Ticker'].to_list()
    return tickers



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
    ticker_list = get_undervalued_stocks()
    for i in ticker_list:
        generate_csv_and_json(i)
    
    
