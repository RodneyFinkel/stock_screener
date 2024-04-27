from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from transformers import pipeline
import yfinance as yf
from goose3 import Goose 
from requests import get
import pandas as pd
from bs4 import BeautifulSoup
import os


extractor = Goose()
pipe = pipeline("text-classification", model="ProsusAI/finbert")
    
def analyze_news_sentiment(news):
    title = news['title']
    response = get(news['link'])
    article = extractor.extract(raw_html=response.content)
    text = article.cleaned_text
    date = article.publish_date
    publisher = news['publisher']
    link = news['link']
    
    if len(text) > 512:
        return {
            'Date': f'{date}',
            'Article sentiment': 'Nan too long'
        }
    else:
        result = pipe(text)
        return {
            'Date': f'{date}',
            'Article title': f'{title}',
            'link': f'{link}',
            'publisher': f'{publisher}',
            'Article sentiment': result[0]['label']
        }

def get_ticker_news_sentiment(ticker):
    ticker_news = yf.Ticker(ticker)
    news_list = ticker_news.get_news()
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        sentiment_results = list(executor.map(analyze_news_sentiment, news_list))
    
    df = pd.DataFrame(sentiment_results)
    return df

def generate_csv_and_json(ticker):
    if not os.path.exists('out'):
        os.makedirs('out')
    
    df = get_ticker_news_sentiment(ticker)
    df.to_csv(f'out/{ticker}.csv', index=False)
    df.to_json(f'out/{ticker}.json', orient='records')
    print(df)
# def get_sp_tickers():
#     # Get sp500 ticker and sector
#     url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
#     response = requests.get(url)
#     soup = BeautifulSoup(response.content, 'html.parser')
    
#     table = soup.find('table', {'class': 'wikitable sortable'})
#     rows = table.find_all('tr')[1:] # skip the header row
    
#     sp500 = []
    
#     for row in rows:
#         cells = row.find_all('td')
#         ticker = cells[0].text.strip()
#         company = cells[1].text.strip()
#         sector = cells[3].text.strip()
#         sp500.append({'ticker': ticker, 'company': company, 'sector': sector})
        
#     return sp500


if __name__ == '__main__':
    ticker_list = ['LMT', 'AAPL', 'GE', 'META', 'TSLA', 'MSFT', 'GOOGL']
    for ticker in ticker_list:
        generate_csv_and_json(ticker)
