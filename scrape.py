from bs4 import BeautifulSoup
import requests
from pprint import pprint

def get_sp_tickers():
    # Get sp500 ticker and sector
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'class': 'wikitable sortable'})
    rows = table.find_all('tr')[1:] # skip the header row
    
    sp500 = []
    
    for row in rows:
        cells = row.find_all('td')
        ticker = cells[0].text.strip()
        company = cells[1].text.strip()
        sector = cells[3].text.strip()
        sp500.append({'ticker': ticker, 'company': company, 'sector': sector})
    pprint(sp500)    
    return sp500

def get_sp500_stocks(sp500):
    sp500_stocks = []
    for stock in sp500:
        try:
            price = get_stock_price2(stock['ticker'])
            sp500_stocks.append((stock['ticker'], stock['sector'], price))
        except:
            stock_issues.write(f'There was an issue with {stock["ticker"]}.')
            
    print(sp500_stocks)
    return sp500_stocks                
                

def get_headers():
        return {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36"}


def get_stock_price2(ticker):
        try:
                url = f'https://finance.yahoo.com/quote/{ticker}'
                response = requests.get(url, headers=get_headers())
                soup = BeautifulSoup(response.content, 'html.parser')
                data = soup.find('fin-streamer', {'data-symbol': ticker})
                pprint(data)
                price = float(data['value'])
                print(price)
        except:
                print(f'Price not available for {ticker}')
                price = 0.0  
                
 
        
if __name__ == "__main__":
    get_sp_tickers()
    get_sp500_stocks(sp500)
    
    
    
    
    
 
    