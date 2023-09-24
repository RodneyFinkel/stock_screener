from bs4 import BeautifulSoup
import requests
from pprint import pprint



def get_headers():
        return {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36"}


def get_stock_price2(ticker):
        # try:
            url = f'https://finance.yahoo.com/quote/{ticker}'
            response = requests.get(url, headers=get_headers())
            soup = BeautifulSoup(response.content, 'html.parser')
            data = soup.find('fin-streamer', {'data-symbol': ticker})
            pprint(data)
            price = float(data['value'])
            print(price)
        
        # except:
        #     print(f'Price not available for {ticker}')
        #     price = 0.0     
        
if __name__ == "__main__":
    ticker = 'BABA'
    get_stock_price2(ticker)
    
    
    
    
    