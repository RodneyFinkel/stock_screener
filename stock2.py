import requests
import pandas as pd
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup


# Stock Class
class Stock:
    def __init__(self, ticker, sector, price=None, data=None):
        self.ticker = ticker
        self.sector = sector
        self.url = f"https://finance.yahoo.com/quote/{self.ticker}/key-statistics?p{self.ticker}"
        # add data
        self.price = price
        self.data = data #self.data2
        # Deep Learning Attributes
        train_data_aux, prices = add_technical_indicators(self.data)
        self.technical_indicators = train_aux_data.iloc[:-10, :].drop('Close', axis=1) # excluding the last 10 days of data and dropping close prices
        # set label as profit loss of 10 day future price from actual price
        labels_aux = (train_data_aux['Close'].shift(-10)) > train_data_aux['Close'].astype(int)
        self.labels = labels_aux[:-10]
        
        # Today features for prediction
        self.today_technical_indicators = prices[['MA20', 'MA50', 'RSI', 'MACD', 'UpperBand', 'LowerBand',]].iloc[-1, :]
        self.labels = pd.DataFrame()
        self.prediction = 0.0       
        # Metrics
        #self.metrics = {} # self.data = {} 
        # Metric aliases pairs
        self.metric_aliases = {
            'Market Cap (intraday)': 'market_cap',
            'Beta (5Y Monthly)': 'beta',
            '52 Week High 3': '52_week_high',
            '52 Week Low 3': '52_week_low',
            '50-Day Moving Average 3': '50_day_ma',
            '200-Day Moving Average 3': '200_day_ma',
            'Avg Vol (3 month) 3': 'avg_vol_3m',
            'Avg Vol (10 day) 3': 'avg_vol_10d',
            'Shares Outstanding 5': 'shares_outstanding',
            'Float 8': 'float',
            '% Held by Insiders 1': 'held_by_insiders',
            '% Held by Institutions 1': 'held_by_institutions',
            'Short Ratio (Jan 30, 2023) 4': 'short_ratio',
            'Payout Ratio 4': 'payout_ratio',
            'Profit Margin': 'profit_margin',
            'Operating Margin (ttm)': 'operating_margin',
            'Return on Assets (ttm)': 'return_on_assets',
            'Return on Equity (ttm)': 'return_on_equity',
            'Revenue (ttm)': 'revenue',
            'Revenue Per Share (ttm)': 'revenue_per_share',
            'Gross Profit (ttm)': 'gross_profit',
            'EBITDA ': 'ebitda',
            'Net Income Avi to Common (ttm)': 'net_income',
            'Diluted EPS (ttm)': 'eps',
            'Total Cash (mrq)': 'total_cash',
            'Total Cash Per Share (mrq)': 'cash_per_share',
            'Total Debt (mrq)': 'total_debt',
            'Total Debt/Equity (mrq)': 'debt_to_equity',
            'Current Ratio (mrq)': 'current_ratio',
            'Book Value Per Share (mrq)': 'book_value_per_share',
            'Operating Cash Flow (ttm)': 'operating_cash_flow',
            'Levered Free Cash Flow (ttm)': 'levered_free_cash_flow'
        }
        self.metrics = scrape_data(self.url, self.metric_aliases)
        
    def __repr__(self):
        return f"Stock: ticker={self.ticker}, sector={self.sector}, price={self.price})"     
        
    # Scrape statistics
    def scrape_data(self):
        page = requests.get(self.url, headers=self.get_headers())
        soup = BeautifulSoup(page.content, 'html.parser')
        
        data = {}
        
        sections = soup.find_all('section', {'data-test': 'qsp-statistics'})
        for section in sections:
            rows = section.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) == 2:
                    metric = cols[0].text.strip()
                    if metric in self.metric_aliases:
                        data[self.metric_aliases[metric]] = cols[1].text.strip()
        
        self.data = data
        pprint(data)

    def get_headers(self):
        return {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36"}
                
       
    # Scrape price
    def get_stock_price(self):
        try:
            url = f'https://finance.yahoo.com/quote/{self.ticker}'
            response = requests.get(url, headers=self.get_headers())
            # print(response)
            soup = BeautifulSoup(response.content, 'html.parser')
            data = soup.find('fin-streamer', {'data-symbol': self.ticker})
            # pprint(data)
            price = float(data['value'])
            print('Stock_price', price)
            self.price = price
        
        except:
            print(f'Price not available for {self.ticker}')
            self.price = 0.0  
            
    
    def get_historical(self):
        stock = yf.Ticker(self.ticker)
        history = stock.history(start='2010-01-01', end='2023-09-24') 
        self.data = history
        
        
    
    def add_technical_indicators(self):
        # get historical stock prices
        prices = self.data 
        if len(prices) < 20:
            return
        
        # calculate 20-day moving average
        prices['MA20'] = prices['Close'].rolling(window=20).mean()
        
        # calculate 50-day moving average
        prices['MA50'] = prices['Close'].rolling(window=50).mean()
        
        # calculate relative strength index (RSI)
        delta = prices['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain/avg_loss
        prices['RSI'] = 100 - (100/(1 + rs))
        
        # calculating moving average convergence divergence (MACD)
        exp1 = prices['Close'].ewm(span=12, adjust=False).mean()
        exp2 = prices['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        prices['MACD'] = macd - signal
        
        # calculate Bollinger Bands
        prices['MA20'] = prices['Close'].rolling(window=20).mean()
        prices['20STD'] = prices['Close'].rolling(window=20).std()
        prices['UpperBand'] = prices['MA20'] + (prices['20STD'] * 2)
        prices['LowerBand'] = prices['MA20'] - (prices['20STD'] * 2)
        
        # Features for deep learning model
        train_data_aux = prices[['Close', 'MA20', 'MA50', 'RSI', 'MACD', 'UpperBand', 'LowerBand']].dropna()
        self.technical_indicators = train_data_aux.iloc[:-10, :].drop('Close', axis=1)
        
        # Set label as profit loss of 10 day future price from actual price
        labels_aux = train_data_aux['Close'].shift(-10) > train_data_aux['Close'].astype(int)
        self.labels = labels_aux[:-10]
        
        # Today features for predicition
        self.today_technical_indicators = prices[['MA20', 'MA50', 'RSI', 'MACD', 'UpperBand', 'LowerBand']].iloc[-1,:]
        
        prices = prices.reset_index()
        
        # store technical indicators in stock data dictionary
        self.data.update(prices[['Date', 'MA20', 'MA50', 'RSI', 'MACD', 'UpperBand', 'LowerBand']].to_dict('list'))
        data2 = self.data
        pprint(data2)
