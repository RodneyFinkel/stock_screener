import requests
import pandas as pd
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup


# Stock Class
class Stock:
    def __init__(self, ticker, sector, price=None, data=None):
        print('initializing stock object')
        self.ticker = ticker
        self.sector = sector
        self.url = f"https://finance.yahoo.com/quote/{self.ticker}/key-statistics?p{self.ticker}"
        # Add Data
        self.price = price
        self.data = data 
        # Deep Learning Attributes
        train_data_aux, prices = add_technical_indicators(self.data)
        self.technical_indicators = train_data_aux.iloc[:-10, :].drop('Close', axis=1) # excluding the last 10 days of data and dropping close prices
        print('technical indicators attribute instantiated')
        print(f"technical indicators shape1: {self.technical_indicators.shape}")
        # labels for training predictive models
        # collect labels as true or false for each stock by labelling those whose price, 10 days ahead of a current date, is higher as True. 
        # this is accomplished by shifting prices by 10 for each current date. The last 10 rows of labels_aux are ommited
        # These labels indicate whether there would be a profit (True) or a loss (False) 
        # in the next 10 days for each day in the dataset, excluding the last 10 days.
        labels_aux = (train_data_aux['Close'].shift(-10)) > train_data_aux['Close'].astype(int)
        self.label = labels_aux[:-10]
        print(f"labels shape1: {self.label.shape}")
        # Today's features for prediction
        self.today_technical_indicators = prices[['MA20', 'MA50', 'RSI', 'MACD', 'UpperBand', 'LowerBand',]].iloc[-1, :]
        self.labels = pd.DataFrame()
        self.prediction = 0.0 
        print('labels set')      
        # Metrics
        # Metric aliases pairs
        # self.metric_aliases = {
        #     'Market Cap (intraday)': 'market_cap',
        #     'Beta (5Y Monthly)': 'beta',
        #     '52 Week High 3': '52_week_high',
        #     '52 Week Low 3': '52_week_low',
        #     '50-Day Moving Average 3': '50_day_ma',
        #     '200-Day Moving Average 3': '200_day_ma',
        #     'Avg Vol (3 month) 3': 'avg_vol_3m',
        #     'Avg Vol (10 day) 3': 'avg_vol_10d',
        #     'Shares Outstanding 5': 'shares_outstanding',
        #     'Float 8': 'float',
        #     '% Held by Insiders 1': 'held_by_insiders',
        #     '% Held by Institutions 1': 'held_by_institutions',
        #     'Short Ratio (Jan 30, 2023) 4': 'short_ratio',
        #     'Payout Ratio 4': 'payout_ratio',
        #     'Profit Margin': 'profit_margin',
        #     'Operating Margin (ttm)': 'operating_margin',
        #     'Return on Assets (ttm)': 'return_on_assets',
        #     'Return on Equity (ttm)': 'return_on_equity',
        #     'Revenue (ttm)': 'revenue',
        #     'Revenue Per Share (ttm)': 'revenue_per_share',
        #     'Gross Profit (ttm)': 'gross_profit',
        #     'EBITDA ': 'ebitda',
        #     'Net Income Avi to Common (ttm)': 'net_income',
        #     'Diluted EPS (ttm)': 'eps',
        #     'Total Cash (mrq)': 'total_cash',
        #     'Total Cash Per Share (mrq)': 'cash_per_share',
        #     'Total Debt (mrq)': 'total_debt',
        #     'Total Debt/Equity (mrq)': 'debt_to_equity',
        #     'Current Ratio (mrq)': 'current_ratio',
        #     'Book Value Per Share (mrq)': 'book_value_per_share',
        #     'Operating Cash Flow (ttm)': 'operating_cash_flow',
        #     'Levered Free Cash Flow (ttm)': 'levered_free_cash_flow'
        # }
        self.metric_aliases = {
            'Market Cap': 'market_cap',
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
            'Operating Margin  (ttm)': 'operating_margin',
            'Return on Assets  (ttm)': 'return_on_assets',
            'Return on Equity  (ttm)': 'return_on_equity',
            'Revenue  (ttm)': 'revenue',
            'Revenue Per Share  (ttm)': 'revenue_per_share',
            'Gross Profit  (ttm)': 'gross_profit',
            'EBITDA': 'ebitda',
            'Net Income Avi to Common  (ttm)': 'net_income',
            'Diluted EPS  (ttm)': 'eps',
            'Total Cash  (mrq)': 'total_cash',
            'Total Cash Per Share  (mrq)': 'cash_per_share',
            'Total Debt  (mrq)': 'total_debt',
            'Total Debt/Equity  (mrq)': 'debt_to_equity',
            'Current Ratio  (mrq)': 'current_ratio',
            'Book Value Per Share  (mrq)': 'book_value_per_share',
            'Operating Cash Flow  (ttm)': 'operating_cash_flow',
            'Levered Free Cash Flow  (ttm)': 'levered_free_cash_flow'
        }
        print('instantiating metrics')
        self.metrics = scrape_data(self.url, self.metric_aliases)
        
        
### Utils Functions ###  
    
def get_headers():
    return {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36"}

def filter_sector(stock, sector):
        return stock.sector == sector
           
def filter_price(stock, min_price, max_price):
    return min_price <= stock.price <= max_price

def filter_metric(stock, metric, operator, value):  
    if metric not in stock.metrics:
        print(f'{metric} not in stock metrics')
        return False
    
    # Format metric value
    metric_value = stock.metrics[metric]
    metric_value = metric_value.replace(',', '.')
    
    # check if the value is 'price'
    if value == 'price':
        value = float(stock.price)
    else:
        value = float(value)

    # Convert value to same units as metric, if necessary
    if 'B' in metric_value:
        metric = metric_value.replace('B', '')
        value = float(value) / 1e9
    elif 'M' in metric_value:
        metric = metric_value.replace('M', '')
        value = float(value) / 1e6
    elif '%' in metric_value:
        metric = metric_value.replace('%', '')
        value = float(value)
    else:
        metric = metric_value
        value = float(value)
        
    # Return false if metric_value is still not a valid float
    try:
        metric = float(metric)
    except ValueError:
        return False

    # Check condition according to operator
    if operator == '>':
        return metric > value
    elif operator == '>=':
        return metric >= value
    elif operator == '<':
        return metric < value
    elif operator == '<=':
        return metric <= value
    elif operator == '==':
        return metric == value
    else:
        raise ValueError(f'Invalid operator: {operator}')
    
def filter_technical_indicators(stock, indicator_name, operator, value):
    if indicator_name not in stock.today_technical_indicators:
        return False
    
    # Obtain the value of the technical indicator
    indicator_value = stock.today_technical_indicators[indicator_name]
    
    # Check if the value is 'price'
    if value == 'price':
        value = float(stock.price)
    else:
        value = float(value)
        
    # Compare according to operator
    if operator == '>':
        return float(indicator_value) > value
    elif operator == '>=':
        return float(indicator_value) >= value
    elif operator == '<':
        return float(indicator_value) < value
    elif operator == '<=':
        return float(indicator_value) <= value
    elif operator == '==':
        return float(indicator_value) == value
    else:
        raise ValueError(f'Invalid operator: {operator}')
    
               
# Scrape statistics 
# def scrape_data(url, metric_aliases):
#     print('initialising scrape_data')
#     page = requests.get(url, headers=get_headers())
#     soup = BeautifulSoup(page.content, 'html.parser')
    
#     data = {}
#     # this needs fixing, yfinance changed source
#     sections = soup.find_all('section', {'data-test': 'qsp-statistics'})
#     for section in sections:
#         rows = section.find_all('tr')
#         for row in rows:
#             cols = row.find_all('td')
#             if len(cols) == 2:
#                 metric = cols[0].text.strip()
#                 if metric in metric_aliases:
#                     data[metric_aliases[metric]] = cols[1].text.strip()
#     print('scrape_data function exit')
#     return data

# Scrape Statisitics fix
def scrape_data(url, metric_aliases):
    print('initialising scrape_data')
    page = requests.get(url, headers=get_headers())
    soup = BeautifulSoup(page.content, 'html.parser')
    
    data_pre = {'financial_highlights': {}, 'trading_information': {}, 'valuation_measures': {}}
    
    # Scrape Financial Highlights Section
    financial_highlights_section = soup.find('section', class_='svelte-14j5zka')
    financial_cards = financial_highlights_section.find_all('section', class_='card small tw-p-0 svelte-1v51y3z sticky')
    
    for card in financial_cards:
        title = card.find('h3', class_='title font-condensed svelte-1v51y3z clip').text.strip()
        table_rows = card.find_all('tr')
        financial_data = {}
        for row in table_rows:
            label = row.find('td', class_='label svelte-vaowmx').text.strip()
            value = row.find('td', class_='value svelte-vaowmx').text.strip()
            financial_data[label] = value
        data_pre['financial_highlights'][title] = financial_data

    # Scrape Trading Information Section
    trading_info_section = soup.find_all('section', class_='svelte-14j5zka')[1]  # Get the second section with the same class
    trading_cards = trading_info_section.find_all('section', class_='card small tw-p-0 svelte-1v51y3z sticky')

    for card in trading_cards:
        title = card.find('h3', class_='title font-condensed svelte-1v51y3z clip').text.strip()
        table_rows = card.find_all('tr')
        trading_data = {}
        for row in table_rows:
            label = row.find('td', class_='label svelte-vaowmx').text.strip()
            value = row.find('td', class_='value svelte-vaowmx').text.strip()
            trading_data[label] = value
        data_pre['trading_information'][title] = trading_data
    
    # Scrape Valuation Measures Section
    valuation_section = soup.find('section', {'data-testid': 'qsp-statistics'})
    if valuation_section:
        valuation_rows = valuation_section.find_all('tr')
        for row in valuation_rows:
            cols = row.find_all('td')
            if len(cols) >= 2:
                metric = cols[0].text.strip()
                value = cols[1].text.strip()
                data_pre['valuation_measures'][metric] = value

    # Activate the traverse_data function here
    data = traverse_data(data_pre, metric_aliases)
    print('scrape_data function exit') 
    return data


data = {}
def traverse_data(data_pre, metric_aliases):
    print('traversing data structure')
    for key, value in data_pre.items():
        if isinstance(value, dict):
            traverse_data(value, metric_aliases)
        else:
            for alias_key, alias_value in metric_aliases.items():
                if key == alias_key or key == alias_value:
                    data[alias_value] = value
                elif value == alias_key or value == alias_value:
                    data[alias_value] = value               
    return data

                
### CACHED FUNCTIONS ###
    

@st.cache_data
def get_stock_price(ticker):
    try:
        url = f'https://finance.yahoo.com/quote/{ticker}'
        response = requests.get(url, headers=get_headers())
        soup = BeautifulSoup(response.content, 'html.parser')
        data = soup.find('fin-streamer', {'data-symbol': ticker})
        price = float(data['value'])
        return  price
    
    except:
        print(f'Price not available for {ticker}')
        price = 0.0  
        return price
    
@st.cache_data
def get_stock_price2(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    
    try:
        if 'currentPrice' in info:
            price = info['currentPrice'] # use currentPrice or regularMarketPrice
            price = float(price)
            return price
        else:
            print(f"Current price not available for {ticker}")
    
    except:
        print(f'Current price not available for {ticker}')
        price = 0.0  
        return price
    
        
        
@st.cache_data
def get_historical(ticker):
    stock = yf.Ticker(ticker)
    history = stock.history(start='2010-01-01', end='2024-04-11') 
    return history
            
@st.cache_data
def add_technical_indicators(data):
        # get historical stock prices
        prices = data 
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
        return train_data_aux, prices
       
        
        
        
        
