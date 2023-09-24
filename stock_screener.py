from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np 
import yfinance as yf 
from sklearn.preprocessing import StandardScaler 
from pprint import pprint
import tensorflow as tf

# Stock Class
class Stock:
    def __init__(self, ticker, sector):
        self.ticker = ticker
        self.sector = sector
        self.price = 0.0
        self.url = f"https://finance.yahoo.com/quote/{self.ticker}/key-statistics?p{self.ticker}"
        self.data = pd.DataFrame()#self.data2
        # Deep Learning Attributes
        self.technical_indicators = pd.DataFrame()
        self.today_technical_indicators = pd.DataFrame()
        self.labels = pd.DataFrame()
        self.prediction = 0.0       
        # Metrics
        self.metrics = {}   # self.data = {}
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
            print(price)
            self.price = price
        
        except:
            print(f'Price not available for {self.ticker}')
            self.price = 0.0  
            
    
    def get_historical(self):
        stock = yf.Ticker(self.ticker)
        history = stock.history(start='2010-01-01', end='2023-06-29') 
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
        self.label = labels_aux[:-10]
        
        # Today features for predicition
        self.today_technical_indicators = prices[['MA20', 'MA50', 'RSI', 'MACD', 'UpperBand', 'LowerBand']].iloc[-1,:]
        
        prices = prices.reset_index()
        
        # store technical indicators in stock data dictionary
        self.data.update(prices[['Date', 'MA20', 'MA50', 'RSI', 'MACD', 'UpperBand', 'LowerBand']].to_dict('list'))
        data2 = self.data
        pprint(data2)




        
# Stocks Screener Class
class StockScreener:
    def __init__(self, stocks, filters):
        self.stocks = stocks
        self.filters = filters

    # Add data to stocks 
    def add_data(self):
        for stock in self.stocks:
            stock.scrape_data()
            stock.get_stock_price()
            stock.get_historical()
            stock.add_technical_indicators() 

    # Select stocks that pass all filters
    def apply_filters(self):
        filtered_stocks = []
        for stock in self.stocks:
            passed_all_filters = True
            for filter_func in self.filters:
                if not filter_func(stock):
                    passed_all_filters = False
                    print(passed_all_filters)
                    break
            if passed_all_filters:
                filtered_stocks.append(stock)  
        print('Filtered_stocks:', filtered_stocks)
        return filtered_stocks
    
    def filter_sector(stock, sector):
        #return stock.sector == sector
        pass 
    
    def filter_price(stock, min_price, max_price):
        return min_price <= stock.price <= max_price
    
    def filter_metric(stock, metric, operator, value):  
        if metric not in stock.data:
            return False

        # Convert value to same units as metric, if necessary
        if 'B' in stock.data[metric]:
            stock.data[metric] = stock.data[metric].replace('B', '')
            value = float(value) / 1e9
        elif 'M' in stock.data[metric]:
            stock.data[metric] = stock.data[metric].replace('M', '')
            value = float(value) / 1e6
        elif '%' in stock.data[metric]:
            stock.data[metric] = stock.data[metric].replace('%', '')
            value = float(value)
        else:
            value = float(value)

        # Check condition according to operator
        if operator == '>':
            return float(stock.data[metric]) > value
        elif operator == '>=':
            return float(stock.data[metric]) >= value
        elif operator == '<':
            return float(stock.data[metric]) < value
        elif operator == '<=':
            return float(stock.data[metric]) <= value
        elif operator == '==':
            return float(stock.data[metric]) == value
        # else:
        
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
        elif opertor == '>=':
            return float(indicator_value) >= value
        elif operator == '<':
            return float(indicator_value) < value
        elif operator == '<=':
            return float(indicator_value) <= value
        elif operator == '==':
            return float(indicator_value) == value
        else:
            return False
            
        
   # Train deep learning models on selected stocks
    def train_models(self):
        # Get data for training and testing
        filtered_stocks = self.apply_filters()
       
        for stock in filtered_stocks:
            train_data = stock.technical_indicators
            train_labels = stock.label
           
            # Normalize the Data
            train_data = self.scaler.fit_transform(train_data)
            train_labels = np.array(train_labels)
            
            #Create and train model
            model = create_model(train_data) # def create_model(train_data): (THIS function is be defined lower)
            model.fit(train_data, train_labels, epochs=10)
            self.models[stock.ticker] = model # models is defined later
            
    # Predict whether new stocks will pass filters (where does new_stocks come from?)
    def predict_stocks(self, new_stocks):
        # Add technical indicators to new stocks
        for stock in new_stocks:
            stock.get_historical()
            stock.add_technical_indicators()
            
        
        # Make predictions for each stock using its corresponding model
        predicted_stocks = []
        for stock in new_stocks:
            if stock.ticker in self.models:
                model = self.models[stock.ticker]
                # Reshape as there is only one sample
                new_feature_aux = np.array(stock.today_technical_indicators).reshape(1,-1)
                new_stock_data = self.scaler.fit_transform(new_features_aux)
                prediction = model.predict(new_stock_data)
                stock.prediction = prediction
                if predicition > 0.5:
                    predicted_stocks.append(stock)
        print('Predicted_Stocks:',predicted_stocks)           
        return predicted_stocks
    
    
    # Simple Dense Model
    def create_model(train_data):
        # Creating a sequential model (neural network suitable for binary classification tasks)
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, input_shape=(train_data.shape[1],), activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
            
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    
    
    
                
                
                          
            
            
            
             