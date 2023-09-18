from bs4 import BeautifulSoup

# Stock Class
class Stock:
    def __init__(self, ticker, sector):
        self.ticker = ticker
        self.sector = sector
        self.price = 0.0
        self.url = f"https://finance.yahoo.com/quote/{self.ticker}/key-statistics?p{self.ticker}"
        self.data = {}
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
        
        
    # Scrape statistics
    def scrape_data(self):
        page = requests.get(self.url, headers=get_headers())
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

    def get_headers():
        return {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36"}
            
       
       
    # Scrape price
    def get_stock_price(self):
        try:
            url = f'https://finance.yahoo.com/quote/{self.ticker}'
            response = requests.get(url, headers=get_headers())
            soup = BeautifulSoup(response.content, 'html.parser')
            data = soup.find('fin-streamer', {'data-symbol': self.ticker})
            price = float(data['value'])
            self.price = price
        
        except:
            print(f'Price not available for {self.ticker}')
            self.price = 0.0     
        
        
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

    # Select stocks that pass all filters
    def apply_filters(self):
        filtered_stocks = []
        for stock in self.stocks:
            passed_all_filters = True
            for filter_func in self.filters:
                if not filter_func(stock):
                    passed_all_filters = False
                    break
            if passed_all_filters:
                filtered_stocks.append(stock)
        return filtered_stocks
    
    def filter_sector(stock, sector):
        return stock.sector == sector
    
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
        # if operator == '>':
        #     return float(stock.data[metric]) > value
        # elif operator == '>=':
        #     return float(stock.data[metric]) >= value
        # elif operator == '<':
        #     return float(stock.data[metric]) < value
        # elif operator == '<=':
        #     return float(stock.data[metric]) <= value
        # elif operator == '==':
        #     return float(stock.data[metric]) == value
        # else:
        
        