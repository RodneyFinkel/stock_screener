from stock_screener import Stock, StockScreener
import requests
from bs4 import BeautifulSoup
from pprint import pprint

# Get sp500 ticker and sector
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

table = soup.find('table', {'class': 'wikitable sortable'})
rows = table.find_all('tr')[1:]  # skip the header row

sp500 = []


for row in rows:
    cells = row.find_all('td')
    ticker = cells[0].text.strip()
    company = cells[1].text.strip()
    sector = cells[3].text.strip()
    sp500.append({'ticker': ticker, 'company': company, 'sector': sector})
    pprint(sp500)
    
    
def get_sp500_stocks():
    sp500_stocks = [Stock(stock['ticker'], stock['sector']) for stock in sp500]
    # print(sp500_stocks)
    return sp500_stocks

# Run example with 2 stocks
# filters = [lambda stock: StockScreener.filter_sector(stock, 'Interactive Media & Services'),
#            lambda stock: StockScreener.filter_price(stock, 60, 200),
#            lambda stock: StockScreener.filter_metric(stock, 'shares_outstanding', '>', 3*1e9)]

# sp500_stocks = [Stock('NKLA', 'Interactive Media & Services'), Stock('GOOGL', 'Interactive Media & Services')]
# screener = StockScreener(sp500_stocks, filters)
# screener.add_data()
# filtered_stocks = screener.apply_filters()

# Run screener for all sp500 tickers
filters = [lambda stock: StockScreener.filter_sector(stock, 'Interactive Media & Services'),
           lambda stock: StockScreener.filter_price(stock, 50, 300),
           lambda stock: StockScreener.filter_metric(stock, 'profit_margin', '>', 5),
           lambda stock: StockScreener.filter_technical_indicators(stock, 'UpperBand', '>', 'price'),
           lambda stock: StockScreener.filter_technical_indicators(stock, 'LowerBand', '<', 'price')          
    ]

sp500_stocks = [Stock('GOOG', 'Interactive Media & Services'), Stock('GOOGL', 'Interactive Media & Services'), Stock('TSLA', 'Automobile Manufacturers'), Stock('META', 'Interactive Media & Services' )]
#sp500_stocks = get_sp500_stocks()
screener = StockScreener(sp500_stocks, filters)
# Add Data
screener.add_data()
# Apply Filters
filtered_stocks = screener.apply_filters()
# Train Model
#screener.train_models()
# Make Predictions
#predicted_stocks = screener.predict_stocks(filtered_stocks)
