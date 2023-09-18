from stock_screener.py import Stock, StockScreener

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
    
    
def get_sp500_stocks():
    sp500_stocks = [Stock(stock['ticker'], stock['sector']) for stock in sp500]
    return sp500_stocks

# Run example with 2 stocks
filters = [lambda stock: filter_sector(stock, 'Interactive Media & Services'),
           lambda stock: filter_price(stock, 60, 200),
           lambda stock: filter_metric(stock, 'shares_outstanding', '>', 3*1e9)]

sp500_stocks = [Stock('GOOGL', 'Interactive Media & Services'), Stock('GOOG', 'Interactive Media & Services')]
screener = StockScreener(sp500_stocks, filters)
screener.add_data()
filtered_stocks = screener.apply_filters()

# Run screener for all sp500 tickers
filters = [lambda stock: filter_sector(stock, 'Asset Management & Custody Banks'),
           lambda stock: filter_price(stock, 50, 200),
           lambda stock: filter_metric(stock, 'profit_margin', '>', 10)]

sp500_stocks = get_sp500_stocks()
screener = StockScreener(sp500_stocks, filters)
screener.add_data()
filtered_stocks = screener.apply_filters()