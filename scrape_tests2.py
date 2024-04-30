import requests
from bs4 import BeautifulSoup
from pprint import pprint
import yfinance as yf

url = f"https://finance.yahoo.com/quote/TSLA/key-statistics?pTSLA"
metric_aliases = {
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

def get_headers():
    return {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36"}


def scrape_data(url, metric_aliases):
    print('initialising scrape_data')
    page = requests.get(url, headers=get_headers())
    soup = BeautifulSoup(page.content, 'html.parser')
    pprint(soup)
    data = {}
    
    sections = soup.find_all('section', {'data-testid':'qsp-statistics'})
    for section in sections:
        rows = section.find_all('tr')
        for row in rows:
            cols = row.find_all('td')
            if len(cols) == 2:
                metric = cols[0].text.strip()
                if metric in metric_aliases:
                    data[metric_aliases[metric]] = cols[1].text.strip()
    print('scrape_data function exit')
    pprint(data)
    return data
                
if __name__ == '__main__':
    scrape_data(url, metric_aliases)



