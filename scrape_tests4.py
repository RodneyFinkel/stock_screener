import requests
from bs4 import BeautifulSoup
from pprint import pprint
import json
import os

url = "https://finance.yahoo.com/quote/AMZN/key-statistics?p=AMZN"

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

def scrape_data(url):
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
    
    return data


data = {}
def traverse_data(data_pre, metric_aliases):
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

def save_to_json(data):
    if not os.path.exists('out'):
        os.makedirs('out')
    with open('out/scraped_data.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)



if __name__ == '__main__':
    data = scrape_data(url)
    pprint(data)
    save_to_json(data)

   
