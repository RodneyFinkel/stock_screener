import requests
from bs4 import BeautifulSoup
import json
from pprint import pprint


def scrape_financials(ticker):
    stock_symbol = ticker
    url = f"https://finance.yahoo.com/quote/{stock_symbol}/key-statistics?p={stock_symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    data = {}
    sections = soup.find_all('section', class_='card small tw-p-0 yf-ispmdb sticky noBackGround')
    
    for section in sections:
        title_tag = section.find('h3', class_='title font-condensed yf-ispmdb clip')
        if not title_tag:
            continue
        title = title_tag.text.strip()
        
        data[title] = {}
        rows = section.find_all('tr', class_='row yf-vaowmx')
        
        for row in rows:
            label_tag = row.find('td', class_='label yf-vaowmx')
            value_tag = row.find('td', class_='value yf-vaowmx')
            
            if label_tag and value_tag:
                label = label_tag.text.strip()
                value = value_tag.text.strip()
                data[title][label] = value
    
    return data

def save_data_to_file(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    print(f'Data saved to {filename}')

if __name__ == '__main__':
    financial_data = scrape_financials("TSLA")
    save_data_to_file(financial_data, 'scrape_financials.json')
    pprint(financial_data)