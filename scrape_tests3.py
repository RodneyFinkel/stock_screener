import requests
from bs4 import BeautifulSoup
from pprint import pprint
import json
import csv
import os

url = "https://finance.yahoo.com/quote/TSLA/key-statistics?p=TSLA"

def get_headers():
    return {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36"}

def scrape_data(url):
    page = requests.get(url, headers=get_headers())
    soup = BeautifulSoup(page.content, 'html.parser')
    
    data = {'financial_highlights': {}, 'trading_information': {}}
    
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
        data['financial_highlights'][title] = financial_data

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
        data['trading_information'][title] = trading_data
    
    return data

def save_to_json(data):
    if not os.path.exists('out'):
        os.makedirs('out')
    with open('out/scraped_data.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

def save_to_csv(data):
    if not os.path.exists('out'):
        os.makedirs('out')
    with open('out/scraped_data.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Category', 'Attribute', 'Value'])
        for category, info in data.items():
            for title, details in info.items():
                for label, value in details.items():
                    writer.writerow([category, title, label, value])

if __name__ == '__main__':
    scraped_data = scrape_data(url)
    pprint(scraped_data)
    save_to_json(scraped_data)
    save_to_csv(scraped_data)
