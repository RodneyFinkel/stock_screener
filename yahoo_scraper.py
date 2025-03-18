import requests
from bs4 import BeautifulSoup
import json

# Define the stock symbol and URL
stock_symbol = "TSLA"
url = f"https://finance.yahoo.com/quote/{stock_symbol}/key-statistics?p={stock_symbol}"

# Send a request to fetch the page content
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
}
response = requests.get(url, headers=headers)

# Parse the page content with BeautifulSoup
soup = BeautifulSoup(response.text, "html.parser")

# data_testids = ['qsp-statistics', 'card-container']
# Locate the 'Valuation Measures' section
valuation_section = soup.find("section", {"data-testid": "qsp-statistics"})

if valuation_section:
    table = valuation_section.find("table")

    if table:
        # Extract table headers (date columns)
        headers = [th.text.strip() for th in table.find_all("th")]

        # Extract the rows and store the values in a dictionary
        valuation_data = {}

        for row in table.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) > 1:
                metric = cols[0].text.strip()
                values = [col.text.strip() for col in cols[1:]]
                valuation_data[metric] = dict(zip(headers[1:], values))  # Skip the first empty header

        # Print extracted valuation data
        for key, value in valuation_data.items():
            print(f"{key}: {value}")
            
        filename = f"{stock_symbol}_valuation_data.json"
        with open(filename, 'w') as file:
            json.dump(valuation_data, file, indent=4)
        print(f'Data saved to {filename}')
        
    else:
        print("Table not found in the valuation section.")
else:
    print("Valuation Measures section not found.")
    

