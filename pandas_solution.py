import requests
import pandas as pd
from bs4 import BeautifulSoup

def get_sp500_companies():
    # Wikipedia URL for S&P 500 companies
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

    # Get the webpage content
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the correct table by ID
    table = soup.find('table', {'id': 'constituents'})

    if not table:
        print("Table not found")
        return None

    # Read table into pandas DataFrame
    df = pd.read_html(str(table))[0]

    # Select only relevant columns: 'Symbol', 'Security', 'GICS Sector'
    df = df[['Symbol', 'Security', 'GICS Sector']]

    return df

if __name__ == "__main__":
    sp500_df = get_sp500_companies()

    # Save to CSV
    if sp500_df is not None:
        sp500_df.to_csv('sp500_companies.csv', index=False)
        print("Data saved to sp500_companies.csv")
    
    # Display the first few rows
    print(sp500_df.head())
