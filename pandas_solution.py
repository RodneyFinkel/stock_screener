import requests
import pandas as pd
from bs4 import BeautifulSoup

def get_sp500_companies():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})

    if not table:
        print("Table not found")
        return None

    df = pd.read_html(str(table))[0]
    df = df[['Symbol', 'Security', 'GICS Sector']]

    return df

if __name__ == "__main__":
    sp500_df = get_sp500_companies()
    if sp500_df is not None:
        sp500_df.to_csv('sp500_companies.csv', index=False)
        print("Data saved to sp500_companies.csv")
    
