from bs4 import BeautifulSoup
import requests
from pprint import pprint
import yfinance as yf
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_headers():
    return {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36"}


def get_sp_tickers():
    # Get sp500 ticker and sector
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'class': 'wikitable sortable'})
    rows = table.find_all('tr')[1:] # skip the header row
    
    sp500 = [] 
    for row in rows:
        cells = row.find_all('td')
        ticker = cells[0].text.strip()
        company = cells[1].text.strip()
        sector = cells[3].text.strip()
        sp500.append({'ticker': ticker, 'company': company, 'sector': sector})  
         
    return sp500


def get_sp500_stocks(sp500):
    sp500_stocks = []
    for stock in sp500:
        #      try:
                print(stock['ticker'])
                price = get_stock_price2(stock['ticker'])
                print(price)
                data = get_historical(stock['ticker'])
                technical_data, prices  = add_technical_indicators(data)
                print(prices.head()) 
                print(prices.shape)
                prices_mod = prices[['MA20', 'MA50', 'RSI', 'MACD', 'UpperBand', 'LowerBand',]].iloc[-1, :]
                print(prices_mod)
                sp500_stocks.append((stock['ticker'], stock['sector'], price))
                plot_technical_indicators(prices, stock['ticker'])
                
                
        #      except:
                    #print((f"There was an issue with {stock['ticker']}."))
                   
    return sp500_stocks                
                
                
        
def get_stock_price2(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    try:
        if 'currentPrice' in info:
                price = info['currentPrice']  # use currentPrice or regularMarketPrice
                return price
        else:
                print(f"Current price not available for {ticker}")
                return None

    except:
            print(f'Current price not available for {ticker}')
            return price
    
 
def get_historical(ticker):
    stock = yf.Ticker(ticker)
    history = stock.history(start='2019-01-01', end='2023-09-24') 
    return history


def add_technical_indicators(data):
        # get historical stock prices
        prices = data 
        if len(prices) < 20:
            return
        
        # calculate 20-day moving average
        prices['MA20'] = prices['Close'].rolling(window=20).mean()
        
        # calculate 50-day moving average
        prices['MA50'] = prices['Close'].rolling(window=50).mean()
        
        # calculate relative strength index (RSI)
        delta = prices['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain/avg_loss
        prices['RSI'] = 100 - (100/(1 + rs))
        
        # calculating moving average convergence divergence (MACD)
        exp1 = prices['Close'].ewm(span=12, adjust=False).mean()
        exp2 = prices['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        prices['MACD'] = macd - signal
        prices['signal'] = signal
        
        # calculate Bollinger Bands
        prices['MA20'] = prices['Close'].rolling(window=20).mean()
        prices['20STD'] = prices['Close'].rolling(window=20).std()
        prices['UpperBand'] = prices['MA20'] + (prices['20STD'] * 2)
        prices['LowerBand'] = prices['MA20'] - (prices['20STD'] * 2)
        
        # Features for deep learning model
        train_data_aux = prices[['Close', 'MA20', 'MA50', 'RSI', 'MACD', 'UpperBand', 'LowerBand']].dropna()
        return train_data_aux, prices

def plot_technical_indicators(prices, ticker):
    
    fig, ax = plt.subplots(4, 1, figsize=(16, 10), dpi=100)

    # Plot 20-day and 50-day moving averages
    ax[0].plot(prices.index, prices['Close'], label='Close Price')
    ax[0].plot(prices.index, prices['Close'], label='Close Price')
    ax[0].plot(prices.index, prices['MA20'], label='20-day MA')
    ax[0].plot(prices.index, prices['MA50'], label='50-day MA')
    ax[0].fill_between(prices.index, prices['MA20'], prices['MA50'], alpha=0.35, color='gray', label='Moving Averages')
    ax[0].set_title(f'{ticker}: Moving Averages')
    ax[0].set_xlabel('Date')
    ax[0].set_ylabel('Price')
    ax[0].legend()


    # Set a dark theme
    # template = "plotly_dark"
    
    # # Create a subplot figure
    # fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=[
    #     f'{ticker}: Moving Averages',
    #     f'{ticker}: Bollinger Bands'
    # ])

    # # Plot 20-day and 50-day moving averages
    # fig.add_trace(go.Scatter(x=prices.index, y=prices['Close'], mode='lines', name='Close Price'), row=1, col=1)
    # fig.add_trace(go.Scatter(x=prices.index, y=prices['MA20'], mode='lines', name='20-day MA'), row=1, col=1)
    # fig.add_trace(go.Scatter(x=prices.index, y=prices['MA50'], mode='lines', name='50-day MA'), row=1, col=1)
    # fig.add_trace(go.Scatter(x=prices.index, y=prices['MA20'], fill='tonexty', fillcolor='rgba(128,128,128,0.3)', name='Moving Averages'), row=1, col=1)

    # # Plot Bollinger Bands
    # fig.add_trace(go.Scatter(x=prices.index, y=prices['Close'], mode='lines', name='Close Price'), row=2, col=1)
    # fig.add_trace(go.Scatter(x=prices.index, y=prices['UpperBand'], mode='lines', name='Upper Band'), row=2, col=1)
    # fig.add_trace(go.Scatter(x=prices.index, y=prices['LowerBand'], mode='lines', name='Lower Band'), row=2, col=1)
    # fig.add_trace(go.Scatter(x=prices.index, y=prices['UpperBand'], fill='tonexty', fillcolor='rgba(128,128,128,0.3)', name='Bollinger Bands'), row=2, col=1)

    # # Update layout for better appearance
    # fig.update_layout(
    #     title=f'{ticker} Technical Indicators',
    #     xaxis_rangeslider_visible=False,
    #     template=template,
    #     height=800,
    #     showlegend=False,
    #     paper_bgcolor='rgba(0,0,0,0)',
    #     plot_bgcolor='rgba(0,0,0,0)',
    # )

    # # Show the figure
    # fig.show()

    # Plot RSI
    ax[1].plot(prices.index, prices['RSI'], label='RSI')
    ax[1].axhline(y=70, color='r', linestyle='--', label='Overbought')
    ax[1].axhline(y=30, color='g', linestyle='--', label='Oversold')
    ax[1].set_title(f'{ticker}: RSI')
    ax[1].set_xlabel('Date')
    ax[1].set_ylabel('RSI')
    ax[1].legend()

    # Plot MACD
    ax[2].plot(prices.index, prices['MACD'], label='MACD')
    ax[2].plot(prices.index, prices['signal'], label='Signal')
    ax[2].set_title(f'{ticker}: MACD')
    ax[2].set_xlabel('Date')
    ax[2].set_ylabel('MACD')
    ax[2].legend()
    
    
    # Plot Bollinger Bands
    ax[3].plot(prices.index, prices['Close'], label='Close Price')
    ax[3].plot(prices.index, prices['UpperBand'], label='Upper Band')
    ax[3].plot(prices.index, prices['LowerBand'], label='Lower Band')
    ax[3].fill_between(prices.index, prices['LowerBand'], prices['UpperBand'], alpha=0.35, color='gray', label='Bollinger Bands')
    ax[3].set_title(f'{ticker}: Bollinger Bands')
    ax[3].set_xlabel('Date')
    ax[3].set_ylabel('Price')
    ax[3].legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sp500 = get_sp_tickers()
    get_sp500_stocks(sp500)
    
    
    
    
    
 
    
    
    
    
 
    