from bs4 import BeautifulSoup
import requests
from pprint import pprint
import yfinance as yf
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots



#ticker_symbol = input('Please select ticker symbol: ')

def get_sp500_stocks(ticker_symbol):
    
    sp500_stocks = []
    print('ticker')
    price = get_stock_price2(ticker_symbol)
    print(price)
    data = get_historical(ticker_symbol)
    technical_data, prices  = add_technical_indicators(data)
    print(prices.head()) 
    print(prices.shape)
    prices_mod = prices[['MA20', 'MA50', 'RSI', 'MACD', 'UpperBand', 'LowerBand',]].iloc[-1, :]
    print(prices_mod)
    sp500_stocks.append((ticker_symbol, price))
    plot_technical_indicators(prices, ticker_symbol)
                
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
    history = stock.history(start='2019-01-01', end='2024-05-04') 
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
    
    # Set background color to an even darker grey
    #plt.style.use('bmh')
    plt.rcParams['figure.facecolor'] = '#555555'  # Set figure background color
    plt.rcParams['axes.facecolor'] = '#555555'    # Set axes background color
    plt.rcParams['grid.color'] = 'white'          # Set gridlines color
     
    fig, ax = plt.subplots(4, 1, figsize=(16, 10), dpi=100)

    # Plot 20-day and 50-day moving averages
    ax[0].plot(prices.index, prices['Close'], label='Close Price', color='white')  # Change color to white
    ax[0].plot(prices.index, prices['MA20'], label='20-day MA', color='blue')      # Change color to blue
    ax[0].plot(prices.index, prices['MA50'], label='50-day MA', color='green')     # Change color to green
    ax[0].fill_between(prices.index, prices['MA20'], prices['MA50'], alpha=0.35, color='#555555', label='Moving Averages')  # Change fill color to an even darker grey
    ax[0].set_title(f'{ticker}: Moving Averages', color='white')  # Set title color to white
    ax[0].set_xlabel('Date', color='white')  # Set xlabel color to white
    ax[0].set_ylabel('Price', color='white')  # Set ylabel color to white
    ax[0].grid(True, color='black', linestyle='--')  # Add gridlines
    ax[0].legend()

    # Plot RSI
    ax[1].plot(prices.index, prices['RSI'], label='RSI', color='cyan')  # Change color to cyan
    ax[1].axhline(y=70, color='r', linestyle='--', label='Overbought')
    ax[1].axhline(y=30, color='g', linestyle='--', label='Oversold')
    ax[1].set_title(f'{ticker}: RSI', color='white')  # Set title color to white
    ax[1].set_xlabel('Date', color='white')  # Set xlabel color to white
    ax[1].set_ylabel('RSI', color='white')  # Set ylabel color to white
    ax[1].grid(True, color='black', linestyle='--')  # Add gridlines
    ax[1].legend()

    # Plot MACD
    ax[2].plot(prices.index, prices['MACD'], label='MACD', color='orange')  # Change color to orange
    ax[2].plot(prices.index, prices['signal'], label='Signal', color='yellow')  # Change color to yellow
    ax[2].set_title(f'{ticker}: MACD', color='white')  # Set title color to white
    ax[2].set_xlabel('Date', color='white')  # Set xlabel color to white
    ax[2].set_ylabel('MACD', color='white')  # Set ylabel color to white
    ax[2].grid(True, color='black', linestyle='--')  # Add gridlines
    ax[2].legend()

    # Plot Bollinger Bands
    ax[3].plot(prices.index, prices['Close'], label='Close Price', color='magenta')  # Change color to magenta
    ax[3].plot(prices.index, prices['UpperBand'], label='Upper Band', color='blue')  # Change color to blue
    ax[3].plot(prices.index, prices['LowerBand'], label='Lower Band', color='green')  # Change color to green
    ax[3].fill_between(prices.index, prices['LowerBand'], prices['UpperBand'], alpha=0.35, color='#555555')  # Change fill color to an even darker grey
    ax[3].set_title(f'{ticker}: Bollinger Bands', color='white')  # Set title color to white
    ax[3].set_xlabel('Date', color='white')  # Set xlabel color to white
    ax[3].set_ylabel('Price', color='white')  # Set ylabel color to white
    ax[3].grid(True, color='black', linestyle='--')  # Add gridlines
    ax[3].legend()

    plt.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":
    get_sp500_stocks(ticker_symbol)
    