### STOCK SCREENER MAIN CLASS ###

from stock_screener2 import StockScreener, get_sp_tickers, get_sp500_stocks
import streamlit as st

if __name__ == '__main__':
    
    # Streamlit Config
    st.set_page_config(page_title='Stock Screener', page_icon=':chart_with_upward_trend:')

    filters = []
    
    sp500 = get_sp_tickers()
    # Get sp500 tickers and sectors
    sp500_stocks = get_sp500_stocks(sp500)
    # Screener
    screener = StockScreener(sp500_stocks, filters)
    
    # Create streamlit app
    screener.create_app()
   
