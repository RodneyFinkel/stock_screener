from stock2 import (Stock, filter_sector, filter_price, filter_metric, filter_technical_indicators, get_stock_price, get_stock_price2, get_historical, add_technical_indicators)
import concurrent.futures
import requests
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler 
import tensorflow as tf
import streamlit as st
from bs4 import BeautifulSoup

       
# Stock Screener Class
class StockScreener:
    def __init__(self, stocks, filters):
        self.stocks = stocks
        self.filters = filters
        self.scaler = StandardScaler()
        self.models = {}
        
    # Select stocks that pass all filters
    def apply_filters(self):
        filtered_stocks = []
        for stock in self.stocks:
            passed_all_filters = True
            for filter_func in self.filters:
                if not filter_func(stock):
                    passed_all_filters = False
                    break
            if passed_all_filters:
                filtered_stocks.append(stock)  
        return filtered_stocks
    
              
   # Train deep learning models on selected stocks
    def train_models(self):
        training_models = st.empty()
        training_models.write('Training Model for each Ticker...')
        # Get data for training and testing
        filtered_stocks = self.apply_filters()
       
        for stock in filtered_stocks:
            train_data = stock.technical_indicators
            train_labels = stock.label
            if len(train_data) == 0:
                continue
            
            # Ensure train_data is a 2D array
            # train_data = np.array(train_data).reshape(-1, 1)
            
            # Normalize the Data
            train_data = self.scaler.fit_transform(train_data)
            train_labels = np.array(train_labels)
            print(f"train_data shape2: {train_data.shape}, train_labels shape2: {train_labels.shape}")
            
            #Create and train model
            model = create_model(train_data) 
            history = model.fit(train_data, train_labels, epochs=100)
            self.models[stock.ticker] = model, history # models needs to be defined as a StockScreener attribute, namely a dictionary
            
        training_models.empty()
        
        return filtered_stocks
            
    # Predict whether new stocks will pass filters 
    def predict_stocks(self, new_stocks):
        # Make predictions for each stock using its corresponding model
        predicted_stocks = []
        for stock in new_stocks:
            if stock.ticker in self.models:
                model, _ = self.models[stock.ticker]
                # Reshape as there is only one sample
                new_features_aux = np.array(stock.today_technical_indicators).reshape(1,-1)
                new_stock_data = self.scaler.fit_transform(new_features_aux)
                prediction = model.predict(new_stock_data)
                stock.prediction = prediction
                print('Predictions', prediction)
                if prediction > 0.5:
                    predicted_stocks.append(stock)
                  
        return predicted_stocks
    
    # Create web app for stock screener 
    def create_app(self):
        
        st.title(':blue[S&P500 Stock Screener with NN based Predictive Model]')
        st.text('Select Stocks News aggregator with article sentiment analysis')
        # Create sidebar for filtering options
        sector_list = sorted(list(set(stock.sector for stock in self.stocks)))
        selected_sector = st.sidebar.selectbox('Sector', ['All'] + sector_list) # parameters: name of the selectbox, choices within the selectbox
        
        min_price = st.sidebar.number_input('Min Price', value=0.0, step=0.01)
        max_price = st.sidebar.number_input('Max Price', value=1000000.0, step=0.01)
        
        metric_list = sorted(list(set(metric for stock in self.stocks for metric in stock.metrics)))
        selected_metric = st.sidebar.selectbox('Metric', ['All'] +  metric_list)
        
        metric_operator_list = ['>', '>=', '<', '<=', '==']
        selected_metric_operator = st.sidebar.selectbox('Metric Operator', metric_operator_list)
        
        metric_value = st.sidebar.text_input('Metric Value', 'Enter value or the word price')
        try:
            metric_value = float(metric_value)
            print(metric_value)
        except:
            pass
        
        indicator_list = sorted(list(set(indicator for stock in self.stocks for indicator in stock.today_technical_indicators.keys())))
        selected_indicator = st.sidebar.selectbox("Indicator", ["All"] + indicator_list)

        indicator_operator_list = [">", ">=", "<", "<=", "=="]
        selected_indicator_operator = st.sidebar.selectbox("Indicator Operator", indicator_operator_list)

        indicator_value = st.sidebar.text_input("Indicator Value", "Enter value or the word price")
        try:
            indicator_value = float(indicator_value)
            print(indicator_value)
        except:
            pass
        
        # update filter list with user inputs
        new_filters = []
        if selected_sector != 'All':
            new_filters.append(lambda stock: filter_sector(stock, selected_sector))
        if selected_metric != 'All':
            new_filters.append(lambda stock: filter_metric(stock, selected_metric, selected_metric_operator, metric_value))
        if selected_indicator != 'All':
            new_filters.append(lambda stock: filter_technical_indicators(stock, selected_indicator, selected_indicator_operator, indicator_value))
        new_filters.append(lambda stock: filter_price(stock, min_price, max_price))
        self.filters = new_filters
        
        # Create 'Apply Filters' button
        if st.sidebar.button('Apply Filters'):
            
            # Apply Filters
            filtered_stocks = self.apply_filters()
            
            # Display Visualizations for filtered stocks
            display_filtered_stocks(filtered_stocks, selected_metric, selected_indicator)
            
        # Create 'Train and Predict Models' button
        if st.sidebar.button('NN Train and Predict'):
            # Train models for each filtered stock
            filtered_stocks = self.train_models()
            # Predict Models
            predicted_stocks = self.predict_stocks(filtered_stocks)
            
            # Display visualizations for filtered stocks
            display_filtered_stocks(predicted_stocks, selected_metric, selected_indicator, self.models)
            
 
# Simple Dense Model
def create_model(train_data):
        # Creating a sequential model (neural network suitable for binary classification tasks)
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, input_shape=(train_data.shape[1],), activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
            
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model   
    
# CNN model
def create_cnn_model(train_data):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(train_data.shape[1], 1)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return cnn_model 
    
def display_filtered_stocks(filtered_stocks, selected_metric, selected_indicator, models=None):
    
    with open('style.css') as f:
        css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    # Display Filtered Stocks
    if len(filtered_stocks) == 0:
        st.write('No stocks match the specified criteria after Screening and Predicting')
    else:
        filtered_tickers = [stock.ticker for stock in filtered_stocks]
        tabs = st.tabs(filtered_tickers)
        for n in range(len(tabs)):
            # Divide the metrics into 3 lists for the 3 columns
            # each item of filtered_stocks is an object of the stock class and metric is a class attribute so metric are accesible with .metric
            metrics = list(filtered_stocks[n].metrics.items())    # .items is a method that returns a view of the metrics as a list of tuples (key-value pairs)
            num_metrics = len(metrics)
            col1_metrics = metrics[:num_metrics//3]
            col2_metrics = metrics[num_metrics//3:(2*num_metrics)//3]
            col3_metrics = metrics[(2*num_metrics)//3:]
            # Create 3 columns inside each tab
            col1, col2, col3 = tabs[n].columns(3)
            
            # Display the metrics in 3 columns
            for metric, value in col1_metrics:
                col1.metric(metric, value)
            for metric, value in col2_metrics:
                col2.metric(metric, value)
            for metric, value in col3_metrics:
                col3.metric(metric, value)
            
            # # Plot Closing Price    
            # fig, ax = plt.subplots(4, 1, figsize=(20, 18))
            # ax[0].plot(filtered_stocks[n].data.index, filtered_stocks[n].data['Close'])
            # ax[0].set_title(f'{filtered_stocks[n].ticker} Close Price')
            # ax[0].set_xlabel('Date')
            # ax[0].set_ylabel('Price')
            
            # # Plot Bollinger Bands
            # ax[1].plot(filtered_stocks[n].data.index[-1095:], filtered_stocks[n].data['Close'][-1095:], label='Close Price')
            # ax[1].plot(filtered_stocks[n].data.index[-1095:], filtered_stocks[n].data['UpperBand'][-1095:], label='Upper Band')
            # ax[1].plot(filtered_stocks[n].data.index[-1095:], filtered_stocks[n].data['LowerBand'][-1095:], label='Lower Band')
            # ax[1].fill_between(filtered_stocks[n].data.index[-1095:], filtered_stocks[n].data['LowerBand'][-1095:], filtered_stocks[n].data['UpperBand'][-1095:], alpha=0.35, color='gray', label='Bollinger Bands')
            # ax[1].set_title(f'{filtered_stocks[n].ticker}: Bollinger Bands')
            # ax[1].set_xlabel('Date_altered')
            # ax[1].set_ylabel('Price')
            
            # # Plot MACD
            # ax[2].plot(filtered_stocks[n].data.index[-1095:], filtered_stocks[n].data['MACD'][-1095:], label='MACD')
            # # ax[2].plot(filtered_stocks[n].data.index, filtered_stocks[n].data['signal'], label='Signal')
            # ax[2].set_title(f'{filtered_stocks[n].ticker}: MACD')
            # ax[2].set_xlabel('Date')
            # ax[2].set_ylabel('MACD')
            
            # # Plot 20-day and 50-day moving averages
            # ax[3].plot(filtered_stocks[n].data.index[-730:], filtered_stocks[n].data['Close'][-730:], label='Close Price')
            # ax[3].plot(filtered_stocks[n].data.index[-730:], filtered_stocks[n].data['MA20'][-730:], label='20-day MA')
            # ax[3].plot(filtered_stocks[n].data.index[-730:], filtered_stocks[n].data['MA50'][-730:], label='50-day MA')
            # ax[3].fill_between(filtered_stocks[n].data.index[-730:], filtered_stocks[n].data['MA20'][-730:], filtered_stocks[n].data['MA50'][-730:], alpha=0.35, color='gray', label='Moving Averages')
            # ax[3].set_title(f'{filtered_stocks[n].ticker}: Moving Averages')
            # ax[3].set_xlabel('Date_altered')
            # ax[3].set_ylabel('Price')
            
            
            # # Streamlit's pyplot function to display the matplotlib figure (fig) inside the current tab (tabs[n]) of the Streamlit app
            # tabs[n].pyplot(fig)
            
            
            #### Display stock graphs ####
            # Set the style to a dark background
            plt.style.use('dark_background')

            # Plot Closing Price    
            fig, ax = plt.subplots(4, 1, figsize=(20, 18))
            ax[0].plot(filtered_stocks[n].data.index, filtered_stocks[n].data['Close'], color='white')  # Specify line color
            ax[0].set_title(f'{filtered_stocks[n].ticker} Close Price', color='white')  # Specify title color
            ax[0].set_xlabel('Date', color='white')  # Specify xlabel color
            ax[0].set_ylabel('Price', color='white')  # Specify ylabel color
            ax[0].grid(True, color='black', linestyle='--')  # Add gridlines
            ax[0].set_facecolor('white')  # Set background color
            ax[0].tick_params(axis='x', colors='white')  # Set x-axis tick color
            ax[0].tick_params(axis='y', colors='white')  # Set y-axis tick color

            # Plot Bollinger Bands
            ax[1].plot(filtered_stocks[n].data.index[-1095:], filtered_stocks[n].data['Close'][-1095:], label='Close Price', color='white')
            ax[1].plot(filtered_stocks[n].data.index[-1095:], filtered_stocks[n].data['UpperBand'][-1095:], label='Upper Band', color='yellow')  # Specify Upper Band color
            ax[1].plot(filtered_stocks[n].data.index[-1095:], filtered_stocks[n].data['LowerBand'][-1095:], label='Lower Band', color='red')  # Specify Lower Band color
            ax[1].fill_between(filtered_stocks[n].data.index[-1095:], filtered_stocks[n].data['LowerBand'][-1095:], filtered_stocks[n].data['UpperBand'][-1095:], alpha=0.35, color='gray', label='Bollinger Bands')
            ax[1].set_title(f'{filtered_stocks[n].ticker}: Bollinger Bands', color='white')
            ax[1].set_xlabel('Date_altered', color='white')
            ax[1].set_ylabel('Price', color='white')
            ax[1].grid(True, color='black', linestyle='--')  # Add gridlines
            ax[1].set_facecolor('white')  # Set background color
            ax[1].tick_params(axis='x', colors='white')  # Set x-axis tick color
            ax[1].tick_params(axis='y', colors='white')  # Set y-axis tick color

            # Plot MACD
            ax[2].plot(filtered_stocks[n].data.index[-1095:], filtered_stocks[n].data['MACD'][-1095:], label='MACD', color='cyan')  # Specify MACD color
            ax[2].set_title(f'{filtered_stocks[n].ticker}: MACD', color='white')
            ax[2].set_xlabel('Date', color='white')
            ax[2].set_ylabel('MACD', color='white')
            ax[2].grid(True, color='black', linestyle='--')  # Add gridlines

            # Plot 20-day and 50-day moving averages
            ax[3].plot(filtered_stocks[n].data.index[-730:], filtered_stocks[n].data['Close'][-730:], label='Close Price', color='white')
            ax[3].plot(filtered_stocks[n].data.index[-730:], filtered_stocks[n].data['MA20'][-730:], label='20-day MA', color='green')  # Specify 20-day MA color
            ax[3].plot(filtered_stocks[n].data.index[-730:], filtered_stocks[n].data['MA50'][-730:], label='50-day MA', color='orange')  # Specify 50-day MA color
            ax[3].fill_between(filtered_stocks[n].data.index[-730:], filtered_stocks[n].data['MA20'][-730:], filtered_stocks[n].data['MA50'][-730:], alpha=0.35, color='gray', label='Moving Averages')
            ax[3].set_title(f'{filtered_stocks[n].ticker}: Moving Averages', color='white')
            ax[3].set_xlabel('Date_altered', color='white')
            ax[3].set_ylabel('Price', color='white')
            ax[3].grid(True, color='black', linestyle='--')  # Add gridlines
            ax[3].set_facecolor('white')  # Set background color
            ax[3].tick_params(axis='x', colors='white')  # Set x-axis tick color
            ax[3].tick_params(axis='y', colors='white')  # Set y-axis tick color

            # Adjusting the background color of subplots
            bkg_color = '#333333' 
            for a in ax:
                a.set_facecolor(bkg_color)  # Set subplot background color

            # Streamlit's pyplot function to display the matplotlib figure (fig) inside the current tab (tabs[n]) of the Streamlit app
            tabs[n].pyplot(fig)
            
            try: 
                model, history = models[filtered_stocks[n].ticker]
                plt.style.use('dark_background')
                fig, ax = plt.subplots(2, 1, figsize=(10, 8))
                # Plot training loss
                ax[0].plot(history.history['loss'])
                ax[0].set_title(f"{filtered_stocks[n].ticker} Training Loss")
                ax[0].set_ylabel("Loss")
                ax[0].grid(True, color='black', linestyle='--')  # Add gridlines
                # Plot training accuracy
                ax[1].plot(history.history['accuracy'])
                ax[1].set_title(f"{filtered_stocks[n].ticker} Training Accuracy")
                ax[1].set_xlabel("Epoch")
                ax[1].set_ylabel("Accuracy")
                ax[1].grid(True, color='black', linestyle='--')  # Add gridlines
                
                # Show the plot in the streamlit app
                dark_grey = '#333333' 
                for a in ax:
                    a.set_facecolor(dark_grey)
                tabs[n].pyplot(fig)           
         
            except:
                
                tabs[n].write("")
                
    # Display table of filtered stocks info
    table_data = [[s.ticker, s.sector, s.price, s.metrics.get(selected_metric, "N/A"), s.today_technical_indicators.get(selected_indicator, "N/A"), float(s.prediction) if s.prediction != 0 else "N/A"] for s in filtered_stocks]
    table_columns = ["Ticker", "Sector", "Price", f"Metric: {selected_metric}", f"Indicator: {selected_indicator}", "Prediction" if any(s.prediction != 0 for s in filtered_stocks) else ""]
    st.write(pd.DataFrame(table_data, columns=table_columns))
    

## GET SP 500 STOCK DATA ##

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

# Run screener for all sp500 tickers
# @st.cache_data
# def get_sp500_stocks(sp500):
    
#     sp500_stocks = []
#     # Streamlit Text
#     stock_download = st.empty()
#     stock_issues = st.empty()
#     # Create Stock object for every stock with data
#     for stock in sp500:
#         stock_download.write(f'Downloading {stock["ticker"]} Data')
#         try:
#             price = get_stock_price2(stock['ticker']) # yf.info from get_stock_price2()in stock2.py not working
#             print(price)
#             data = get_historical(stock['ticker'])
#             #print(f"Ticker: {stock['ticker']}, Sector: {stock['sector']}, Price: {price}, Data: {data}")
#             print('test1')
#             #test = Stock(stock['ticker'], stock['sector'], price, data)
#             sp500_stocks.append(Stock(stock['ticker'], stock['sector'], price, data))
#             #print(sp500_stocks)
#             print('all debug tests passed')
#             stock_download.empty()
#         except:
#             print('GAAAAAAHHH!!')
#             stock_issues.write(f'There was an issue with {stock["ticker"]}.')
            
#     stock_issues.empty()
#     return sp500_stocks      

# Run screener for all sp500 tickers
# @st.cache_data
# def get_sp500_stocks(sp500):
    
#     sp500_stocks = [] 
#     # Streamlit Text
#     stock_download = st.empty()
#     stock_test = st.empty()
#     stock_issues = st.empty()
#     # # Create Stock object for every stock with data
#     for stock in sp500:
#         stock_download.write(f'Downloading {stock["ticker"]} Data')
        
#         try:
#             price = get_stock_price2(stock['ticker']) 
#             print(price)
#             data = get_historical(stock['ticker'])
#             technical_data, prices  = add_technical_indicators(data)
#             prices_last = prices[['MA20', 'MA50', 'RSI', 'MACD', 'UpperBand', 'LowerBand',]].iloc[-1, :]
#             stock_test.write(f"Ticker: {stock['ticker']}, Sector: {stock['sector']}, Price:{price}, Prices:{prices_last}, Technical_Data:{technical_data.columns} ")
#             print('test1')
#             stock_instance = Stock(stock['ticker'], stock['sector'], price, data)
#             sp500_stocks.append(stock_instance)
#             # sp500_stocks.append(Stock(stock['ticker'], stock['sector'], price, data))
#             print('all debug tests passed')
#             stock_download.empty()
#             stock_test.empty()
#         except:
#             print('GAAAAAAHHH!!')
#             stock_issues.write(f'There was an issue with {stock["ticker"]}.')
    
    
            
#     stock_issues.empty()
#     return sp500_stocks                
                          
                          
# Multithreaded alternative
def fetch_stock_data(stock):
    try:
        price = get_stock_price2(stock['ticker'])
        data = get_historical(stock['ticker'])
        technical_data, prices = add_technical_indicators(data)
        prices_last = prices[['MA20', 'MA50', 'RSI', 'MACD', 'UpperBand', 'LowerBand',]].iloc[-1, :]
        
        print('test1')
        return Stock(stock['ticker'], stock['sector'], price, data)
    except Exception as e:
        print(f"Error fetching data for {stock['ticker']}: {e}")
        return None


# Run screener for all sp500 tickers
# Function to get SP500 stocks concurrently
@st.cache_data
def get_sp500_stocks(sp500):
    sp500_stocks = []
    stock_download = st.empty()
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Use ThreadPoolExecutor to fetch stock data concurrently
        futures = {executor.submit(fetch_stock_data, stock): stock for stock in sp500}
        
        for future in concurrent.futures.as_completed(futures):
            stock = futures[future]
            try:
                stock_instance = future.result()
                if stock_instance:
                    sp500_stocks.append(stock_instance)
                    print('all debug tests passed')
            except Exception as e:
                print(f"Error processing {stock['ticker']}: {e}")

    stock_download.empty()
    return sp500_stocks                             
                
            
            
    
    
    
    
    
                
            
                          
            
            
            
             