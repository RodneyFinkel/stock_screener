from stock2 import (Stock, filter_sector, filter_price, filter_metric, filter_technical_indicator, get_stock_price, get_historical)
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
        # Get data for training and testing
        filtered_stocks = self.apply_filters()
       
        for stock in filtered_stocks:
            train_data = stock.technical_indicators
            train_labels = stock.labels
            
            # Ensure train_data is a 2D array
            # train_data = np.array(train_data).reshape(-1, 1)
            
            # Normalize the Data
            train_data = self.scaler.fit_transform(train_data)
            train_labels = np.array(train_labels)
            
            #Create and train model
            model = create_model(train_data) 
            model.fit(train_data, train_labels, epochs=20)
            self.models[stock.ticker] = model # models needs to be defined as a StockScreener attribute, namely a dictionary
            
    # Predict whether new stocks will pass filters (new_stocks gets passed as filtered_stocks in app.py)
    def predict_stocks(self, new_stocks):
        # Add technical indicators to new stocks
        for stock in new_stocks:
            stock.get_historical()
            stock.add_technical_indicators()
            
        
        # Make predictions for each stock using its corresponding model
        predicted_stocks = []
        stock_instance_predictions = []
        for stock in new_stocks:
            if stock.ticker in self.models:
                model = self.models[stock.ticker]
                # Reshape as there is only one sample
                new_feature_aux = np.array(stock.today_technical_indicators).reshape(1,-1)
                new_stock_data = self.scaler.fit_transform(new_feature_aux)
                prediction = model.predict(new_stock_data)
                stock.prediction = prediction
                print('Predictions', prediction)
                if prediction > 0.5:
                    predicted_stocks.append(stock)
                    stock_instance_predictions.append(prediction)
        print('Predicted_Stocks:', predicted_stocks, 'Stock_Instance_Predictions', stock_instance_predictions)           
        return predicted_stocks
    
# The create_model() function needs to be defined outside the StockScreener Class to be in scope  
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
    
    
    
    
                
            
                          
            
            
            
             