Stock Prediction Project

Overview
This repository contains a Python script (train.py) designed to process stock data and train a machine learning model for stock price prediction. The goal of this script is to individually handle CSV files for different companies, train a Long Short-Term Memory (LSTM) model on each company's data, and produce a graph of the predictions.

Script Details
train.py
The train.py script performs the following tasks:

Data Preparation:

Loads stock data from CSV files.
Computes moving averages (MA50 and MA200).
Scales the data using Min-Max Scaling.
Model Training:

Creates an LSTM model to predict stock prices.
Trains the model using a portion of the data and evaluates its performance.
Model Selection:

Tries different hyperparameters (LSTM units, dense layer units, epochs) to find the best model based on RMSE (Root Mean Squared Error).
Model Saving:

Saves the best-performing model to the models directory.
Prediction Visualization:

Generates a graph of the predicted vs. actual stock prices.
Dataset
The dataset used in this project consists of CSV files where each file contains historical stock data for a company. The columns typically include:

Open: Opening price of the stock
High: Highest price of the stock
Low: Lowest price of the stock
Close: Closing price of the stock
Volume: Trading volume
The dataset files should be placed in the stock_test directory.

