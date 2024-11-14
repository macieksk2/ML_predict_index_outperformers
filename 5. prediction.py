# -*- coding: utf-8 -*-
"""
prediction.py

1. Create first training dataset
2. Based on model calibration, select stocks in the latest quarters
"""
### PACKAGES
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import math 
import json

from utils import data_string_to_float, status_calc

def build_data_set(features_list):
    """
    Reads the keystats.csv file and prepares it for scikit-learn
    :return: X_train and y_train numpy arrays
    """
    # Opening JSON file
    f = open('INPUT\input_scrap.json')
    # returns JSON object as a dictionary
    data = json.load(f)
    
    # Retrieve input details
    # The percentage by which a stock has to beat the S&P500 to be considered a 'buy'
    outperformance_  = data['outperform_idx'] * 100
    
    training_data = pd.read_csv("OUTPUT/keystats_NEW.csv", index_col="date")
    training_data.dropna(axis=0, how="any", inplace=True)
    training_data = training_data.replace([np.inf, -np.inf], 0)
    vars_ = ['Year', 'Quarter', 'Ticker', 'Close', 'SP500', 'Close_YoYpc', 'SP500_YoYpc']
    vars_ = vars_ + features_list
    training_data = training_data[vars_]
    features = features_list

    X_train = training_data[features].values
    # Generate the labels: '1' if a stock beats the S&P500 by more than 10%, else '0'.
    y_train = list(
        status_calc(
            training_data["Close_YoYpc"] * 100,
            training_data["SP500_YoYpc"] * 100,
            outperformance_,
        )
    )

    return X_train, y_train


def predict_stocks(datafile_name, quarters, features_list = [], is_past = False):
    print(f'Testing in: {quarters}')
    # Opening JSON file
    f = open('INPUT\input_scrap.json')
    # returns JSON object as a dictionary
    data = json.load(f)
    # Retrieve input details
    # The percentage by which a stock has to beat the S&P500 to be considered a 'buy'
    outperformance_  = data['outperform_idx']
    
    X_train, y_train = build_data_set(features_list)
    # Remove the random_state parameter to generate actual predictions
    clf = RandomForestClassifier(n_estimators=100, random_state=137)
    clf.fit(X_train, y_train)

    # Now we get the actual data from which we want to generate predictions.
    data = pd.read_csv(datafile_name, index_col="date")
    if is_past: 
        # Read in historical stocks and SP500 prices
        hist_daily_stock_prices            = pd.read_csv('OUTPUT/stock_prices.csv')
        hist_daily_idx_prices              = pd.read_csv('OUTPUT/sp500_index.csv')
        hist_daily_stock_prices.index      = hist_daily_stock_prices['Date']
        hist_daily_stock_prices['Year']    = [x.year for x in pd.to_datetime(hist_daily_stock_prices['Date'])]
        hist_daily_stock_prices['Quarter'] = [str(x.year) + 'Q' + str(math.ceil(x.month / 3)) for x in pd.to_datetime(hist_daily_stock_prices['Date'])]
        hist_daily_idx_prices['Year']      = [x.year for x in pd.to_datetime(hist_daily_idx_prices['Date'])]
        hist_daily_idx_prices['Quarter']   = [str(x.year) + 'Q' + str(math.ceil(x.month / 3)) for x in pd.to_datetime(hist_daily_idx_prices['Date'])]
    for q in quarters:
        print(q)
        # Filter specific quarter for testing
        data_flt = data[data.Quarter == q]
        data_flt['Close_YoYpc'] = 0
        data_flt['SP500_YoYpc'] = 0
        data_flt.dropna(axis=0, how="any", inplace=True)
        data_flt = data_flt.replace([np.inf, -np.inf], 0)
        features = features_list
        X_test = data_flt[features].values
        z = data_flt["Ticker"].values
    
        # Get the predicted tickers
        y_pred = clf.predict(X_test)
        if sum(y_pred) == 0:
            print("No stocks predicted!")
        else:
            invest_list = z[y_pred].tolist()
            invest_list = np.unique(invest_list)
            print(f"{len(invest_list)} stocks predicted to outperform the S&P500 by more than {outperformance_}%:")
            print(" ".join(invest_list))
        if is_past:    
            # Select the portfolio, calculate average return in the next year
            beg = hist_daily_stock_prices.loc[(hist_daily_stock_prices['Quarter'] == q),invest_list.tolist()].iloc[-1]
            end = hist_daily_stock_prices.loc[(hist_daily_stock_prices['Quarter'] == q[:3] + str(int(q[3]) + 1) + q[4:]),invest_list.tolist()].iloc[-1]
            portf_ret = np.mean(end / beg - 1)
            beg_idx = hist_daily_idx_prices.loc[(hist_daily_idx_prices['Quarter'] == q),'Close'].iloc[-1]
            end_idx = hist_daily_idx_prices.loc[(hist_daily_idx_prices['Quarter'] == q[:3] + str(int(q[3]) + 1) + q[4:]), 'Close'].iloc[-1]
            sp500_ret = np.mean(end_idx / beg_idx - 1)
            print(f"{portf_ret} is portfolio return")
            print(f"{sp500_ret} is index return")
    return invest_list


if __name__ == "__main__":
    print("Building dataset and predicting stocks...")
    # Selection in last in-sample quarter, calculate average return in the following 12 months against index
    predict_stocks(datafile_name = "OUTPUT/keystats_NEW.csv", 
                   features_list = ['debtequityratio', 'cashonhand_YoYpc', 'grossmargin', 'Price/Book', 'Revenue Per Share', 
                                    'currentratio', 'cashonhand', 'totalliabilities', 'bookvaluepershare'], 
                   quarters = ['2023Q3'], is_past = True)
    # Selection in the upcoming four quarters, where no forward return has been calculated
    predict_stocks("OUTPUT/keystats_NEW_OOS.csv", 
                   features_list = ['debtequityratio', 'cashonhand_YoYpc', 'grossmargin', 'Price/Book', 'Revenue Per Share', 
                                    'currentratio', 'cashonhand', 'totalliabilities', 'bookvaluepershare'], 
                   quarters = ['2023Q4', '2024Q1', '2024Q2','2024Q3'])
