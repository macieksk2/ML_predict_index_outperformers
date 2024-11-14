### PACKAGES
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import time
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import pandas_datareader.data as web
from pandas.api.types import CategoricalDtype
import os
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup as bs
import requests, re, json
from pandas_datareader import data as pdr

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import psycopg2.extras

################## FUNCTIONS #####################################
### 1. data scrapping.py
def retrieve_data(driver, url, item = "financial-statements"):
    """
    Retrieve data (numbers, dates) for each item from macrotrends:
        - financial-statements
        - balance-sheet
        - cash-flow-statement
        - financial-ratios
    """
    
    header = {
      "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
      "X-Requested-With": "XMLHttpRequest"
    }
    URL = url + "/" + item + "?freq=Q"
    time.sleep(4)
    r = requests.get(URL, headers = header)
    p = re.compile(r'var originalData = (.*);')
    p2 = re.compile(r'datafields:[\s\S]+(\[[\s\S]+?\]),')
    p3 = re.compile(r'\d{4}-\d{2}-\d{2}')
    data = json.loads(p.findall(r.text)[0])
    s = re.sub('\r|\n|\t|\s','',p2.findall(r.text)[0])
    fields = p3.findall(s)
    # Only headers of interest.
    fields.insert(0, 'field_name') 
    results = []
    
    # Loop initial list of dictionaries
    for item in data: 
        row = {}
        # Loop keys of interest to extract from current dictionary
        for f in fields: 
            # This is an html value field so needs re-parsing
            if f == 'field_name':  
                soup2 = bs(item[f],'lxml')
                row[f] = soup2.select_one('a,span').text
            else:
                row[f] = item[f]
        results.append(row)
    
    df = pd.DataFrame(results, columns = fields) 
    df_t = df.transpose()
    df_t.columns = df_t.iloc[0]
    df_t = df_t.iloc[1:]
    
    return df_t

def repl_sp_signs(obj):
    return obj.replace(",",".").replace("$","").replace(" ", "")
    

def def_headers(obj):
    """
    Define headers for data
    """
    last_key = None
    out = {}
    for i in obj:
        if not (i.replace('.','').isdigit() or i == '-' or i.startswith('-')):
            last_key = i
        elif last_key in out:
            out[last_key].append(i)
        else:
            out[last_key] = [i] 
    return out


def execute_values(conn, df, table):
    """
    https://www.geeksforgeeks.org/how-to-write-pandas-dataframe-to-postgresql-table/
    """
    tuples = [tuple(x) for x in df.to_numpy()]
    cols = ','.join(list(df.columns))
    # SQL query to execute
    query = "INSERT INTO %s(%s) VALUES %%s" % (table, cols)
    cursor = conn.cursor()
    try:
        psycopg2.extras.execute_values(cursor, query, tuples)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    print("execute_values() done")
    cursor.close()

def generate_financial_plots(input_, tick_):
    """
    Create a plot with four rows and two columns including relevant financial features:
        - Revenue
        - Income variables
        - Cash
        - Debt
        - EPS
        - ROE / ROA / ROI
        - Evolution of number of shares
    """
    # Generate plots
    fig1, f1_axes = plt.subplots(ncols=2, nrows=5, figsize=(30,20))
    fig1.suptitle (tick_, size=50)
    # Revenue vs Income
    f1_axes[0, 0].plot(input_['Revenue'], lw=2, marker='.', markersize=10, label="Revenue")
    f1_axes[0, 0].plot(input_['GrossProfit'], lw=2, marker='.', markersize=10, label="Gross Profit")
    f1_axes[0, 0].plot(input_['NetIncome'], lw=2, marker='.', markersize=10, label="Net Income")
    f1_axes[0, 0].plot(input_['EBITDA'], lw=2, marker='.', markersize=10, label="EBITDA")
    # Cash vs Debt
    f1_axes[1, 0].plot(input_['CashOnHand'], lw=2, marker='.', markersize=10, label="Cash on Hand")
    f1_axes[1, 0].plot(input_['LongTermDebt'], lw=2, marker='.', markersize=10, label="Long Term Debt")
    # Cashflows
    f1_axes[2, 0].plot(input_['CashFlowFromOperatingActivities'], lw=2, marker='.', markersize=10, label="CF from Operating Activity")
    f1_axes[2, 0].plot(input_['FreeCashFlowPerShare'] * input_['SharesOutstanding'] / 1000, lw=2, marker='.', markersize=10, label="FCF")
    f1_axes[2, 0].plot(input_['CashFlowFromInvestingActivities'], lw=2, marker='.', markersize=10, label="CF from Investing Activities")
    f1_axes[2, 0].plot(input_['NetIncome'], lw=2, marker='.', markersize=10, label="Net Income")
    # Assets vs Equity
    f1_axes[3, 0].plot(input_['TotalAssets'], lw=2, marker='.', markersize=10, label="Total Assets")
    f1_axes[3, 0].plot(input_['ShareHolderEquity'], lw=2, marker='.', markersize=10, label="Shareholder Equity")
    # Current Ratio
    f1_axes[4, 0].plot(input_['CurrentRatio'], lw=2, marker='.', markersize=10, label="Current Ratio") # ['CurrentRatio']/10000
    # Assets / Liab
    f1_axes[0, 1].plot(input_['TotalAssets'], lw=2, marker='.', markersize=10, label="Total Assets")
    f1_axes[0, 1].plot(input_['TotalLiabilities'], lw=2, marker='.', markersize=10, label="Total Liabilities")
    f1_axes[0, 1].plot(input_['TotalCurrentAssets'], lw=2, marker='.', markersize=10, label="Current Assets")
    f1_axes[0, 1].plot(input_['TotalCurrentLiabilities'], lw=2, marker='.', markersize=10, label="Current Liabilities")
    # EPS
    f1_axes[1, 1].plot(input_['EPS-EarningsPerShare'], lw=2, marker='.', markersize=10, label="EPS")
    # Ratios
    f1_axes[2, 1].plot(input_['ROE-ReturnOnEquity']/10000, lw=2, marker='.', markersize=10, label="ROE")
    f1_axes[2, 1].plot(input_['ROA-ReturnOnAssets']/10000, lw=2, marker='.', markersize=10, label="ROA")
    f1_axes[2, 1].plot(input_['ROI-ReturnOnInvestment']/10000, lw=2, marker='.', markersize=10, label="ROI")
    # D / E
    f1_axes[3, 1].plot(input_['Debt/EquityRatio'], lw=2, marker='.', markersize=10, label="Debt/Equity Ratio") # ['Debt/EquityRatio']/10000
    # Shares Outstanding
    f1_axes[4, 1].plot(input_['SharesOutstanding'], lw=2, marker='.', markersize=10, label="Shares Outstanding")
     
    f1_axes[0, 0].legend()
    f1_axes[1, 0].legend()
    f1_axes[2, 0].legend()
    f1_axes[3, 0].legend()
    f1_axes[4, 0].legend()
    f1_axes[0, 1].legend()
    f1_axes[1, 1].legend()
    f1_axes[2, 1].legend()
    f1_axes[3, 1].legend()
    f1_axes[4, 1].legend()
    
def yf_scrap_stock_price_iteratively(idx_start, idx_end):
    """
    This is an alternative iterative solution to building the stock dataset, which may be necessary if the
    tickerlist is too big.
    Instead of downloading all at once, we download ticker by ticker and append to a dataframe.
    This will download data for tickerlist[idx_start:idx_end], which makes this method suitable
    for chunking data.

    :param idx_start: (int) the starting index of the tickerlist
    :param idx_end: (int) the end index of the tickerlist
    """

    # Opening JSON file
    f = open('INPUT\input_scrap.json')
    # Returns JSON object as a dictionary
    data = json.load(f)
    
    # Retrieve input details
    start = data['YF_START_DATE']
    end   = data['YF_END_DATE']
    tickers_filename = data['tickers_filename']
    # Pull tickers csv
    tickers = pd.read_csv('INPUT/' + tickers_filename)
    ticker_list = tickers['sp500_yf_tickers']
    df = pd.DataFrame()

    for ticker in ticker_list:
        ticker = ticker.upper()

        stock_ohlc = pdr.get_data_yahoo(ticker, start=start, end=end)
        if stock_ohlc.empty:
            print(f"No data for {ticker}")
            continue
        adj_close = stock_ohlc["Adj Close"].rename(ticker)
        df = pd.concat([df, adj_close], axis=1)
    df.to_csv("OUTPUT/stock_prices.csv")
    
### 2. data processing.py    
def moving_average(a, n=3):
    """
    Moving average based on numpy array
    At the beginning add NAs fit DF format / length
    """
    a_tr = np.nan_to_num(a, nan = 0)
    ret = np.cumsum(a_tr, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    out =  ret[n - 1:] / n
    # Add NAs at the beginning (not calculated MAs) to fit DataFrame format
    out = np.concatenate(([np.nan] * (n - 1), out))
    return out

def fill_nas(database, var_, idx, is_var_handle = False, handle_type = 'ratio', handle_ratio_nom = "", handle_ratio_denom = ""):
    """
    EPS = Net Income / Sharesout
    EBITDA = Net Income
    Grossmargin = Gross profit / revenue
    Operating margin = Operating income / revenue
    Current ratio = Total current assets / Total current liabilities
    Long Term Debt = Total Liabilities
    D / E = Long Term Debt / Shareholder Equity
    
    handle_type = {'ratio', 'value'}
    """
    # In case of all values being empty, either fill in with a ratio of two variables or one defined feature
    if database[database['Ticker'] == idx][var_].isna().sum() == len(database[database['Ticker'] == idx]):
        if is_var_handle and handle_type == 'ratio':
            database.loc[database['Ticker'] == idx, var_] = database[database['Ticker'] == idx][handle_ratio_nom] / database[database['Ticker'] == idx][handle_ratio_denom]
        elif is_var_handle and handle_type == 'value':
            database.loc[database['Ticker'] == idx, var_] = database[database['Ticker'] == idx][handle_ratio_nom]
        else:
            database.loc[database['Ticker'] == idx, var_] = 0
    # In case less values are NAs, forward fill the values. Remaining fill with 0
    elif database[database['Ticker'] == idx][var_].isna().sum() > 0:
        database.loc[database['Ticker'] == idx, var_] = database.loc[database['Ticker'] == idx, var_].fillna(method ='ffill')
        database.loc[(database['Ticker'] == idx) & (database[var_].isna()), var_] = 0
        
    return database.loc[database['Ticker'] == idx, var_]
        

### 3. optimal features.py    
def status_calc(stock, sp500, outperformance=10):
    """A simple function to classify whether a stock outperformed the S&P500
    :param stock: stock price
    :param sp500: S&P500 price
    :param outperformance: stock is classified 1 if stock price > S&P500 price + outperformance
    :return: true/false
    """
    if outperformance < 0:
        raise ValueError("outperformance must be positive")
    return stock - sp500 >= outperformance

### 5. prediction.py
def data_string_to_float(number_string):
    """
    The result of our regex search is a number stored as a string, but we need a float.
        - Some of these strings say things like '25M' instead of 25000000.
        - Some have 'N/A' in them.
        - Some are negative (have '-' in front of the numbers).
        - As an artifact of our regex, some values which were meant to be zero are instead '>0'.
    We must process all of these cases accordingly.
    :param number_string: the string output of our regex, which needs to be converted to a float.
    :return: a float representation of the string, taking into account minus sign, unit, etc.
    """
    # Deal with zeroes and the sign
    if ("N/A" in number_string) or ("NaN" in number_string):
        return "N/A"
    elif number_string == ">0":
        return 0
    elif "B" in number_string:
        return float(number_string.replace("B", "")) * 1000000000
    elif "M" in number_string:
        return float(number_string.replace("M", "")) * 1000000
    elif "K" in number_string:
        return float(number_string.replace("K", "")) * 1000
    else:
        return float(number_string)
