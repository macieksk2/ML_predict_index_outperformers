# -*- coding: utf-8 -*-
"""
data processing.py

1. QoQ, YoYpc transform
2. Define YoY returns
3. Create feature variables
4. Calculate MA of prices
5. Split in-sample piece of data (with available future 12m return) and out-of-sample part; store in csvs
"""
### PACKAGES
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from   matplotlib.backends.backend_pdf import PdfPages
import math
import os
import json
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.api import OLS
import statsmodels.api as sm

import psycopg2 as pg
from   psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import psycopg2.extras
# Utils functions
import utils
from utils import moving_average, fill_nas

### INPUT 
print("***************************************")
print("0. Input")
print("***************************************")
def data_process():
    # Database settings
    # Opening JSON file
    f = open('INPUT\input_scrap.json')
    # returns JSON object as a dictionary
    data = json.load(f)
    
    # Retrieve input details
    url_             = data['url']
    tickers_filename = data['tickers_filename']
    database_name_   = data['database']
    table_name_      = data['database']
    do_from_sql      = data['do_sql']
    user             = data['user']
    password_        = data['password']
    host             = data['host']
    port             = data['port']
    
    outperform_idx   = data['outperform_idx']
    
    # Pull tickers csv
    tickers          = pd.read_csv('INPUT/' + tickers_filename)
    sp500_tickers    = tickers['sp500_tickers']
    sp500_yf_tickers = tickers['sp500_yf_tickers']
    
    ## Import share prices
    close_  = np.array([])
    date_   = np.array([])
    ticker_ = np.array([])
    k = 0
    print("***************************************")
    print("1. Start downloading stock prices")
    print("***************************************")
    # Take the prices from csv from script 1
    # In case of no data, put in NAs
    # Put into numpy array to improve efficiency
    stock_prices = pd.read_csv('OUTPUT\stock_prices.csv')
    stock_prices.index = stock_prices['Date']
    for i in sp500_yf_tickers:
        print(i)
        try:
            close_  = np.concatenate([close_, np.array(stock_prices[i])])
            date_   = np.concatenate([date_, np.array(stock_prices[i].index)])
            ticker_ = np.concatenate([ticker_, np.repeat(str(i),  len(stock_prices[i].index))])
        except:
            close_  = np.concatenate([close_, np.repeat(0, 1)])
            date_   = np.concatenate([date_, np.repeat('1900-01-01', 1)])
            ticker_ = np.concatenate([ticker_, np.repeat(str(i), 1)])                            
        k += 1
    
    out_ticker_ = pd.DataFrame(columns = ["Date", "Ticker", "Close"])
    out_ticker_['Date']    = date_
    out_ticker_['Ticker']  = ticker_
    out_ticker_['Close']   = close_
    # Add SPY index historical values
    sp500_index = pd.read_csv('OUTPUT\sp500_index.csv')
    sp500_index.index = sp500_index['Date']
    
    indexDf_to_concat = pd.DataFrame()
    indexDf_to_concat['Date'] = sp500_index.index
    indexDf_to_concat.index = sp500_index.index
    # Take adj close price
    indexDf_to_concat['Close'] = sp500_index['Adj Close']
    indexDf_to_concat.pop('Date')
    indexDf_to_concat.columns = ['SP500']
    
    # Create a macrotrends to YF mapping
    tickers_mapping = pd.DataFrame(columns = ['Ticker', 'YF'])
    tickers_mapping['Ticker'] = sp500_tickers
    tickers_mapping['YF'] = sp500_yf_tickers
    
    print("***************************************")
    print('2. Pull financials from SQL, merge with YF prices')
    print("***************************************")
    if do_from_sql:
        # Connect with Postgre SQL database
        # Inputs from JSON
        conn  = pg.connect("dbname='" + database_name_ + 
                           "' user=" + user + 
                           " host=" + host + 
                           " port=" + port + 
                           " password='" + password_ + "'")
        cur = conn.cursor()
        # Query to get column names
        cur.execute("""SELECT * FROM %s LIMIT 0""" % database_name_)
        colnames = [desc[0] for desc in cur.description]
        # Query to get values
        cur.execute("""SELECT * FROM %s""" % database_name_)
        query_results = pd.DataFrame(cur.fetchall())
        query_results.columns = colnames
        # Get ticker, year and quarter from date_ticker column
        query_results['Ticker']  = [x[0] for x in query_results['date_ticker'].str.split('-')]
        query_results['Year']    = [int(x[1]) for x in query_results['date_ticker'].str.split('-')]
        query_results['Quarter'] = [str(int(x[1])) + 'Q' + str(math.ceil(int(x[2]) / 3)) for x in query_results['date_ticker'].str.split('-')]
        query_results = query_results.drop_duplicates()
        # Keep only the last price in each quarter for each ticker
        out_ticker_['Date']    = pd.to_datetime(out_ticker_['Date'])
        out_ticker_['Year']    = [x.year for x in out_ticker_['Date']]
        out_ticker_['Quarter'] = [str(x.year) + 'Q' + str(math.ceil(x.month / 3)) for x in out_ticker_['Date']]
        out_ticker_last        = out_ticker_.drop_duplicates(['Ticker', 'Year', 'Quarter'], keep='last')
        # Merge with index hisotiricals
        indexDf_to_concat.index = pd.to_datetime(indexDf_to_concat.index)
        out_ticker_last = pd.merge(out_ticker_last, indexDf_to_concat, how = 'left', on = ['Date'])
        # Map with YF ticker and remove macrotrends ticker
        query_results = query_results.merge(tickers_mapping, on='Ticker', how='left')
        query_results.pop('Ticker')
        query_results['Ticker'] = query_results['YF']
        query_results.pop('YF')
        # Merge database financials with YE stock price
        q_res_stk_prc = pd.merge(query_results, 
                                 out_ticker_last[['Ticker', 'Year', 'Quarter', 'Close', 'SP500']], 
                                 left_on  = ['Ticker', 'Year', 'Quarter'], 
                                 right_on = ['Ticker', 'Year', 'Quarter'], 
                                 how='left')
        # Store the updated table in SQL database
        # Close the cursor and connection to so the server can allocate
        # Bandwidth to other requests
        cur.close()
        conn.close()
    
    print("***************************************")
    print('3. Y definition / storing')
    print("***************************************")
    # Calculate forward YoYpc change in stock and index price (to check the impact of current fundamental features on future value of the predictor)
    q_res_stk_prc['Close_YoYpc'] = q_res_stk_prc.groupby('Ticker')['Close'].pct_change(-4)
    # Transform from Y(t) / Y(t+4) - 1 to Y(t+4) / Y(t) - 1
    q_res_stk_prc['Close_YoYpc'] = -q_res_stk_prc['Close_YoYpc'] / (q_res_stk_prc['Close_YoYpc'] + 1)
    # Calculate forward YoYpc change in stock and index price (to check the impact of current fundamental features on future value of the predictor)
    q_res_stk_prc['SP500_YoYpc'] = q_res_stk_prc.groupby('Ticker')['SP500'].pct_change(-4)
    # Transform from Y(t) / Y(t+4) - 1 to Y(t+4) / Y(t) - 1
    q_res_stk_prc['SP500_YoYpc'] = -q_res_stk_prc['SP500_YoYpc'] / (q_res_stk_prc['SP500_YoYpc'] + 1)
    # Create a boolean variable, 1 if stock outperformed index by n% YoY
    q_res_stk_prc['CloseOutIDX'] = [1 if q_res_stk_prc['Close_YoYpc'][x] - q_res_stk_prc['SP500_YoYpc'][x] > outperform_idx 
                                                else 0 for x in range(len(q_res_stk_prc))]
    
    q_res_stk_prc = q_res_stk_prc.sort_values('date')
    
    # Store in csv file
    q_res_stk_prc.to_csv('OUTPUT\macrotrends_yf_stockprices_database_full_index.csv')

def parsing_keystats():
    database = pd.read_csv('OUTPUT/macrotrends_yf_stockprices_database_full_index.csv')
    
    # Database settings
    # Opening JSON file
    f = open('INPUT\input_scrap.json')
    # returns JSON object as a dictionary
    data = json.load(f)
    
    # Retrieve input details
    END_DATE = data['train_END_DATE']
    
    # Check first main variables
    # In case all values are NAs, insert a predefined ratio or other feauture
    # In case some are NAs, fill forward
    for i in np.unique(database['Ticker']):
        print(i)
        # TO BE REVIEWED IN CASE MORE VARIABLES TO BE INCLUDED IN THE MODEL (CURRENTLY MORE FEATURES LEADS TO LOWER PERFORMANCE)
        # for c in database.columns:
        #     # For most features either fill forward per ticker or set to 0
        #     if c not in ['date', 'date_ticker', 'Year', 'Quarter', 'Ticker', 'Close', 'SP500','Close_YoYpc', 'SP500_YoYpc', 'CloseOutIDX',
        #                  'epsearningspershare', 'ebitda', 'grossmargin', 'operatingmargin', 'currentratio', 'longtermdebt', 'debtequityratio', 'researchanddevelopmentexpenses']:
        #         database.loc[database['Ticker'] == i, c]     = fill_nas(database,    c, i, is_var_handle = False, handle_type = 'ratio', handle_ratio_nom = "",  handle_ratio_denom = "")

        database.loc[database['Ticker'] == i, 'epsearningspershare'] = fill_nas(database,    'epsearningspershare',    i, is_var_handle = True, handle_type = 'ratio', handle_ratio_nom = "netincome",          handle_ratio_denom = "sharesoutstanding")
        database.loc[database['Ticker'] == i, 'ebitda']              = fill_nas(database,    'ebitda',                 i, is_var_handle = True, handle_type = 'value', handle_ratio_nom = "netincome",          handle_ratio_denom = "")
        database.loc[database['Ticker'] == i, 'grossmargin']         = fill_nas(database,    'grossmargin',            i, is_var_handle = True, handle_type = 'ratio', handle_ratio_nom = "grossprofit",        handle_ratio_denom = "revenue")
        database.loc[database['Ticker'] == i, 'operatingmargin']     = fill_nas(database,    'operatingmargin',        i, is_var_handle = True, handle_type = 'ratio', handle_ratio_nom = "operatingincome",    handle_ratio_denom = "revenue")
        database.loc[database['Ticker'] == i, 'currentratio']        = fill_nas(database,    'currentratio',           i, is_var_handle = True, handle_type = 'ratio', handle_ratio_nom = "totalcurrentassets", handle_ratio_denom = "totalcurrentliabilities")
        database.loc[database['Ticker'] == i, 'longtermdebt']        = fill_nas(database,    'longtermdebt',           i, is_var_handle = True, handle_type = 'value', handle_ratio_nom = "totalliabilities",   handle_ratio_denom = "")
        database.loc[database['Ticker'] == i, 'debtequityratio']     = fill_nas(database,    'debtequityratio',        i, is_var_handle = True, handle_type = 'ratio', handle_ratio_nom = "longtermdebt",       handle_ratio_denom = "shareholderequity")
        
    # Calculate QoQ / YoY pc changes
    vars_ = database.columns
    vars_ = vars_.drop(['date', 'date_ticker', 'Ticker', 'Year', 'Quarter', 'Close', 'Close_YoYpc', 'SP500', 'SP500_YoYpc'])
    for c in vars_:
        database[c] = database[c].astype(float)
        # TO BE REVIEWED IN CASE MORE VARIABLES TO BE INCLUDED IN THE MODEL (CURRENTLY MORE FEATURES LEADS TO LOWER PERFORMANCE)
        database[c + '_QoQpc'] = database.groupby('Ticker')[c].pct_change()#.fillna(0)
        database[c + '_YoYpc'] = database.groupby('Ticker')[c].pct_change(4)#.fillna(0)
        
    # TO BE REVIEWED IN CASE MORE VARIABLES TO BE INCLUDED IN THE MODEL (CURRENTLY MORE FEATURES LEADS TO LOWER PERFORMANCE)
    # Create a summary of NAs per feature    
    # Remove features with NAs exceeding 10% of the records (slightly above no of records lost by YoYpc transform)
    # nans_ = pd.DataFrame(columns = ['name', 'no_nans'])
    # nans_['name'] = database.columns.tolist()
    # nans_['no_nans'] = [database[x].isna().sum() for x in database.columns]
    # print(nans_)
    # col_select = nans_[nans_.no_nans <= 0.1 * len(database)]['name'] 
    # database = database[col_select.tolist()]
    
    # Market Cap = No shares outstanding * share price * 1mln
    database['Price']                    = database['Close']
    database['Market Cap']               = database['sharesoutstanding'] * database['Price'] * 10**(6)
    # "Enterprise Value = Cap + Debt - Cash
    database['Enterprise Value']         = database['Market Cap'] + database['totalliabilities'] - database['cashonhand'] 
    # "Trailing P/E = Share Price / sum(EPS from last 4 quarters)
    database['EPS_roll_sum']             = database.groupby('Ticker')['epsearningspershare'].transform(lambda x: x.rolling(4, 4).sum())
    database['Trailing P/E']             = database['Price'] / database['EPS_roll_sum']
    # "Price/Sales
    database['Revenue_roll_sum']         = database.groupby('Ticker')['revenue'].transform(lambda x: x.rolling(4, 4).sum())
    database['Price/Sales']              = database['Market Cap'] / database['Revenue_roll_sum'] / 10**(6)
    # Price/Book
    database['Price/Book']               = database['Price'] / database['bookvaluepershare']
    # Enterprise Value/Revenue
    database['Enterprise Value/Revenue'] = database['Enterprise Value'] / database['Revenue_roll_sum'] / 10**(6)
    # Enterprise Value/EBITDA
    
    database['EBITDA_roll_sum']          = database.groupby('Ticker')['ebitda'].transform(lambda x: x.rolling(4, 4).sum())
    database['Enterprise Value/EBITDA']  = database['Enterprise Value'] / database['EBITDA_roll_sum'] / 10**(6)
    
    # "Revenue Per Share",
    database['Revenue Per Share']        = database['revenue'] / database['sharesoutstanding']
    
    # Import historical daily stock prices and SP500
    hst_d_stk_prc = pd.read_csv('OUTPUT/stock_prices.csv')
    hst_d_idx_prc = pd.read_csv('OUTPUT/sp500_index.csv')
    
    hst_d_stk_prc.index = hst_d_stk_prc['Date']
    hst_d_idx_prc.index = hst_d_idx_prc['Date']
    
    hst_d_stk_prc.pop('Date')
    hst_d_idx_prc.pop('Date')
    # Algorithm based on numpy array for performance sake
    for comp in np.unique(hst_d_stk_prc.columns):
        arr_temp  = np.array(hst_d_stk_prc[comp])
        _15D_arr  = moving_average(arr_temp, 15)
        _50D_arr  = moving_average(arr_temp, 50)
        _200D_arr = moving_average(arr_temp, 200)
        hst_d_stk_prc[str('15DMA_' + comp)]  = _15D_arr
        hst_d_stk_prc[str('50DMA_' + comp)]  = _50D_arr
        hst_d_stk_prc[str('200DMA_' + comp)] = _200D_arr
    
    hst_d_stk_prc['Year']    = [str(datetime.strptime(x, '%Y-%m-%d').date().year) for x in hst_d_stk_prc.index]
    hst_d_stk_prc['Quarter'] = [str(datetime.strptime(x, '%Y-%m-%d').date().year) + 'Q' + str(math.ceil(datetime.strptime(x, '%Y-%m-%d').date().month / 3)) for x in hst_d_stk_prc.index]
    
    hst_d_idx_prc['Year']    = [str(datetime.strptime(x, '%Y-%m-%d').date().year) for x in hst_d_idx_prc.index]
    hst_d_idx_prc['Quarter'] = [str(datetime.strptime(x, '%Y-%m-%d').date().year) + 'Q' + str(math.ceil(datetime.strptime(x, '%Y-%m-%d').date().month / 3)) for x in hst_d_idx_prc.index]
    
    hst_d_stk_prc = hst_d_stk_prc.drop_duplicates('Quarter', keep = 'last')
    hst_d_idx_prc = hst_d_idx_prc.drop_duplicates('Quarter', keep = 'last')
    
    # Loop through database, insert appropriate values into 50D and 200D MA
    database['15DMA']  = ''
    database['50DMA']  = ''
    database['200DMA'] = ''
    for i in np.unique(database['Ticker']):
        ticker = i
        # Overwrite only if ticker present in YF database
        if ticker in hst_d_stk_prc.columns:
            # In case when more obs in one quarter, keep the first one (on case observed for BBY)
            if len(database.loc[database['Ticker'] == ticker, '50DMA']) > len(hst_d_stk_prc[hst_d_stk_prc['Quarter'].isin(np.unique(database[database['Ticker'] == ticker]['Quarter']))]['50DMA_' + ticker].values):
                database.loc[database['Ticker'] == ticker] = database.loc[database['Ticker'] == ticker][~database.loc[database['Ticker'] == ticker]['Quarter'].duplicated(keep='first')]
            database.loc[database['Ticker'] == ticker, '15DMA']  = hst_d_stk_prc[hst_d_stk_prc['Quarter'].isin(np.unique(database[database['Ticker'] == ticker]['Quarter']))]['15DMA_'  + ticker].values        
            database.loc[database['Ticker'] == ticker, '50DMA']  = hst_d_stk_prc[hst_d_stk_prc['Quarter'].isin(np.unique(database[database['Ticker'] == ticker]['Quarter']))]['50DMA_'  + ticker].values
            database.loc[database['Ticker'] == ticker, '200DMA'] = hst_d_stk_prc[hst_d_stk_prc['Quarter'].isin(np.unique(database[database['Ticker'] == ticker]['Quarter']))]['200DMA_' + ticker].values
    
    # Add variable = 1 if Close is above 15/50/200MA and zero otherwise
    database['close_ab_15DMA']  = np.where(database['Close']  > pd.to_numeric(database['15DMA']),  1, 0)
    database['close_ab_50DMA']  = np.where(database['Close']  > pd.to_numeric(database['50DMA']),  1, 0)
    database['close_ab_200DMA'] = np.where(database['Close']  > pd.to_numeric(database['200DMA']), 1, 0)
    # Add variable = 1 if short-term MA is above long-term and zero otherwise
    database['15DMA_ab_50DMA']  = np.where(database['15DMA'] > pd.to_numeric(database['50DMA']),   1, 0)
    database['50DMA_ab_200DMA'] = np.where(database['50DMA'] > pd.to_numeric(database['200DMA']),  1, 0)
    
    # Graham number (trailing annual EPS)
    # In case of negative EPS, put negative without square root
    # GN = sqrt(15 * 1.5 * EPS * BV per share)
    database['GrahamNumber'] = [np.sqrt(22.5 * database.loc[x, 'EPS_roll_sum'] * database.loc[x, 'bookvaluepershare']) if database.loc[x, 'EPS_roll_sum'] > 0 and database.loc[x, 'bookvaluepershare'] > 0
                                else 0 for x in range(len(database))]
    
    # Check if Graham number is below above and has been calculated
    database['GrahamNumber_to_Price'] = np.where(database['GrahamNumber'] > 0, database['GrahamNumber'] / database['Close'], 0)
    # Define keystats (right now all variables, to be determined if should be preselected earlier)
    keystats = database[['date',
                         'Year',
                         'Quarter',
                         'Ticker',
                         'Close',
                         'Close_YoYpc',
                         'SP500',
                         'SP500_YoYpc',
                         'Market Cap',
                         'Trailing P/E',
                         'Price/Sales',
                         'Price/Book',
                         'Enterprise Value/Revenue',	
                         'Enterprise Value/EBITDA',
                         'grossmargin',
                         'grossmargin_YoYpc',
                         'netprofitmargin',
                         'netprofitmargin_YoYpc',
                         'operatingmargin',
                         'roareturnonassets',
                         'roereturnonequity',
                         'roireturnoninvestment',
                         'Revenue Per Share',
                         'revenue_QoQpc',
                         'revenue_YoYpc',
                         'epsearningspershare',
                         'epsearningspershare_QoQpc',
                         'epsearningspershare_YoYpc',
                         'cashonhand',
                         'cashonhand_YoYpc',
                         'totalliabilities',
                         'totalliabilities_YoYpc',
                         'debtequityratio',
                         'debtequityratio_QoQpc',
                         'debtequityratio_YoYpc',
                         'currentratio',
                         'currentratio_QoQpc',
                         'currentratio_YoYpc',
                         'bookvaluepershare',
                         'bookvaluepershare_QoQpc',
                         'bookvaluepershare_YoYpc',
                         'operatingcashflowpershare',
                         'freecashflowpershare',
                         'operatingcashflowpershare_QoQpc',
                         'freecashflowpershare_QoQpc',
                         'operatingcashflowpershare_YoYpc',
                         'freecashflowpershare_YoYpc',
                         'sharesoutstanding_QoQpc',
                         'sharesoutstanding_YoYpc',
                         'close_ab_15DMA',
                         'close_ab_50DMA',
                         'close_ab_200DMA',
                         '15DMA_ab_50DMA',
                         '50DMA_ab_200DMA',
                         'GrahamNumber_to_Price'
                         ]]
    # Split into train set/oos set (with no information of 12m forward return) and store as csvs
    keystats_train = keystats[keystats['date'] <= END_DATE]
    keystats_test  = keystats[keystats['date'] > END_DATE]
    keystats_train.to_csv('OUTPUT/keystats_new.csv')
    keystats_test.to_csv('OUTPUT/keystats_new_OOS.csv')


if __name__ == "__main__":
    data_process()
    parsing_keystats()