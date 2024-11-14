"""
data scrapping.py

1. Scrap macrotrends.com 
2. Store historicals relevant for fundamental analysis: 
    - Revenue
    - Net Income
    - Expenses
    - Capex
    - Depr/Amort
    - FCF
    - P/E, P/B, P/S, 
    - Casf Flow Statement
    - Debt, D/E
    - No shares outstanding
    - Dividends
    - RoE, RoA
3. Visualize historicals
4. Scrap historical stock prices from Yahoo Finance
5. Scrap historical SP500 prices from Yahoo Finance

"""
### PACKAGES
# Scraping
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup as bs
import yfinance as yf
import pandas_datareader.data as web
from pandas_datareader import data as pdr
import requests
from pandas.api.types import CategoricalDtype

# SQL
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import psycopg2.extras

# General
import time
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import requests, re
import json

# Utils functions
import utils
from utils import retrieve_data, repl_sp_signs, def_headers, execute_values, generate_financial_plots

def macrotrends_scrap():
    """
    1. Retrieve input from JSON and csv (tickers)
    2. Scrap macrotrends
    3. Plot relevant features
    4. Store either in SQL base or csv
    """
    ###############################################################################################################################################################
    # Import input details from JSON file
    ###############################################################################################################################################################
    # Opening JSON file
    f = open('INPUT\input_scrap.json')
    # returns JSON object as a dictionary
    data = json.load(f)
    # Retrieve input details
    url_             = data['url']
    tickers_filename = data['tickers_filename']
    database_name    = data['database']
    do_sql           = data['do_sql']
    user             = data['user']
    password         = data['password']
    host             = data['host']
    port             = data['port']
    # Pull tickers csv
    tickers = pd.read_csv('INPUT/' + tickers_filename)
    z = tickers['sp500_tickers']
    # Start SQL engine
    # Establishing connection
    conn = psycopg2.connect(
        database = database_name,
        user     = user,
        password = password,
        host     = host,
        port     = port
    )
    ###############################################################################################################################################################
    ### Scrap macrotrends
    ### Retrieve the HTML
    ### Scraping financial data from MacroTrends website (Income Statement, Balance Sheet, Cash Flow Statement, Ratios)
    ### Frequency: Quarterly
    ###############################################################################################################################################################
    # getting correct urls for each ticker
    for i in z:
        a=i
        print(a)
        url = url_
        # Start the session
        driver = webdriver.Chrome() 
        # Go to macrotrends website
        driver.get(url)
        # Find textox to search for companies
        box = driver.find_element(By.CSS_SELECTOR, ".js-typeahead")
        # Type in company name
        box.send_keys(a)
        time.sleep(1)
        # Go down the list by one and move to this link
        box.send_keys(Keys.DOWN, Keys.RETURN)
        time.sleep(2)
        # Save the new link aress
        geturl = driver.current_url
        time.sleep(2)
        # Check if the ticker is available in MacroTrends
        if "stocks" in geturl:
            # Split URL by /
            geturlsp = geturl.split("/", 10)
            # Create new URL for charts
            geturlf = url+"stocks/charts/" + geturlsp[5] + "/" + geturlsp[6] + "/"
            # Check if the data in the ticker is available
            fsurl = geturlf + "income-statement?freq=Q"
            driver.get(fsurl)
            if driver.find_elements(By.CSS_SELECTOR, "div.jqx-grid-column-header:nth-child(1) > div:nth-child(1) > div:nth-child(1) > span:nth-child(1)"):
                # Financial-statements
                fsb = retrieve_data(driver, url = geturlf, item = "financial-statements")
                # Balance sheet statement
                bsb = retrieve_data(driver, url = geturlf, item = "balance-sheet")
                # Cash flow statement
                cfb = retrieve_data(driver, url = geturlf, item = "cash-flow-statement")
                # Financial ratios
                frb = retrieve_data(driver, url = geturlf, item = "financial-ratios")
                driver.quit()
                # Transform data
                raw_data = [fsb, bsb, cfb, frb]
                output = []
                for item in raw_data:
                    # Remove blanks from variables, split variables into lists
                    # Do not remove dots in the financial ratios
                    item = repl_sp_signs(item)
                    item.columns = item.columns.str.replace(" ", "")
                    out = item
                    output.append(out)
                # Concatenate whole data, remove duplicates
                fsconcat = output[0]
                fsdados = fsconcat.drop_duplicates()
                bsconcat = output[1]
                bsdados = bsconcat.drop_duplicates()
                cfconcat = output[2]
                cfdados = cfconcat.drop_duplicates()
                frconcat = output[3]
                frdados = frconcat.drop_duplicates()
                # Creating final dataframe
                ca = fsdados.merge(bsdados, left_index=True, right_index=True)
                cb = ca.merge(cfdados, left_index=True, right_index=True)
                complete = cb.merge(frdados, left_index=True, right_index=True)  
                # Replace all comas with dots
                complete = complete.replace(",", ".")
                complete.columns = complete.columns.str.replace(',', '.')
                # Sort by date
                complete.sort_index(inplace=True)
                # Convert all columns of DataFrame
                complete = complete.apply(pd.to_numeric) 
                ###############################################################################################################################################################
                # Generate plots
                ###############################################################################################################################################################
                generate_financial_plots(complete, a)
                # Creating folder for data and images
                if not os.path.exists("STOCKUS/" + a):
                    os.makedirs("STOCKUS/" + a)
                plt.savefig("STOCKUS/" + a + "/" + a + "data.png")
                # Remove duplicate rows
                complete = complete[~complete.index.duplicated(keep = 'last')]
                complete.to_excel(os.path.join("STOCKUS/" + a, a + ".xlsx"), sheet_name = a)
                ###############################################################################################################################################################
                # Insert into database
                ###############################################################################################################################################################
                if do_sql:
                    # Store in SQL database
                    # Remove duplicate rows
                    # Remove special signs
                    complete.columns = complete.columns.str.replace("&", "And").str.replace("-","").str.replace("/","").str.replace("(","").str.replace(")","")
                    # Create a column storing index 
                    complete["Date"] = complete.index
                    # Create column storing Ticker + Date
                    complete["Date_Ticker"] = [str(a) + "-" + str(x) for x in complete["Date"]]
                    df_columns_ = list(complete)
                    # Insert a data type
                    df_columns = [str(x) + " varchar(40)" for x in df_columns_[:-2]]
                    df_columns.append(str(df_columns_[-2]) + " date")
                    df_columns.append(str(df_columns_[-1]) + " char(40)")
                    # Create (col1,col2,...)
                    columns = ",".join(df_columns).lower()
                    columns = columns.replace(".","")
                    # Creating a cursor
                    cursor = conn.cursor()
                    
                    # Check if the table already exists
                    # Rename columns with dots
                    complete = complete.rename(columns={'Property.Plant.AndEquipment': 'PropertyPlantAndEquipment',
                                                        'NetChangeInProperty.Plant.AndEquipment': 'NetChangeInPropertyPlantAndEquipment'})
                    sql = "CREATE TABLE IF NOT EXISTS " + database_name + "(" + columns + ");" 
                    sql = eval('''sql''')
                    try:
                        cursor.execute(sql)
                        conn.commit()
                    except:
                        conn.rollback()
                    execute_values(conn, complete, database_name)

                # Confirmation message for ticker that exists and have data
                print("SUCCESS")
            # Error message for ticker that exists but have no data
            else:
                driver.quit()
                print("EMPTY TICKER")
        # Error message for ticker that doesn't exist
        else:
                print("INVALID TICKER")
    # Final message
    print("FINISHED")

def yf_scrap_stock_price():
    """
    Scrap from yahoo finance and store a dataset containing the selected stock prices (JSON file)
    :returns: stock_prices.csv
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
    ticker_list = tickers['sp500_yf_tickers'].values
    ticker_list = ticker_list.tolist()
    # Get all Adjusted Close prices for all the tickers in the list between START_DATE and END_DATE
    print(ticker_list)
    all_data = pdr.get_data_yahoo(ticker_list, start, end)
    stock_data = all_data["Adj Close"]

    # Remove any columns that hold no data, and print their tickers.
    stock_data.dropna(how="all", axis=1, inplace=True)
    missing_tickers = [ticker for ticker in ticker_list if ticker.upper() not in stock_data.columns]
    print(f"{len(missing_tickers)} tickers are missing: \n {missing_tickers} ")
    # If there are only some missing datapoints, forward fill.
    stock_data.ffill(inplace=True)
    stock_data.to_csv("OUTPUT/stock_prices.csv")

def yf_scrap_sp500_price():
    """
    Scrap from yahoo finance and store a dataset with sp500 prices
    :returns: sp500_index.csv
    """
    # Opening JSON file
    f = open('INPUT\input_scrap.json')
    # returns JSON object as a dictionary
    data = json.load(f)
    
    # Retrieve input details
    start = data['YF_START_DATE']
    end   = data['YF_END_DATE']
    
    index_data = pdr.get_data_yahoo("SPY", start=start, end=end)
    index_data.to_csv("OUTPUT/sp500_index.csv")

if __name__ == "__main__":
    # Opening JSON file
    f = open('INPUT\input_scrap.json')
    # returns JSON object as a dictionary
    data = json.load(f)
    do_macro_scrap   = data['do_macro_scrap']
    if do_macro_scrap:
        macrotrends_scrap()
    # https://stackoverflow.com/questions/74883892/pandas-datareader-no-longer-working-with-yahoo-finance
    yf.pdr_override()
    yf_scrap_stock_price()
    yf_scrap_sp500_price()
