# PREREQUISITS:
1. Postgre SQL Tool (e.g. pgadmin)
2. Anaconda / Spyder with packages listed in requirements.txt

# FOLDER STUCTURE

## INPUT
1. input_scrap.json - JSON file with parameters used in different stages of the process:
      Used in data scrapping.py / data processing.py
      a. "url": "https://www.macrotrends.net/" - URL of macrotrends, source of historical financial information on stocks scrapped 
      b. "tickers_filename": "tickers.csv"     - name of a file in INPUT folder with a mapping of tickers between macrotrends (sometimes a stock ticker is not enough to select a particual stock by the user) to Yahoo Finance tickers
      c. "do_macro_scrap": "False"             - a flag if set to True activates a function scrapping information from macrotrends. If a database already exists locally, can be set to False
      d. "database": "sp500_20241023_freqQ"    - a name of database in Postgre SQL Tool (needs to be created before scraping)
      e. "do_sql": "True"                      - a flag if set to True stores the scrapped information in SQL database. When set to False, dumps it to csv file.
      f. "user": "postgres"                    - a user name in Postgre database.
      g. "password": "password"                - a password in Postgre database.
      h. "host": "localhost"                   - a localhost in Postgre database.
      i. "port": "5432"                        - a port in Postgre database.
      j. "YF_START_DATE": "2003-01-01"         - first date used to scrap stock prices from Yahoo Finance
      k. "YF_END_DATE": "2024-10-31"           - end date used to scrap stock prices from Yahoo Finance
      l. "train_END_DATE": "2023-09-30"        - last date used to train the ML model
      Used in optimal features.py
      m. "outperform_idx": 0.1                 - a number of percent by which the stock needs to outperform index in the next 12 months (right now SP500)
      n. "RandomForest_no_estimators": 100     - a number of trees in Random Forest
      Used in backtesting.py
      o. "Backtest_initial_capital": 10000     - backtesting, starting capital
      p. "Backtest_increment": 3000            - backtesting, new investment each quarter
      q. "Backtest_provision": 0.0025          - backtesting, provision for buy/sell transaction
      r. "Backtest_tax_rate": 0.19             - backtesting, capital gains tax
## OUTPUT

## STOCKUS

# SCRIPTS:

## 1. data scrapping.py

1. Scrap macrotrends.com (Income Statement, Balance Sheet, Cash Flow statement, Key ratios)
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

## 2. data processing.py

1. QoQ, YoYpc transform
2. Define YoY returns
3. Create feature variables
4. Calculate MA of prices
5. Split in-sample piece of data (with available future 12m return) and out-of-sample part; store in csvs

## 3. optimal features.py

1. Start with all features selected in script 2.
2. Fit Random Forest, test on the training set, calculate feature importance
3. Reduce iteratively the feature set with the least important feature, recalibrate the model, check ROC
4. Pick the best set of features to be utilized in the following 4. backtesting.py script

## 4. backtesting.py

1. Split train/test sets
2. Fit RF algorithm based on selected features in prev script
3. Calculate Accuracy/Precision, confusion matrix, ROC/AUC curve
4. Print out select stocks in each quarter
5. Run full backtesting of a portfolio run according to the model vs a simple strategy of buying index ETF

## 5. prediction.py

1. Create first training dataset
2. Based on model calibration, select stocks in the latest quarters


