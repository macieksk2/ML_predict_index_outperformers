# GENERAL INFORMATION
Scripts are used to fit a ML (Random Forest) model predicting based on current financial information (Income Statement, Balance Sheet, Cash Flow, Key Ratios) a portfolio of stocks likely outperforming index in the following 12 months.

The performance (in terms of ROC/precision/recall in testing set and in backtesting of the strategy) is highly dependent on different factors: </br>
- Features selected - full set significantly underperforms preselected intuitive variables (revenue, ROE, cash on hand, debt-to-equity, Market Cap, liabilities etc)</br>
- Prunning, changing maximum depth and other hyperparameters can lead to lower number of trades but better overall portfolio performance (script 3.0, 3)</br>
- There are still a lot of potential candidates for sensible predictors - lags, political cycles (presidential in US), economic cycles (level of interset rates, expansion of credit)</br>
- Scrapping EDGAR database to get more detailed information for each stock</br>
- Scrapping other indices (Nasdaq, Russell) and other geographies (Europe, Asia) to look for more inefficiencies</br>
- Change model methodology to a more sophisticated algorithm (e.g. neural networks)

# PREREQUISITES
1. Postgre SQL Tool (e.g. pgadmin)
2. Anaconda / Spyder with packages listed in requirements.txt

# INPUT
- Hyperparamters in JSON file:</br>
      "url": "https://www.macrotrends.net/" </br>
      "tickers_filename": "tickers.csv" </br>
      "do_macro_scrap": "False" </br>
      "database": "sp500_20241023_freqQ" </br>
      "do_sql": "True" </br>
      "user": "postgres" </br>
      "password": "password" </br>
      "host": "localhost" </br>
      "port": "5432" </br>
      "YF_START_DATE": "2003-01-01" </br>
      "YF_END_DATE": "2024-10-31" </br>
      "train_END_DATE": "2023-09-30" </br>
      "outperform_idx": 0.1 </br>
      "RandomForest_no_estimators": 100 </br>
      "Backtest_initial_capital": 10000 </br>
      "Backtest_increment": 3000 </br>
      "Backtest_provision": 0.0025 </br>
      "Backtest_tax_rate": 0.19
- A list of tickers for macrotrends and Yahoo Finance (separate list since not each ticker is useful for scrapping)

# OUTPUT
- A database of scrapped financials and prices
- Key statistics (predefined set of variables) for X and Y sets
- Backtesting of the portofolio on the testing set
- Selection of stocks out of sample

# SCRIPTS

### 1 data scrapping.py

1. Scrap macrotrends.com (Income Statement, Balance Sheet, Cash Flow statement, Key ratios)
2. Store historicals relevant for fundamental analysis: 
    - Revenue
    - Net Income
    - Expenses
    - Capex
    - Depr/Amort
    - FCF
    - P/E, P/B, P/S, 
    - Cash Flow Statement
      
3. Visualize historicals
4. Scrap historical stock prices from Yahoo Finance
5. Scrap historical SP500 prices from Yahoo Finance


### 2 data processing.py
### ! IMPORTANT A list of features to be selected is currently coded in the script (a full set of variables led to significantly lower performance)
1. QoQ, YoYpc transform
2. Define YoY returns
3. Create feature variables
4. Calculate MA of prices
5. Split in-sample piece of data (with available future 12m return) and out-of-sample part; store in csvs

 ### 3 optimal features.py

1. Start with all features selected in script 2.
2. Fit Random Forest, test on the training set, calculate feature importance
3. Reduce iteratively the feature set with the least important feature, recalibrate the model, check ROC
4. Pick the best set of features to be utilized in the following 4. backtesting.py script

### 4 backtesting.py

1. Split train/test sets, either randomly or chronologically
2. Fit RF algorithm based on selected features in prev script
3. Calculate Accuracy/Precision, confusion matrix, ROC/AUC curve
4. Print out select stocks in each quarter
5. Run full backtesting of a portfolio run according to the model vs a simple strategy of buying index ETF

### 5 prediction.py

1. Create first training dataset
2. Based on model calibration, select stocks in the latest quarters
