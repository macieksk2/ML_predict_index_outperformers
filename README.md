# GENERAL INFORMATION
Scripts are used to fit a ML (Random Forest) model predicting based on current financial information (Income Statement, Balance Sheet, Cash Flow, Key Ratios) a portfolio of stocks likely outperforming index in the following 12 months.

# PREREQUISITES
1. Postgre SQL Tool (e.g. pgadmin)
2. Anaconda / Spyder with packages listed in requirements.txt

# INPUT
- Hyperparamters in JSON file:
      "url": "https://www.macrotrends.net/",
      "tickers_filename": "tickers.csv",
      "do_macro_scrap": "False",
      "database": "sp500_20241023_freqQ",
      "do_sql": "True",
      "user": "postgres",
      "password": "passwrod",
      "host": "localhost",
      "port": "5432",
      "YF_START_DATE": "2003-01-01",
      "YF_END_DATE": "2024-10-31",
      "train_END_DATE": "2023-09-30",
      "outperform_idx": 0.1,
      "RandomForest_no_estimators": 100,
      "Backtest_initial_capital": 10000,
      "Backtest_increment": 3000,
      "Backtest_provision": 0.0025,
      "Backtest_tax_rate": 0.19

# OUTPUT


# SCRIPTS

I data scrapping.py

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
    -       "url": "https://www.macrotrends.net/",
      "tickers_filename": "tickers.csv",
      "do_macro_scrap": "False",
      "database": "sp500_20241023_freqQ",
      "do_sql": "True",
      "user": "postgres",
      "password": "passwrod",
      "host": "localhost",
      "port": "5432",
      "YF_START_DATE": "2003-01-01",
      "YF_END_DATE": "2024-10-31",
      "train_END_DATE": "2023-09-30",
      "outperform_idx": 0.1,
      "RandomForest_no_estimators": 100,
      "Backtest_initial_capital": 10000,
      "Backtest_increment": 3000,
      "Backtest_provision": 0.0025,
      "Backtest_tax_rate": 0.19Debt, D/E
    - No shares outstanding
    - Dividends
    - RoE, RoA
3. Visualize historicals
4. Scrap historical stock prices from Yahoo Finance
5. Scrap historical SP500 prices from Yahoo Finance


II data processing.py

1. QoQ, YoYpc transform
2. Define YoY returns
3. Create feature variables
4. Calculate MA of prices
5. Split in-sample piece of data (with available future 12m return) and out-of-sample part; store in csvs

III optimal features.py

1. Start with all features selected in script 2.
2. Fit Random Forest, test on the training set, calculate feature importance
3. Reduce iteratively the feature set with the least important feature, recalibrate the model, check ROC
4. Pick the best set of features to be utilized in the following 4. backtesting.py script

IV backtesting.py

1. Split train/test sets
2. Fit RF algorithm based on selected features in prev script
3. Calculate Accuracy/Precision, confusion matrix, ROC/AUC curve
4. Print out select stocks in each quarter
5. Run full backtesting of a portfolio run according to the model vs a simple strategy of buying index ETF

V prediction.py

1. Create first training dataset
2. Based on model calibration, select stocks in the latest quarters
