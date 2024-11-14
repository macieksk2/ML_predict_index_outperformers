PREREQUISITS:
1. Postgre SQL Tool (e.g. pgadmin)
2. Anaconda / Spyder with packages listed in requirements.txt


SCRIPTS:

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
    - Debt, D/E
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
2. Fit Random Forrest, test on the training set, calculate feature importance
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


