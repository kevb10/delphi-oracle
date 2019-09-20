# Quandl for financial analysis, pandas and numpy for data manipulation
# fbprophet for additive models, #pytrends for Google trend data
import pandas as pd
import numpy as np
import fbprophet
import pytrends
import json
import pickle
import os
from pytrends.request import TrendReq


# Class for analyzing and (attempting) to predict future prices
# Contains a number of visualizations and analysis methods
class Oracle():
    
    # Initialization requires a ticker symbol
    def __init__(self, ticker, sesh):
        # Enforce capitalization
        ticker = ticker.strip().upper()
        self.alpha_vantage_api_key = 'YHST3SHIZZX5VRUA'

        self.session = sesh 

        # Initialize Alpha Vantage
        self.macd_data = self.get_macd(ticker)
        self.rsi_data = self.get_stoch_rsi(ticker)

        # Symbol is used for labeling plots
        self.symbol = ticker

        # Retrieval the financial data
        try:
            stock = self.get_stock(ticker)
        except Exception as e:
            print('Error Retrieving Data.')
            print(e)
            return
        
        # Columns required for prophet
        stock['ds'] = stock['timestamp']

        if ('Adj. Close' not in stock.columns):
            stock['Adj. Close'] = stock['close']
            stock['Adj. Open'] = stock['open']
        
        stock['y'] = stock['Adj. Close']
        stock['Daily Change'] = stock['Adj. Close'] - stock['Adj. Open']
        
        # Data assigned as class attribute
        self.stock = stock.copy()

        self.stock = self.remove_weekends(self.stock)
        
        # Minimum and maximum date in range
        self.min_date = min(stock['timestamp'])
        self.max_date = max(stock['timestamp'])
        
        # Find max and min prices and dates on which they occurred
        self.max_price = np.max(self.stock['y'])
        self.min_price = np.min(self.stock['y'])
        
        self.min_price_date = self.stock[self.stock['y'] == self.min_price]['timestamp']
        self.min_price_date = self.min_price_date[self.min_price_date.index[0]]
        self.max_price_date = self.stock[self.stock['y'] == self.max_price]['timestamp']
        self.max_price_date = self.max_price_date[self.max_price_date.index[0]]
        
        # The starting price (starting with the opening price)
        self.starting_price = float(self.stock.ix[0, 'Adj. Open'])
        
        # The most recent price
        self.most_recent_price = float(self.stock.ix[len(self.stock) - 1, 'y'])

        # Whether or not to round dates
        self.round_dates = True
        
        # Number of years of data to train on
        self.training_years = 3

        # Prophet parameters
        # Default prior from library
        self.changepoint_prior_scale = 0.05 
        self.weekly_seasonality = True
        self.daily_seasonality = False
        self.monthly_seasonality = True
        self.yearly_seasonality = True
        self.changepoints = None
 

    def close(self):
        self.session.close()

    """
    MACD daily, close
    """
    def get_macd(self, ticker):
        url = ('https://www.alphavantage.co/query?function=MACD&symbol=' + 
        ticker + 
        '&interval=daily&series_type=close&apikey=' + 
        self.alpha_vantage_api_key)

        return json.loads(self.session.get(url, verify=False).content)

    """
    Stoch RSI daily, close, period 200, fast k 14, fast d, 14
    """
    def get_stoch_rsi(self, ticker):
        url = ('https://www.alphavantage.co/query?function=STOCHRSI&symbol=' + 
        ticker + 
        '&interval=daily&time_period=200&series_type=close&fastkperiod=14&fastdperiod=14&fastdmatype=1&apikey=' + 
        self.alpha_vantage_api_key)

        return json.loads(self.session.get(url, verify=False).content)

    """
    Fetch stock data
    """
    def get_stock(self, ticker):
        url = ('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' +
        ticker + 
        '&apikey=' + self.alpha_vantage_api_key + 
        '&outputsize=full&datatype=csv')

        stock_data = pd.read_csv(url, parse_dates=['timestamp'])
        stock_data['index'] = stock_data['timestamp']
        stock_data = stock_data.set_index('index')

        return stock_data
    
    """
    Make sure start and end dates are in the range and can be
    converted to pandas datetimes. Returns dates in the correct format
    """
    def handle_dates(self, start_date, end_date):
         # Default start and end date are the beginning and end of data
        if start_date is None:
            start_date = self.min_date
        if end_date is None:
            end_date = self.max_date
        
        try:
            # Convert to pandas datetime for indexing dataframe
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
        
        except Exception as e:
            print('I cant handle this format bruh lol')
            print(e)
            return
                
        return start_date, end_date
        
    """
    Return the dataframe trimmed to the specified range.
    """
    def make_df(self, start_date, end_date, df=None):
        # Default is to use the object stock data
        if not df:
            df = self.stock.copy()
        
        start_date, end_date = self.handle_dates(start_date, end_date)
        
        # keep track of whether the start and end dates are in the data
        start_in = True
        end_in = True

        # If user wants to round dates (default behavior)
        if self.round_dates:
            # Record if start and end date are in df
            if (start_date not in list(df['timestamp'])):
                start_in = False
            if (end_date not in list(df['timestamp'])):
                end_in = False

            # If both are not in dataframe, round both
            if (not end_in) & (not start_in):
                trim_df = df[(df['timestamp'] >= start_date.date()) & 
                             (df['timestamp'] <= end_date.date())]
            
            else:
                # If both are in dataframe, round neither
                if (end_in) & (start_in):
                    trim_df = df[(df['timestamp'] >= start_date.date()) & 
                                 (df['timestamp'] <= end_date.date())]
                else:
                    # If only start is missing, round start
                    if (not start_in):
                        trim_df = df[(df['timestamp'] > start_date.date()) & 
                                     (df['timestamp'] <= end_date.date())]
                    # If only end is imssing round end
                    elif (not end_in):
                        trim_df = df[(df['timestamp'] >= start_date.date()) & 
                                     (df['timestamp'] < end_date.date())]
            
        return trim_df
    
    # Remove weekends from a dataframe
    def remove_weekends(self, dataframe):
        
        # Reset index to use ix
        dataframe = dataframe.reset_index(drop=True)
        
        weekends = []
        
        # Find all of the weekends
        for i, date in enumerate(dataframe['ds']):
            if (date.weekday()) == 5 | (date.weekday() == 6):
                weekends.append(i)
            
        # Drop the weekends
        dataframe = dataframe.drop(weekends, axis=0)
        
        return dataframe
        
    # Create a Facebook prophet model without training
    def create_model(self):

        # Make the model
        model = fbprophet.Prophet(daily_seasonality=self.daily_seasonality,  
                                  weekly_seasonality=self.weekly_seasonality, 
                                  yearly_seasonality=self.yearly_seasonality,
                                  changepoint_prior_scale=self.changepoint_prior_scale,
                                  changepoints=self.changepoints)
        
        if self.monthly_seasonality:
            # Add monthly seasonality
            model.add_seasonality(name = 'monthly', period = 30.5, fourier_order = 5)
        
        return model
      
    # Find accuracy
    def find_accuracy(self):
        # Default start date is one year before end of data
        # Default end date is end date of data
        start_date = self.max_date - pd.DateOffset(years=1)
        end_date = self.max_date
            
        start_date, end_date = self.handle_dates(start_date, end_date)
        
        # Training data starts self.training_years years before start date and goes up to start date
        train = self.stock[(self.stock['timestamp'] < start_date.date()) & 
                           (self.stock['timestamp'] > (start_date - pd.DateOffset(years=self.training_years)).date())]
        
        # Testing data is specified in the range
        test = self.stock[(self.stock['timestamp'] >= start_date.date()) & (self.stock['timestamp'] <= end_date.date())]
        
        # Create and train the model
        model = self.create_model()
        model.fit(train)
        
        # Make a future dataframe and predictions
        future = model.make_future_dataframe(periods = 365, freq='D')
        future = model.predict(future)
        
        # Merge predictions with the known values
        test = pd.merge(test, future, on = 'ds', how = 'inner')
        train = pd.merge(train, future, on = 'ds', how = 'inner')
        
        # Calculate the differences between consecutive measurements
        test['pred_diff'] = test['yhat'].diff()
        test['real_diff'] = test['y'].diff()
        
        # Correct is when we predicted the correct direction
        test['correct'] = (np.sign(test['pred_diff']) == np.sign(test['real_diff'])) * 1
        
        # Accuracy when we predict increase and decrease
        increase_accuracy = 100 * np.mean(test[test['pred_diff'] > 0]['correct'])
        decrease_accuracy = 100 * np.mean(test[test['pred_diff'] < 0]['correct'])

        # Calculate mean absolute error
        test_errors = abs(test['y'] - test['yhat'])
        test_mean_error = np.mean(test_errors)

        train_errors = abs(train['y'] - train['yhat'])
        train_mean_error = np.mean(train_errors)

        # Calculate percentage of time actual value within prediction range
        test['in_range'] = False

        for i in test.index:
            if (test.ix[i, 'y'] < test.ix[i, 'yhat_upper']) & (test.ix[i, 'y'] > test.ix[i, 'yhat_lower']):
                test.ix[i, 'in_range'] = True

        in_range_accuracy = 100 * np.mean(test['in_range'])

        return in_range_accuracy

    def should_trade(self):
        rsi_val = float(self.rsi_data['Technical Analysis: STOCHRSI'][self.max_date.strftime('%Y-%m-%d')]['FastD'])
        if self.is_divergence() and rsi_val > 20:
            return True
        else:
            return False
        
    def is_divergence(self):
        today_uptrend = False
        today_downtrend = False
        prev_uptrend = False
        prev_downtrend = False

        # Let's look at the history 2 days ago
        days_ago = 2
        todays_date = self.max_date.strftime('%Y-%m-%d')

        today_macd_val = float(self.macd_data['Technical Analysis: MACD'][todays_date]['MACD'])
        today_macd_signal_val = float(self.macd_data['Technical Analysis: MACD'][todays_date]['MACD_Signal'])

        gap = abs(today_macd_val - today_macd_signal_val)

        # Basically if the gap is big enough, then cool, otherwise no shot
        # Revist number from time to time to see if it's good enough
        if (gap < .15):
            return False
                
        if today_macd_val > today_macd_signal_val:
            today_uptrend = True
        else:
            today_downtrend = True

        for i in range(days_ago,1,-1):
            previous_date = (self.max_date - pd.DateOffset(days=i)).strftime('%Y-%m-%d')
            prev_macd_val = float(self.macd_data['Technical Analysis: MACD'][previous_date]['MACD'])
            prev_macd_signal_val = float(self.macd_data['Technical Analysis: MACD'][previous_date]['MACD_Signal'])

            if prev_macd_val > prev_macd_signal_val:
                prev_uptrend = True
            else:
                prev_downtrend = True

            if today_uptrend != prev_uptrend or today_downtrend != prev_downtrend:
                return True

        return False


    # Predict the future price for a given range of days
    def predict_future(self, days=30):
        #Set the best changepointprior
        # self.changepoint_prior_scale = self.changepoint_prior_validation()
        
        model = None
        isNew = False

        if not os.path.exists('persistence/' + self.symbol):
            with open('persistence/' + self.symbol, 'w'): pass
        else:
            if os.path.getsize('persistence/' + self.symbol) > 0:
                with open('persistence/' + self.symbol, 'rb') as handle:
                    model = pickle.load(handle)

        if model is None:
            # Use past self.training_years years for training
            train = self.stock[self.stock['timestamp'] > (max(self.stock['timestamp']) - pd.DateOffset(years=self.training_years)).date()]
            
            model = self.create_model()
            model.fit(train)
            isNew = True

        # Future dataframe with specified number of days to predict
        future = model.make_future_dataframe(periods=days, freq='D')
        future = model.predict(future)
            
        # Only concerned with future dates
        future = future[future['ds'] >= max(self.stock['timestamp']).date()]
        
        # Remove the weekends
        future = self.remove_weekends(future)
        
        # Calculate whether increase or not
        future['diff'] = future['yhat'].diff()
    
        future = future.dropna()

        # Find the prediction direction and create separate dataframes
        future['direction'] = (future['diff'] > 0) * 1
        
        # Rename the columns for presentation
        future = future.rename(columns={'ds': 'timestamp', 'yhat': 'estimate', 'diff': 'change', 
                                        'yhat_upper': 'upper', 'yhat_lower': 'lower'})

        if isNew:
            with open('persistence/' + self.symbol, 'wb') as handle:
                pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        return future


    def changepoint_prior_validation(self):
        # Default start date is two years before end of data
        # Default end date is one year before end of data
        start_date = self.max_date - pd.DateOffset(years=2)
        end_date = self.max_date - pd.DateOffset(years=1)

        changepoint_priors = [0.001, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        
        # Convert to pandas datetime for indexing dataframe
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        start_date, end_date = self.handle_dates(start_date, end_date)
                               
        # Select self.training_years number of years
        train = self.stock[(self.stock['timestamp'] > (start_date - pd.DateOffset(years=self.training_years)).date()) & 
        (self.stock['timestamp'] < start_date.date())]
        
        # Testing data is specified by range
        test = self.stock[(self.stock['timestamp'] >= start_date.date()) & (self.stock['timestamp'] <= end_date.date())]

        eval_days = (max(test['timestamp']).date() - min(test['timestamp']).date()).days
        
        results = pd.DataFrame(0, index = list(range(len(changepoint_priors))), 
            columns = ['cps', 'train_err', 'train_range', 'test_err', 'test_range'])

        # Make these very high because we want them to be as close to 0 as possible
        best_cps_val = 1000
        best_accuracy = 1000
        # Iterate through all the changepoints and find the best one
        for i, prior in enumerate(changepoint_priors):
            results.ix[i, 'cps'] = prior
            
            # Select the changepoint
            self.changepoint_prior_scale = prior
            
            # Create and train a model with the specified cps
            model = self.create_model()
            model.fit(train)
            future = model.make_future_dataframe(periods=eval_days, freq='D')
                
            future = model.predict(future)
            
            # Training results and metrics
            train_results = pd.merge(train, future[['ds', 'yhat', 'yhat_upper', 'yhat_lower']], on = 'ds', how = 'inner')
            avg_train_error = np.mean(abs(train_results['y'] - train_results['yhat']))
            avg_train_uncertainty = np.mean(abs(train_results['yhat_upper'] - train_results['yhat_lower']))
            
            results.ix[i, 'train_err'] = avg_train_error
            results.ix[i, 'train_range'] = avg_train_uncertainty
            
            # Testing results and metrics
            test_results = pd.merge(test, future[['ds', 'yhat', 'yhat_upper', 'yhat_lower']], on = 'ds', how = 'inner')
            avg_test_error = np.mean(abs(test_results['y'] - test_results['yhat']))
            avg_test_uncertainty = np.mean(abs(test_results['yhat_upper'] - test_results['yhat_lower']))
            
            results.ix[i, 'test_err'] = avg_test_error
            results.ix[i, 'test_range'] = avg_test_uncertainty

            average_errors = (avg_test_error + avg_test_uncertainty) / 2
            acc = self.find_accuracy()
            accuracy = average_errors / acc

            if min(best_accuracy, accuracy) < best_accuracy:
                best_cps_val = prior
                best_accuracy = accuracy

        return best_cps_val
       