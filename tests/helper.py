import pandas as pd
import numpy as np 

class Helper():
    def __init__(self, oracle):
        self.stock = oracle.stock
        self.model = oracle.create_model()
        self.oracle = oracle

    # Evaluate prediction model for one year
    def prediction(self, nshares):
        # Default start date is one year before end of data
        # Default end date is end date of data
        start_date, end_date = self.handle_dates(self.oracle.max_date - pd.DateOffset(years=1), self.oracle.max_date)
        
        # Training data starts self.training_years years before start date and goes up to start date
        train = self.stock[(self.stock['timestamp'] < start_date.date()) & 
                            (self.stock['timestamp'] > (start_date - pd.DateOffset(years=self.oracle.training_years)).date())]
        
        # Testing data is specified in the range
        test = self.stock[(self.stock['timestamp'] >= start_date.date()) & (self.stock['timestamp'] <= end_date.date())]
        
        # Create and train the model
        self.model.fit(train)

        
        # Make a future dataframe and predictions
        future = self.model.make_future_dataframe(periods = 365, freq='D')
        future = self.model.predict(future)
        
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

        # Only playing the stocks when we predict the stock will increase
        test_pred_increase = test[test['pred_diff'] > 0]
        
        test_pred_increase.reset_index(inplace=True)
        prediction_profit = []
        
        # Iterate through all the predictions and calculate profit from playing
        """
        for i, date in enumerate(test_pred_increase['timestamp']):

            current_date = date.strftime('%Y-%m-%d')
            macd_val = float(self.macd_data['Technical Analysis: MACD'][current_date]['MACD'])
            rsi_val = float(self.rsi_data['Technical Analysis: STOCHRSI'][current_date]['FastD'])

            # If we predicted up and the price goes up, we gain the difference
            # If we predicted up and the price goes down, we lose the difference
            if macd_val > 0:
                prediction_profit.append(nshares * test_pred_increase.ix[i, 'real_diff'])
            else: 
                if rsi_val > 20:
                    prediction_profit.append(nshares * test_pred_increase.ix[i, 'real_diff'])
                else:
                    prediction_profit.append(nshares * 1)

        """

        # Iterate through all the predictions and calculate profit from playing
        for i, correct in enumerate(test_pred_increase['correct']):
            
            # If we predicted up and the price goes up, we gain the difference
            if correct == 1:
                prediction_profit.append(nshares * test_pred_increase.ix[i, 'real_diff'])
            # If we predicted up and the price goes down, we lose the difference
            else:
                prediction_profit.append(nshares * test_pred_increase.ix[i, 'real_diff'])
        
        test_pred_increase['pred_profit'] = prediction_profit

        # Put the profit into the test dataframe
        test = pd.merge(test, test_pred_increase[['ds', 'pred_profit']], on = 'ds', how = 'left')
        test.ix[0, 'pred_profit'] = 0

        # Profit for either method at all dates
        test['pred_profit'] = test['pred_profit'].cumsum().ffill()
        test['hold_profit'] = nshares * (test['y'] - float(test.ix[0, 'y']))

        result = {}

        # Display some friendly information about the perils of playing the stock market
        result['prediction'] = np.sum(prediction_profit)
        result['hold'] = float(test.ix[len(test) - 1, 'hold_profit'])

        return result

    # Calculate and plot profit from buying and holding shares for specified date range
    def buy_and_hold(self, start_date=None, end_date=None, nshares=1):    
        start_date, end_date = self.oracle.handle_dates(start_date, end_date)
            
        # Find starting and ending price of stock
        start_price = float(self.stock[self.stock['timestamp'] == start_date]['Adj. Open'])
        end_price = float(self.stock[self.stock['timestamp'] == end_date]['Adj. Close'])
        
        # Make a profit dataframe and calculate profit column
        profits = self.oracle.make_df(start_date, end_date)
        profits['hold_profit'] = nshares * (profits['Adj. Close'] - start_price)
        
        # Total profit
        total_hold_profit = nshares * (end_price - start_price)


    """
    Make sure start and end dates are in the range and can be
    converted to pandas datetimes. Returns dates in the correct format
    """
    def handle_dates(self, start_date, end_date):
        
        
        # Default start and end date are the beginning and end of data
        if start_date is None:
            start_date = self.oracle.min_date
        if end_date is None:
            end_date = self.oracle.max_date
        
        try:
            # Convert to pandas datetime for indexing dataframe
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
        
        except Exception as e:
            print('Enter valid pandas date format.')
            print(e)
            return
        
        return start_date, end_date