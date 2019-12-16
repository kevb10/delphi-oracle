import json
import datetime
from botocore.vendored import requests

def screen(event, context):
    # Get ticker symbol from parameter
    ticker = event['ticker']
    
    # ALPHAVANTAGE API Key
    alpha_vantage_api_key = 'YHST3SHIZZX5VRUA'
    
    # Compose URL for request
    url = ('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + 
            ticker + 
            '&interval=daily&series_type=close&apikey=' + 
            alpha_vantage_api_key)
            
    # Get stock raw data        
    stock_data = json.loads(requests.Session().get(url, verify=False).content)
    filter_key = 'Time Series (Daily)'
    # Filter it
    stock_data = stock_data[filter_key]
    # Sort it by date
    sorted_data = sorted(stock_data.items(), key=lambda x: datetime.datetime.strptime(x[0], '%Y-%m-%d'))
 
    # Copy paste to a different function
    # sma_ten = find_sma(10, stock_data)
    close_price = find_close(stock_data, sorted_data)
    sma_twenty = find_sma(20, stock_data)
    result = "Can't trade"
    
    if did_breakout(stock_data, sorted_data, sma_twenty):
        result = "Ha! Let's trade"
    
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
    
    
def find_sma(length, data):
    close_key = '4. close'
    ma_ten_raw = 0

    # getting current date and time
    d = datetime.datetime.today()
    date_key = d.strftime('%Y-%m-%d')
    
    i = 0
    
    while i < length:
      y = datetime.timedelta(days=-1) # previous day
      d = d + y
      date_key = d.strftime('%Y-%m-%d')
    
      try:
        ma_ten_raw = ma_ten_raw + float(data[date_key][close_key])
        i += 1
      except:
        continue
    
    return ma_ten_raw / length
    
def find_close(data, sorted_data):
  close_key = '4. close'
  return float(data[sorted_data[-1][0]][close_key])

# See github issue # for the explanation of this
def did_breakout(data, sorted_data, basis):
  close_key = '4. close'
  low_key = '3. low'

  # Current month
  current_month = sorted_data[-1][0]
  
  # Last month
  last_month = sorted_data[-2][0]
  
  return  (float(data[last_month][low_key]) < basis) and (float(data[last_month][close_key]) > basis) and (float(data[current_month][close_key]) > float(data[last_month][close_key])) 

