from django.http import HttpResponse

from .models import Stock
from stocker import Stocker

# matplotlib pyplot for plotting
import matplotlib.pyplot as plt

import matplotlib

def index(request):
  # TODO: Create a simple landing page
  return HttpResponse("WELCOME", content_type='application/json')

def predict(request, ticker_name):
  company = Stocker(ticker=ticker_name)
  data = company.retrieve_stock_data(start_date=None, end_date=None, stats=['Adj. Close'], plot_type='basic')
  json = data.to_json(orient='split')

  print(company.create_prophet_model())

  return HttpResponse(json, content_type='application/json')
