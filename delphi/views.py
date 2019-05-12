from django.shortcuts import render

from .models import Stock
from stocker import Stocker

# matplotlib pyplot for plotting
import matplotlib.pyplot as plt

import matplotlib

def index(request):
  template = 'delphi/index.html'
  prediction = {}
  context = {
      'prediction': prediction,
  }

  company = Stocker(ticker="MSFT", exchange='EOD')

  # stock_history = company.stock()

  if company:
      company.plot_stock(start_date=None, end_date=None, stats=['Adj. Close'], plot_type='basic')

  return render(request, template, context)


def predict(request, ticker_name, exchange, days):
	company = Stocker(ticker=ticker_name, exchange='EOD')

	if company:
		Stocker.plot_stock(start_date=None, end_date=None, stats=['Adj. Close'], plot_type='basic')
