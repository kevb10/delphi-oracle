import json

from django.http import HttpResponse

from stocker import Stocker


def index(request):
    # TODO: Create a simple landing page
    return HttpResponse("WELCOME", content_type='application/json')


def predict(request, ticker_name):
    company = Stocker(ticker=ticker_name)
    stock_data = company.retrieve_stock_data(start_date=None, end_date=None, stats=['Adj. Close'], plot_type='basic')
    json_stock_data = stock_data.to_json(orient='columns')

    return HttpResponse(json_stock_data, content_type='application/json')


def evaluate(request, ticker_name):
    company = Stocker(ticker=ticker_name)
    past_performance, model_prediction = company.create_prophet_model()
    past_performance = past_performance.to_json(orient='columns')
    model_prediction = model_prediction.to_json(orient='columns')

    evaluation_model = {
        'history': past_performance,
        'prediction': model_prediction
    }

    json_evaluation_model = json.dumps(evaluation_model)

    return HttpResponse(json_evaluation_model, content_type='application/json')
