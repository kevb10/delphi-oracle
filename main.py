from flask import Flask
from flask import request 
from flask import render_template
from oracle import Oracle 
import requests

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def main():
    prediction_estimate = [] * 5
    prediction_direction = [] * 5
    result = {} 

    if request.method == 'POST':
        symbol = request.form.get('symbol')
        stock = Oracle(symbol,  requests.Session())
        prediction = stock.predict_future(days=5)
        prediction_estimate = prediction['estimate'].values
        prediction_direction = prediction['direction'].values
        result["ticker"] = symbol.upper()

    result["prediction"] = prediction_estimate
    result["direction"] = prediction_direction
    
    return render_template('main.html', future=result)

