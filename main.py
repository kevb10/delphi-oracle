from flask import Flask
from flask import request 
from flask import render_template
from oracle import Oracle 
import requests

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def main():
    prediction_estimate = [] 
    prediction_direction = [] 
    result = {}

    if request.method == 'POST':
        symbol = request.form.get('symbol')
        stock = Oracle(symbol,  requests.Session())
        prediction = stock.predict_future(days=15)
        prediction_estimate = prediction['estimate'].to_numpy()
        prediction_direction = prediction['direction'].to_numpy()
        result["ticker"] = symbol.upper()

    result["prediction"] = prediction_estimate
    result["direction"] = prediction_direction
    
    return render_template('main.html', future=result)

