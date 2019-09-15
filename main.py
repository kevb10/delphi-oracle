from flask import Flask
from flask import request 
from flask import render_template
from oracle import Oracle 
import requests

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def main():
    prediction_result = [] * 5
    result = {}

    if request.method == 'POST':
        symbol = request.form.get('symbol')
        stock = Oracle(symbol,  requests.Session())
        prediction = stock.predict_future(days=5)
        prediction_result = prediction['estimate'].values
        result["ticker"] = symbol.upper()

    result["prediction"] = prediction_result
    
    return render_template('main.html', future=result)

