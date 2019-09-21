from flask import Flask
from flask import request 
from flask import render_template
from oracle import Oracle 
import requests

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def main():
    result = {}

    if request.method == 'POST':
        symbol = request.form.get('symbol')
        stock = Oracle(symbol,  requests.Session())
        prediction = stock.predict_future(days=15)
        prediction["timestamp"] = prediction['timestamp'].dt.day
        result["ticker"] = symbol.upper()
        result["prediction"] = prediction
    
    return render_template('main.html', future=result)

