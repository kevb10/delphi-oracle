from flask import Flask
from flask import request 
from flask import render_template
from oracle import Oracle 
import numpy as np 
import requests

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def main():
    result = {}

    if request.method == 'POST':
        symbol = request.form.get('symbol')
        stock = Oracle(symbol,  requests.Session())
        prediction = stock.predict_future(15)
        accuracy, increase_accuracy, decrease_accuracy, pred_profit, hold_profit = stock.evaluate_prediction()
                
        prediction["timestamp"] = prediction['timestamp'].dt.day
        result["ticker"] = symbol.upper()
        result["prediction"] = prediction
        result["increase_accuracy"] = increase_accuracy
        result["decrease_accuracy"] = decrease_accuracy
        result["in_range_accuracy"] = accuracy
        result["pred_profit"] = pred_profit
        result["hold_profit"] = hold_profit
    
    return render_template('main.html', future=result)

