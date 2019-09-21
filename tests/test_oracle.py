import unittest
from oracle import Oracle
from helper import Helper
import requests

class TestStockPrediction(unittest.TestCase):

    def test_stock_initalized(self):
        oracle = Oracle("tsla", requests.session())
        helper = Helper(oracle)
        result = helper.prediction(1000)
        
        prediction = result['prediction']
        hold = result['hold']

        print(prediction)
        print(hold)

        assert prediction >= hold

if __name__ == '__main__':
    unittest.main()