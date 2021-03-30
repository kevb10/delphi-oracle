import unittest
from oracle import Oracle
from helper import Helper
import requests

class TestStockPrediction(unittest.TestCase):

    def test_stock_initalized(self):
        oracle = Oracle("msft", requests.session())
        helper = Helper(oracle)
        result = helper.prediction(1000)
        
        prediction = result['prediction']
        hold = result['hold']

        print(prediction)
        print(hold)

        assert prediction >= hold

class TestReporter(unittest.TestCase):
    
    def test_reporter(self):
        oracle = Oracle("msft", requests.session())
        oracle.report()
        oracle.close()

        self.assertEqual(0,0)

if __name__ == '__main__':
    unittest.main()