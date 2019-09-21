import unittest
from oracle import Oracle
from helper import Helper
import requests

class TestStockPrediction(unittest.TestCase):

    # def setUp(self):
        # self.oracle = Oracle("msft", requests.session())
        # self.helper = Helper(self.oracle)

    def test_stock_initalized(self):
        self.oracle = Oracle("msft", requests.session())
        self.helper = Helper(self.oracle)
        result = self.helper.prediction(1000)
        
        prediction = result['prediction']
        hold = result['hold']

        print(prediction)
        print(hold)

        assert prediction >= hold

    def test_reporter(self):
        self.oracle = Oracle("msft", requests.session())
        self.oracle.report()

        assert 0 == 0

if __name__ == '__main__':
    unittest.main()