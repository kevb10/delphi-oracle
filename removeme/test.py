import unittest
from stocker import Stocker

class TestStockPrediction(unittest.TestCase):

    def test_stock_initalized(self):
        amazon = Stocker('AMZN')
        self.assertEqual(0,0)

if __name__ == '__main__':
    unittest.main()