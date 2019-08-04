import unittest
from delphi import Delphi

class TestStockPrediction(unittest.TestCase):

    def test_stock_initalized(self):
        delphi = Delphi()
        self.assertEqual(0,0)

if __name__ == '__main__':
    unittest.main()