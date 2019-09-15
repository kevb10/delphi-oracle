from oracle import Oracle 
import csv
import pandas as pd
import requests

class Delphi():

    def __init__(self):
        self.session = requests.Session()
        self.reporter()

    def reporter(self):
        # Find me some money makers given a list
        # of all publicly traded companies
        companies = pd.read_table("companies.csv", sep=",")
        # errors = ""
        with open("report.txt", "a+") as report:
            for company in companies:
                try:
                    if self.is_money_maker(company):
                        report.write(company + "\n")

                except Exception as e:
                    print(e)
                    # errors += "," + company

        # print(errors)

    def is_money_maker(self, company):
        stock = Oracle(company, self.session)
        prediction = stock.predict_future(days=15)
        should_trade = stock.should_trade()
        # stock.close()

        return should_trade 
