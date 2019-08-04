from oracle import Oracle 
import csv
import pandas as pd


class Delphi():

    def __init__(self):
        # Find me some money makers given a list
        # of all publicly traded companies
        companies = pd.read_table("companies.csv", sep=",")
        for company in companies:
            print(company)
            if self.is_money_maker(company):
                #Add to database
                print("Yeah let's make money")
            else:
                print("Damn, naw that aint it fam")


    def is_money_maker(self, company):
        stock = Oracle(company)
        prediction = stock.predict_future(days=15)
        should_trade = stock.should_trade()

        return should_trade 
