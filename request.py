# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 21:08:01 2018

@author: mrclx
"""

import requests
import json
import pandas as pd

# login data for intrinio API.
USER = '349b60cec630e203f91192e0e0eca870'
PW = '2c4c43248913d328f56a25d8ac49d994'

# Function to download data about Europe Small 200: download(europe_small_200(), Values)
# Function to download data about S&P 500: download(sp_500(end_slice=76), Values)
# Function to download data about DAX 30: download(dax_30(), Values)

# attributes to be requested. Includes features and label.
Values = ["dividendyield", "earningsyield", "enterprisevalue", "evtoebit",
          "evtoebitda", "evtofcff", "evtoinvestedcapital", "evtonopat",
          "evtoof", "pricetoearnings", "ebitdagrowth", "freecashflow",
          "revenuegrowth"]

def download(company, value):
    
    # request every combination of company and value from API.
    for item1 in company:
        for item2 in value:
            
            url = 'https://api.intrinio.com/historical_data?identifier=' + item1 + '&item=' + item2
            r = requests.get(url, auth=(USER, PW))
            
            # converts json.
            stmt= r.json()
            
            # save data in json format.
            file_name = 'data-'+item1+'-'+item2+'.txt'
            with open(file_name, 'w') as outfile:
                json.dump(stmt, outfile)
                print(item1 + "-" + item2)

def sp_500(start_slice=None, end_slice=None):
    
    # import Excel file.
    file = "SP-500-Stocks.xlsx"
    xl = pd.ExcelFile(file)
    
    # parse and clean data.
    data = xl.parse(0, header = [0], skiprows = None)
    data = data.iloc[:, 0]
    data = data.dropna()
    
    # slicing data
    company_names = list(data)
    company_names_sliced = company_names[start_slice:end_slice]
    
    return company_names_sliced

def europe_small_200():
        
    # import Excel file.
    file = "europe_small_200.xlsx"
    xl = pd.ExcelFile(file)
    
    # parse and clean data.
    data = xl.parse(0, header = [1], skiprows = [2])    
    data = data.iloc[:, 0]
    data = data.dropna()
    company_names = list(data)
    
    return company_names

def dax_30():
    
    # import Excel file.
    file = "dax-30.xlsx"
    xl = pd.ExcelFile(file)
    
    # parse and clean data.
    data = xl.parse(0)    
    data = data.iloc[:, 0]
    data = data.dropna()
    company_names = list(data)
    
    return company_names