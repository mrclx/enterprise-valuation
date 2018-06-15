# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 16:28:03 2018

@author: mrclx
"""
import pandas as pd
import glob
from scipy import stats

def data_cleaning(folder, file_features, file_labels):

    # Fetches names of available file.
    files = glob.glob("{}*.txt".format(folder))

    # Initializes empty list.
    data_list = []
    print("Reading...")

    for file_name in files:
    
        # Opens file.
        with open(file_name, 'r') as file:
            # TextWrapper to object string.
            stmt_string = file.read()
        
        # Converts JSON to DataFrame.
        stmt_df = pd.read_json(stmt_string)
        stmt_df_data = stmt_df.loc[:, ["data",
                                       "identifier",
                                       "item"]]
        
        # Appends DaFrame to list of DataFrames.
        data_list.append(stmt_df_data)
    print("data_list created.")

    # Concatenates DataFrames.
    all_stmt = pd.concat(data_list, ignore_index = True)
    print("DataFrames concatenated.")

    # Converts and unnests column "data" to DataFrame
    all_stmt["date"] = all_stmt["data"].apply(lambda x: x["date"])
    all_stmt["value"] = all_stmt["data"].apply(lambda x: x["value"])
    print("Column 'data' unnested.")

    # Pivots DataFrame.
    all_stmt["new_index"] = all_stmt["identifier"] + all_stmt["date"]
    all_stmt["duplicate"] = all_stmt["new_index"] + all_stmt["item"]
    all_stmt = all_stmt.drop_duplicates(subset = "duplicate")
    all_stmt = all_stmt.pivot(index = "new_index",
                              columns = "item",
                              values = "value")
    print("DataFrame pivoted.")
    
    # Drops rows with NaN values.
    all_stmt = all_stmt[(all_stmt != "nm")]
    all_stmt = all_stmt.dropna()
    print("NA values dropped.")
    
    # Normalization. Calculates z-score of each value.
    ind = all_stmt.index
    col = all_stmt.columns
    all_stmt = stats.zscore(all_stmt.values.tolist(), axis = 0)
    all_stmt = pd.DataFrame(all_stmt, index = ind, columns = col)
    print("Data normalized.")
    
    # Separates features from labels. Saves DataFrames as CSV.
    labels = all_stmt["enterprisevalue"]
    labels.to_csv(file_labels)
    features = all_stmt.drop(["enterprisevalue"], axis = 1)
    features.to_csv(file_features)
    print("DataFrame stored as '{0}' and '{1}'.".format(file_features,
                                                      file_labels))