# -*- coding: utf-8 -*-
"""
Created on Sun May 13 18:07:41 2018

@author: mrclx
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def input_df():
    
    # import features and labels.
    features = pd.read_csv("_features.csv", index_col = 0)
    labels = pd.read_csv("_labels.csv", header = None, index_col = 0)
    
    # concat features and labels.
    df = pd.concat([features, labels], axis = 1)
    
    return df
    

def scatter_input():
    
    sns.set()
    
    df = input_df()
    
    # plot enterprisevalue against every feature.
    for key in list(df.iloc[:, :-2].columns.values):
    
        df.plot(kind = "scatter",
                x = key,
                y = df.iloc[:, -1].name,
                logx = True,
                logy = True)
    
        plt.ylabel("enterprisevalue")
        plt.savefig("./plt/scatter-{}-ev.png".format(key))
    
def box_input():
    
    sns.set()

    df = input_df()
    
    # create box plot for every column.
    for key in list(df.columns.values):
        
        plt.boxplot(df[key], whis = "range")
        plt.autoscale(axis = "y")
        plt.margins(0.02)
        
        # set x-ticks.
        if key == 1 :
            key = "enterprisevalue"
            plt.xticks([1], [key])
            
        else :
            plt.xticks([1], [key])
        
        plt.savefig("./plt/boxplt-{}.png".format(key))
        plt.show()
        
def hparams_loss_plot():
    
    sns.set()
    
    normed = pd.read_csv("./logs/hparams_log.csv", index_col = 0)
    unnormed = pd.read_csv("./logs_dropout=None/hparams_log.csv", index_col = 0)
    
    plt.plot(normed["avg_loss"], marker = ".", color = "blue",
             label = "Normierte Daten (Z-Wert)")
    plt.plot(unnormed["avg_loss"], marker = ".", color = "red",
             label = "Unnormierte Daten")
    
    plt.yscale("log")
    
    plt.ylabel("Average Loss")
    plt.xlabel("Iteration")
    
    plt.legend(loc="upper right")
    plt.annotate("Abbruch, da Modell divergiert", xy = (7, 1e22),
                 xytext = (7, 1e16), arrowprops = {"color": "red"})
    
    plt.margins(0.02)
    
    plt.savefig("./plt/hparams_loss_normed_unnormed.png")
    
def hparams_dimensions_plot():
    
    # seaborn settings.
    sns.set()
    
    # import data from hparams_log.csv.
    data = pd.read_csv("./logs/hparams_log.csv", index_col = 0)
    data = data[["num_layers", "num_nodes", "avg_loss"]]
    
    # star where dimensions of model with minimal loss is located.
    x = data[data["avg_loss"] == data["avg_loss"].min()]["num_nodes"]
    y = data.loc[data["avg_loss"] == data["avg_loss"].min()]["num_layers"]
    plt.plot(x, y, marker = "*", markersize = 15, markerfacecolor = "blue")
    
    # make position more realizeable.
    plt.annotate("optimal model", xy = (15, 1.1),
                 xytext = (100, 2), arrowprops = {"color": "blue"})
    
    # transform data for contourf.
    data = data.pivot_table(index = "num_layers",
                            columns = "num_nodes",
                            values = "avg_loss",
                            aggfunc="min",
                            fill_value = 30
                            )
    
    # create meshgrid. x-axis: num_nodes. y_axis = num_layers.
    u = np.array(data.columns)
    v = np.array(data.index)
    X, Y = np.meshgrid(u, v)
    
    # color: avg_loss
    Z = np.array(data)
    
    plt.contourf(X, Y, Z, 50, cmap = "afmhot")
    
    plt.xlabel("number of nodes per layer")
    plt.ylabel("number of layers")
    plt.colorbar(label = "average loss")
    plt.margins(0.02)
    
    plt.savefig("./plt/hparams_dimensions.png")
    
    