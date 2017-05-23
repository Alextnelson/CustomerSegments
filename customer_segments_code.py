# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:27:16 2017

@author: alexandernelson
"""

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualizations code visuals.py
import visuals as vs

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print("Wholesale customers dataset has {} samples with {} features each.".format(*data.shape))
except:
    print("Dataset could not be loaded. Is the dataset missing?")
    
# Display a description of the dataset
display(data.describe())
print(data.head())
print(data.columns)
print(data.loc[:, ['Fresh', 'Frozen', 'Detergents_Paper']])

# Selected three random customers from sample by randomly selecting indice numbers for indices
indices = [5, 88, 274]


# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices, :], columns = data.keys()).reset_index(drop = True)
print("Chosen samples of wholesale customers dataset: ")
display(samples)

# Copied the DataFrame, using the 'drop' function to drop the given feature
new_data = pd.DataFrame(data).drop('Fresh', axis=1)
print(new_data.head())

# Split the data into training and testing sets using the given feature as the target
X_train, X_test, y_train, y_test = (None, None, None, None)

# TODO: Create a decision tree regressor and fit it to the training set
regressor = None

# TODO: Report the score of the prediction using the testing set
score = None


df.drop('column_name', axis=1, inplace=True)