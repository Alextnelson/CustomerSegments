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
detergents = data.loc[:,'Detergents_Paper']
new_data = pd.DataFrame(data).drop('Detergents_Paper', axis=1)

# Split the data into training and testing sets using the given feature as the target
from sklearn.model_selection import train_test_split

features = new_data
target = detergents
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=0)

# Created a decision tree regressor and fit it to the training set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

# Score of the prediction using the testing set
score = regressor.score(X_test, y_test)
print(score)

# Produce a scatter matrix for each pair of features in the data
pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

# TODO: Scale the data using the natural logarithm
log_data = None

# TODO: Scale the sample data using the natural logarithm
log_samples = None

# Produce a scatter matrix for each pair of newly-transformed features
pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');