# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:27:16 2017

@author: alexandernelson
"""

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames
from sklearn.decomposition import PCA

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

# Scale the data using the natural logarithm
log_data = np.log(data)

# Scale the sample data using the natural logarithm
log_samples = np.log(samples)

# Produce a scatter matrix for each pair of newly-transformed features
pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde')

# For each feature find the data points with extreme high or low values
for feature in log_data.keys():
    
    # Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature], 25)
    
    # Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature], 75)
    
    # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = (Q3 - Q1) * 1.5
    
    # Display the outliers
    print("Data points considered outliers for the feature '{}':".format(feature))
    display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])
    
    
    
# Select the indices for data points you wish to remove
outliers  = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))].index.get_values()

# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)
good_data.head()


# Apply PCA by fitting the good data with the same number of dimensions as features
pca = PCA(n_components=6)
pca.fit(good_data)

# Transform log_samples using the PCA fit above
pca_samples = pca.fit_transform(log_samples)

# Generate PCA results plot
pca_results = vs.pca_results(good_data, pca)

# Display sample log-data after having a PCA transformation applied
display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))

# Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components=2)

# Transform the good data using the PCA fit above
reduced_data = pca.fit_transform(good_data)

# Transform log_samples using the PCA fit above
pca_samples = pca.fit_transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

# Display sample log-data after applying PCA transformation in two dimensions
display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))

# Fit a clustering algorithm to the reduced_data and assign it to clusterer.
# Predict the cluster for each data point in reduced_data using clusterer.predict and assign them to preds.
# Find the cluster centers using the algorithm's respective attribute and assign them to centers.
# Predict the cluster for each sample data point in pca_samples and assign them sample_preds.
# Import sklearn.metrics.silhouette_score and calculate the silhouette score of reduced_data against preds.
# Assign the silhouette score to score and print the result.

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9]

for n_cluster in range_n_clusters:
    # Apply your clustering algorithm of choice to the reduced data 
    clusterer = KMeans(n_clusters=n_cluster, random_state=10).fit(reduced_data)
    # Predict the cluster for each data point
    preds = clusterer.predict(reduced_data)
    # Find the cluster centers
    centers = clusterer.cluster_centers_

    # Predict the cluster for each transformed sample data point
    sample_preds = clusterer.predict(pca_samples)

    # Calculate the mean silhouette coefficient for the number of clusters chosen
    score = silhouette_score(reduced_data, preds)
    print("For n_clusters =", n_cluster,
          "The average silhouette_score is :", score)

# Chose n_clusters = 2 based on silhouette score  
# Apply your clustering algorithm of choice to the reduced data 
clusterer = KMeans(n_clusters=2, random_state=10).fit(reduced_data)
    
# Predict the cluster for each data point
preds = clusterer.predict(reduced_data)

# Find the cluster centers
centers = clusterer.cluster_centers_

# Predict the cluster for each transformed sample data point
sample_preds = clusterer.predict(pca_samples)

# Calculate the mean silhouette coefficient for the number of clusters chosen
score = silhouette_score(reduced_data, preds)

# Display the results of the clustering from implementation
vs.cluster_results(reduced_data, preds, centers, pca_samples)

# Apply the inverse transform to centers using pca.inverse_transform and assign the new centers to log_centers.
# Apply the inverse function of np.log to log_centers using np.exp and assign the true centers to true_centers.

# TODO: Inverse transform the centers
log_centers = pca.inverse_transform(centers)

# TODO: Exponentiate the centers
true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)

# Display the predictions
for i, pred in enumerate(sample_preds):
    print("Sample point", i, "predicted to be in Cluster", pred)

# Display the clustering results based on 'Channel' data
vs.channel_results(reduced_data, outliers, pca_samples)

