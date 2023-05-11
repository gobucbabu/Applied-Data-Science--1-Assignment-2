# -*- coding: utf-8 -*-
"""
Created on Wed May 10 19:48:46 2023

@author: gobub
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cluster_tools as ct
import sklearn.cluster as cluster
import sklearn.metrics as skmet

def read_df(fname):
    """
    Function to read the csv file into a dataframe and do some cleaning.
    Takes the filename as the argument.
    """
    # read file into a dataframe
    df0 = pd.read_csv(fname, on_bad_lines='skip', skiprows=4)
    # some cleaning
    df0.drop(columns=["Country Code"], axis=1, inplace=True)
    df1 = df0.sort_index().fillna(0)

    return df1


# calling the functions
df_arableland_0 = read_df("API_AG.LND.ARBL.ZS_DS2_en_csv_v2_5362201.csv")
df_cerealyield_0 = read_df("API_AG.YLD.CREL.KG_DS2_en_csv_v2_5362385.csv")

# making copies of the dataframes to make clusters with
df_arableland = df_arableland_0.loc[:, ["Country Name", "2000"]].copy()
df_cerealyield = df_cerealyield_0.loc[:, ["Country Name", "2000"]].copy()

# exploering the datasets
# print(df_arableland.describe())
# print(df_cerealyield.describe())

# merging the dataframes to cluster
df_cluster_0 = pd.merge(df_arableland, df_cerealyield, on="Country Name")
df_cluster = df_cluster_0.rename(columns={"2000_x": "Arable Land",
                                          "2000_y": "Cereal Yield"})
df_cluster = df_cluster.set_index("Country Name")

# creating a scatter matrix to verify we can form clusters
pd.plotting.scatter_matrix(df_cluster, figsize=(6, 6), s=5, alpha=0.8)

# normalising the values, to create meaningful clusters, using 
# the cluster tools providec
df_cluster_norm, df_min, df_max = ct.scaler(df_cluster)

# calculating the silhouette score

print("n score")
# loop over number of clusters
for ncluster in range(2, 10):

    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(df_cluster_norm) # fit done on x,y pairs
    labels = kmeans.labels_
    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_
    # calculate the silhoutte score
    print(ncluster, skmet.silhouette_score(df_cluster, labels))

# plotting the clusters with the best k value
k = 5
# setting up clusterer with the expected number of clusters
kmeans = cluster.KMeans(n_clusters=k)
# Fit the data, results are stored in the kmeans object
kmeans.fit(df_cluster_norm)  # fit done on x,y pairs
labels = kmeans.labels_
# extract the estimated cluster centres
cen = kmeans.cluster_centers_
xcen = cen[:, 0]
ycen = cen[:, 1]
# plotting the clusters using plt.scatter
plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')
plt.scatter(df_cluster["Arable Land"], df_cluster["Cereal Yield"], 10, labels,
            marker="o", cmap=cm)
plt.scatter(xcen, ycen, 45, "k", marker="d")
plt.xlabel("Arable land")
plt.ylabel("Cereal Yield")
plt.show()#



