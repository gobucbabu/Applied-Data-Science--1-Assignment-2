# -*- coding: utf-8 -*-
"""
Created on Wed May 10 19:48:46 2023

@author: gobub
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
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
df_AFFV_0 = read_df("API_NV.AGR.TOTL.ZS_DS2_en_csv_v2_5359510.csv")

# extracting the columns we want for clustering and
# making copies of those dataframes 
df_arableland = df_arableland_0.loc[:, ["Country Name", "2010"]].copy()
df_AFFV = df_AFFV_0.loc[:, ["Country Name", "2010"]].copy()

# exploering the datasets
# print(df_arableland.describe())
# print(df_AFFV.describe())

# merging the dataframes to cluster
df_cluster_0 = pd.merge(df_arableland, df_AFFV, on="Country Name")
df_cluster = df_cluster_0.rename(columns={"2010_x": "Arable Land",
                                          "2010_y": "AFFV"})
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
    kmeans.fit(df_cluster_norm)  # fit done on x,y pairs
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

# creating a custom colormap
colors = mc.ListedColormap(['b', 'r', 'g', 'y', 'darkmagenta'])
# plotting the clusters using plt.scatter
plt.figure(figsize=(6, 6))
plt.scatter(df_cluster_norm["Arable Land"], df_cluster_norm["AFFV"], 12,
            labels, marker="o", cmap=colors)
plt.scatter(xcen, ycen, 50, "k", marker="d")
plt.xlabel("Arable land")
plt.ylabel("AFFV")
plt.legend(handles=plt.scatter(df_cluster_norm["Arable Land"],
                               df_cluster_norm["AFFV"], 12,
            labels, marker="o", cmap=colors).legend_elements()[0],
           labels=range(kmeans.n_clusters))
plt.savefig("tempfig.png", dpi=300)
plt.show()

# move the cluster centres to the original scale
cen_0 = ct.backscale(cen, df_min, df_max)
xcen_0 = cen_0[:, 0]
ycen_0 = cen_0[:, 1]
# cluster by cluster
plt.figure(figsize=(6, 6))
plt.scatter(df_cluster["Arable Land"], df_cluster["AFFV"], 10,
            labels, marker="o", cmap=colors)
plt.scatter(xcen_0, ycen_0, 45, "k", marker="d")
plt.xlabel("Arable land")
plt.ylabel("AFFV")
plt.legend(handles=plt.scatter(df_cluster_norm["Arable Land"],
                               df_cluster_norm["AFFV"], 12,
            labels, marker="o", cmap=colors).legend_elements()[0],
           labels=range(1, kmeans.n_clusters+1))
plt.show()


