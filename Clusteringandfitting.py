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
import scipy.optimize as opt
import errors as err
import seaborn as sns

def read_df(fname):
    """
    Function to read the csv file into a dataframe and create a transposed
    dataframe also. Returns both the dataframes after some cleaning.
    Takes the filename as the parameter.
    """
    # read file into a dataframe
    df10 = pd.read_csv(fname, on_bad_lines='skip', skiprows=4)
    # some cleaning
    df10.drop(columns=["Country Code"], axis=1, inplace=True)
    df1 = df10.sort_index().fillna(0)
    # transposing and cleaning
    df20 = df1.drop(columns=["Indicator Name", "Indicator Code"])
    df20.set_index("Country Name")
    df21 = df20.T
    df2 = df21.T.set_index("Country Name").T

    return df1, df2


def findsilhouette(df):
    """
    Function to find the silhouette values of a dataframe to find
    the best K value.
    Paramter: The (normalised) dataframe with values to be clustered
    """
    print("n score")
    # loop over number of clusters
    for ncluster in range(2, 10):

        # set up the clusterer with the number of expected clusters
        kmeans = cluster.KMeans(n_clusters=ncluster)
        # Fit the data, results are stored in the kmeans object
        kmeans.fit(df)  # fit done on x,y pairs
        labels = kmeans.labels_
        # extract the estimated cluster centres
        cen = kmeans.cluster_centers_
        # calculate the silhoutte score
        print(ncluster, skmet.silhouette_score(df_cluster, labels))

    return


def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0
    and growth rate g.
    """
    t = t - 1960
    f = n0 * np.exp(g*t)

    return f


def logistic(t, a, b, c):
    """Fit it into a logistic curve. Parameters:
        t- the time series
        a- maximum value of curve
        b- growth rate
        c- point of inflection
    """
    k = a / (1 + np.exp(-b * (t - c)))

    return k


# setting the style
sns.set_style("darkgrid")


# calling the functions
a, b = read_df("API_AG.LND.ARBL.ZS_DS2_en_csv_v2_5362201.csv")
df_arableland_0, df_arableland_0T = a, b
c, d = read_df("API_NV.AGR.TOTL.ZS_DS2_en_csv_v2_5359510.csv")
df_AFFV_0, df_AFFV_0T = c, d

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
# the cluster tools provided
df_cluster_norm, df_min, df_max = ct.scaler(df_cluster)

# determining the silhouette values
findsilhouette(df_cluster_norm)

# plotting the clusters with the best
# k value(decided using the silhouette values found)
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
# labels, title, legend etc

plt.xlabel("Arable land")
plt.ylabel("Agriculture, forestry, and fishing, value  added (% of GDP)")
plt.legend(handles=plt.scatter(df_cluster_norm["Arable Land"],
                               df_cluster_norm["AFFV"], 12, labels,
                               marker="o", cmap=colors).legend_elements()[0],
           labels=range(1, kmeans.n_clusters+1))
plt.title("Clusters of Countries in 2010", fontweight='bold')
plt.savefig("Cluster of Countries in 2010.png", dpi=250, bbox_inches='tight')
plt.show()

# move the cluster centres to the original scale
cen_0 = ct.backscale(cen, df_min, df_max)
xcen_0 = cen_0[:, 0]
ycen_0 = cen_0[:, 1]
# cluster by cluster
plt.figure(figsize=(6, 6))
plt.scatter(df_cluster["Arable Land"], df_cluster["AFFV"], 12,
            labels, marker="o", cmap=colors)
plt.scatter(xcen_0, ycen_0, 50, "k", marker="d")
plt.xlabel("Arable land")
plt.ylabel("Agriculture, forestry, and fishing, value  added (% of GDP)")
plt.legend(handles=plt.scatter(df_cluster_norm["Arable Land"],
                               df_cluster_norm["AFFV"], 12, labels,
                               marker="o", cmap=colors).legend_elements()[0],
           labels=range(1, kmeans.n_clusters+1))
plt.title("Clusters of countries with original centres", fontweight='bold')
plt.show()


# editing the dataframes to fit the data
df_AFFV_0T["Years"] = df_AFFV_0T.index
df_AFFV_0T["Years"] = pd.to_numeric(df_AFFV_0T["Years"], errors='coerce')
df_AFFV_0T["China"] = pd.to_numeric(df_AFFV_0T["China"], errors='coerce')

plt.figure()
df_AFFV_0T.plot("Years", "China")
plt.show()

# fitting using the exponential function
df_AFFV_0T["Exponential fit"] = exponential(df_AFFV_0T["Years"],
                                            40.1752942, -0.0260827510540983)

# fitting using the logistic function
df_AFFV_0T["Logistic fit"] = logistic(df_AFFV_0T["Years"],
                                      41, -0.09, 1985)

# plotting the fitted data
plt.figure()
df_AFFV_0T.plot("Years", ["Exponential fit", "Logistic fit", "China"],
                color=['r', 'b', 'g'])
plt.title("Fitted data", fontweight='bold')
plt.xlabel("Years")
plt.ylabel("Agriculture, forestry, and fishing, \n value  added (% of GDP)")
plt.legend(["Exponential fitting", "Logarithmic fitting", "China's data"])
plt.savefig("Fitted datas.png", dpi=250, bbox_inches='tight')
plt.show()

# forecast using logistic fitting for prediction
nyears = np.arange(1960, 2031)
forecast = logistic(nyears, 41, -0.09, 1985)
plt.plot(df_AFFV_0T["Years"], df_AFFV_0T["China"], label='China', color='b')
plt.plot(nyears, forecast, label='forecast', color='magenta')
plt.title("Prediction graph", fontweight='bold')
plt.xlabel("Years")
plt.ylabel("Agriculture, forestry, and fishing, \n value  added (% of GDP)")
plt.legend(["China's data", "Forecasted graph"])
plt.legend()
plt.savefig("Prediction.png", dpi=250, bbox_inches='tight')
plt.show()

# found sigma values I found using bootstrap resampling cause
# it didn't work out the other way
ra1 = np.array([0.0349, 0.0362, 0.0339])
sigma = np.sqrt(ra1)

# using error function anyway
low, up = err.err_ranges(nyears, logistic, [41, -0.09, 1985], sigma)

# plotting :(
plt.plot(df_AFFV_0T["Years"], df_AFFV_0T["China"], label='China')
plt.plot(nyears, forecast, label='forecast')
plt.fill_between(nyears, low, up, color="yellow", alpha=0.7)
plt.legend()
plt.show()
