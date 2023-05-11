# -*- coding: utf-8 -*-
"""
Created on Thu May 11 19:05:53 2023

@author: gobub
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import errors as err
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


def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and growth rate g."""
    t = t - 1960
    f = n0 * np.exp(g*t)
    
    return f


def logistic(t, a, b, c):
    """Fit it into a logistic curve""" 
    return a / (1 + np.exp(-b * (t - c)))

c, d = read_df("API_NV.AGR.TOTL.ZS_DS2_en_csv_v2_5359510.csv")
df_AFFV_0, df_AFFV_0T = c, d

df_AFFV_0T["Years"] = df_AFFV_0T.index
df_AFFV_0T["Years"] = pd.to_numeric(df_AFFV_0T["Years"], errors='coerce')
df_AFFV_0T["China"] = pd.to_numeric(df_AFFV_0T["China"], errors='coerce')


# plt.figure()
# df_AFFV_0T.plot("Years", "China")
# plt.show()


# # fitting with curve_fit
# # param, covar = opt.curve_fit(exponential,
#                               df_AFFV_0T["Years"], df_AFFV_0T["China"],
#                               p0=(23.1752942, -0.260827510540983), maxfev=500000)


# fitting using the exponential function
df_AFFV_0T["fit1"] = exponential(df_AFFV_0T["Years"],
                                40.1752942, -0.0260827510540983)

# fitting using the logistic function
df_AFFV_0T["fit2"] = logistic(df_AFFV_0T["Years"],41, -0.09, 1985)

plt.figure()                                
# df_AFFV_0T.plot("Years", ["fit1", "fit2", "China"])
# plt.show()

nyears = np.arange(1960, 2031)
forecast = logistic(nyears, 41, -0.09, 1985)
plt.plot(df_AFFV_0T["Years"], df_AFFV_0T["China"], label='China')
plt.plot(nyears, forecast, label='forecast')
plt.legend()
plt.show()

# found sigma values using bootstrap resampling
ra1 = np.array([0.0349, 0.0362, 0.0339])
sigma = np.sqrt(ra1)
low, up = err.err_ranges(nyears, logistic, [41, -0.09, 1985], sigma)

plt.plot(df_AFFV_0T["Years"], df_AFFV_0T["China"], label='China')
plt.plot(nyears, forecast, label='forecast')
plt.fill_between(nyears, low, up, color="yellow", alpha=0.7)
plt.legend()
plt.show()


