# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 18:03:36 2020

grouping module

@author: Katsuhisa Morita
"""

import pandas as pd
import numpy as np
import scipy.stats as stats

def group_sep(df, target="", method="median", iqrco=1, alpha=0.05):
    """
    df : input dataframe contains target value
    
    target : is used to distinguish group
    
    method : {"IQR" : median + iqrco * IQR,
              "median" : is upper than median,
              "grubbs" : smirnov-grubbs test}
    
    """
    
    if len(target) != 0:
        target_list = list(df.iloc[target,:])
        outliers = select_outlier(target_list,method=method,iqrco=iqrco,alpha=alpha)
        df.loc["group"] = outliers
        return df
    else:
        df_res = pd.DataFrame(columns=df.columns)
        for i in df.index:
            df_res.loc[i,:] = select_outlier(list(df.loc[i,:]),iqrco=iqrco,alpha=alpha)
        return df_res

def select_outlier(lis,method="median",iqrco=1,alpha=0.05):
    
    if method == "IQR":
        res = select_outlier_iqr(lis,iqrco=iqrco)
    elif method == "median":
        res = select_outlier_median(lis)
    elif method == "grubbs":
        res = select_outlier_grubbs(lis,alpha)
    else:
        print("Input method is not implemented")
    
    return res
    
def select_outlier_iqr(lis,iqrco=1):
    q75, q25 = np.percentile(lis, [75 ,25])
    iqr = q75 - q25
    upper = q75 + iqr * iqrco
    res = [1 if i>upper else 0 for i in lis]
    return res

def select_outlier_median(lis):
    med = np.median(lis)
    res = [1 if i>med else 0 for i in lis]
    return res

def select_outlier_grubbs(lis,alpha=0.05):
    x = list(lis)
    while True:
        n = len(x)
        t = stats.t.isf(q=(alpha / n) / 2, df=n - 2)
        tau = (n - 1) * t / np.sqrt(n * (n - 2) + n * t * t)
        i_max = np.argmax(x)
        myu, std = np.mean(x), np.std(x, ddof=1)
        tau_far = np.abs((x[i_max] - myu) / std)
        if tau_far < tau:
            break
        x.pop(i_max)
    i_max = x[i_max]
    res = [1 if i>i_max else 0 for i in lis]
    return res