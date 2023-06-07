# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 18:05:27 2020

@author: Katsuhisa Morita
"""

# import
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from scipy.stats import rankdata

# modules
from . import DEG_extraction as deg

## ここどうしよう
import deconv as dec


def deconvolution(df_mix="",df_ref="",DEGnumber=150,z=True,pretrimming=True,sep="_",num=10):
    # 既にz scoreになっているものについて例外処理
    if df_mix.min().min() < 0:
        print("mixture data is already z score")
        if pretrimming:
            df_mix, df_ref = intersection_index(df_mix,df_ref)
            df_ref = create_ref(df_ref,number=DEGnumber,sep=sep)
            df_ref = 2 ** df_ref
            df_ref = standardz_sample(df_ref)
            df_mix = quantile(df_mix,method="median")
            df_mix = standardz_sample(df_mix)
        
    else:
        if pretrimming:
            df_mix = trimming(df_mix)           
            df_mix, df_ref = intersection_index(df_mix,df_ref)

        df_ref = create_ref(df_ref,number=DEGnumber,sep=sep)

        # processing
        if z:
            df_ref = 2 ** df_ref
            df_ref = standardz_sample(df_ref)

            df_mix = 2 ** df_mix
            df_mix = quantile(df_mix,method="median")
            df_mix = standardz_sample(df_mix)
        else:
            df_mix = quantile(df_mix,method="median")
        
    # deconvolution
    dat = dec.deconv.Deconvolution()
    dat.do_fit(file_dat=df_mix,file_ref=df_ref,method='NuSVR',combat=False,max_iter=1000000,number_of_repeats=num)
    res = dat.get_res()
    return res.T


def standardz_sample(x):
    pop_mean = x.mean(axis=0)
    pop_std = x.std(axis=0)
    df = (x - pop_mean).divide(pop_std)
    df = df.replace(np.inf,np.nan)
    df = df.replace(-np.inf,np.nan)
    df = df.dropna()
    print('standardz population control')
    return df

def create_ref(df,number=150,sep="_",limit_CV=1):
    df = np.log2(df+1)
    cluster, a = sepmaker(df,sep=sep)
    dat = deg.reference()
    dat.prepare(df,sep=cluster)
    dat.DEG_extraction(method='ttest',number=number,q_limit=0.1,limit_CV=1)
    res = dat.get_res()   
    df = df.loc[res,:]
    df = df_median(df,sep=sep)
    return df

def df_median(df,sep="_"):
    df.columns=[i.split(sep)[0] for i in list(df.columns)]
    df = df.groupby(level=0,axis=1).median()
    return df

def intersection_index(df,df2):
    df.index = [i.upper() for i in df.index]
    df2.index = [i.upper() for i in df2.index]
    ind = list(set(df.index) & set(df2.index))
    df = df.loc[ind,:]
    df2 = df2.loc[ind,:]    
    return df,df2

def sepmaker(df,sep='.',position=0):
    samples = list(df.columns)
    sample_unique = []
    seps=[]
    ap1 = sample_unique.append
    ap2 = seps.append
    for i in samples:
        if i.split(sep)[position] in sample_unique:
            number = sample_unique.index(i.split(sep)[position])
            ap2(number)
        else:
            ap1(i.split(sep)[position])
            ap2(len(sample_unique)-1)
    return seps, sample_unique

def array_imputer(df,threshold=0.9,strategy="median",trim=1.0):
    df = df.replace(0,np.nan)
    thresh = int(threshold*float(len(list(df.columns))))
    df = df.dropna(thresh=thresh)
    imr = SimpleImputer(strategy=strategy)
    imputed = imr.fit_transform(df.values.T) # impute in columns
    df_res = pd.DataFrame(imputed.T,index=df.index,columns=df.columns)
    print("strategy: {}".format(strategy))
    return df_res


def trimming(df):
    # same index median
    df.index = [str(i) for i in df.index]
    df2 = pd.DataFrame()
    dup = df.index[df.index.duplicated(keep="first")]
    gene_list = pd.Series(dup).unique().tolist()
    if len(gene_list) != 0:
        for gene in gene_list:
            new = df.loc[gene].median()
            df2[gene] = new
        df = df.drop(gene_list)
        df = pd.concat([df,df2.T])

    if len(df.T) != 1:    
        df = array_imputer(df)
    else:
        df = df.where(df>1)
        df = df.dropna()
        print("Trimming finished")

    return df

def quantile(df,method="median"):
    """
    quantile normalization of dataframe (variable x sample)
    
    Parameters
    ----------
    df: dataframe
        a dataframe subjected to QN
    
    method: str, default "median"
        determine median or mean values are employed as the template    

    """
    #print("quantile normalization (QN)")
    df_c = df.copy() # deep copy
    lst_index = list(df_c.index)
    lst_col = list(df_c.columns)
    n_ind = len(lst_index)
    n_col = len(lst_col)

    ### prepare mean/median distribution
    x_sorted = np.sort(df_c.values,axis=0)[::-1]
    if method=="median":
        temp = np.median(x_sorted,axis=1)
    else:
        temp = np.mean(x_sorted,axis=1)
    temp_sorted = np.sort(temp)[::-1]

    ### prepare reference rank list
    x_rank_T = np.array([rankdata(v,method="ordinal") for v in df_c.T.values])

    ### conversion
    rank = sorted([v + 1 for v in range(n_ind)],reverse=True)
    converter = dict(list(zip(rank,temp_sorted)))
    converted = []
    converted_ap = converted.append  
    for i in range(n_col):
        transient = [converter[v] for v in list(x_rank_T[i])]
        converted_ap(transient)
    np_data = np.matrix(converted).T
    df2 = pd.DataFrame(np_data)
    df2.index = lst_index
    df2.columns = lst_col
    return df2