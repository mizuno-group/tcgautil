# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 15:36:08 2020

@author: Katsuhisa Morita
"""

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter as KMF
from lifelines import fitters as fit
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

from .load import load
from .processing import processing
from .deconvolution.deconvolution import deconvolution
from .deconvolution.group import group_sep
from .tensor import TDFE


class tcgautil():
    """
    Main Module to handle tcga data
    
    methods
    ----------
    # load #
    load data
        
    # processing #
    data processing for analysis
    
    # deconvolution #
    Estimate immune cell pupulation by deconvolution
    
    # add_immune #
    
    # plot_KM_immunes #
    
    ----------

    
    """
    
    def __init__(self):
        self.__transcriptome_data = pd.DataFrame()
        self.__transcriptome_data_sep = pd.DataFrame()
        self.__clinical_data = pd.DataFrame()
        self.__prognosis_data = pd.DataFrame()
        self.__metadata = pd.DataFrame()
        self.__immune_population = pd.DataFrame()
        self.__fit = pd.DataFrame()
        self.__tensor_rsm = pd.DataFrame()
        self.__tensor = TDFE()
        
    # setter / getter
    def set_data(self, transcriptome_data=None, clinical_data=None, prognosis_data = None):
        self.__transcriptome_data = transcriptome_data
        self.__clinical_data = clinical_data
        self.__prognosis_data = prognosis_data
    
    def set_path(self,path=None):
        try:
            self.__transcriptome_data = pd.read_csv(path+"/transcript.csv",index_col=0)
        except:
            pass
        try:
            self.__clinical_data = pd.read_csv(path+"/clinical.csv",index_col=0)
        except:
            pass
        try:
            self.__prognosis_data = pd.read_csv(path+"/prognosis.csv",index_col=0)
        except:
            pass
        try:
            self.__immune_population = pd.read_csv(path+"/immune.csv",index_col=0)
        except:
            pass
    
    def get_transcript(self):
        return self.__transcriptome_data
    
    def get_clinical(self):
        return self.__clinical_data

    def get_prognosis(self):
        return self.__prognosis_data

    def get_meta(self):
        return self.__metadata        
    
    def export_path(self,path=None):
        self.__transcriptome_data.to_csv(path+"/transcript.csv")
        self.__clinical_data.to_csv(path+"/clinical.csv")
        self.__prognosis_data.to_csv(path+"/prognosis.csv")
        self.__metadata.to_csv(path+"/metadata.csv")
        self.__immune_population.to_csv(path+"/immune.csv")
        self.__fit.to_csv(path+"/fit.csv")
        
    # main
    def load(self, transcript=None, meta=None, sample=None, clinical=None):
        """
        load files
        
        ----------
        parameter : transcriptome = folder path contains transcriptome .gz files
                    clinical = Path for clinical.tsv
                    sample = Path for samplesheet (tsv file)
                    meta = Path for Metadata (json file)

        
        ----------
        """
        
        dat = load()
        dat.load(transcript=transcript, meta=meta, sample=sample, clinical=clinical)
        self.__transcriptome_data, self.__clinical_data, self.__metadata = dat.get_data()
        
        
    def processing(self, file_ann="", quantile=True, combat=True, cutoff=1.0, intersection=True, batch_target=["gender","plate"], par_prior=False):
        """
        processing files

        ----------    
        parameter : quantile = boolean / transcriptome quantile normalization
                    combat = boolean / conduct combat correction
                    cutoff = float (>0) / trimming cutoff value 
                    intersection = boolean / take prognosis and transcriptome index intersection
                    batch_target = list / combat batch correction targeet like ["gender","plate"]
                    par_prior = boolean / combat parametric estimation
        
        ----------
        """
        dat = processing()
        dat.set_data(self.__transcriptome_data, self.__clinical_data)
        dat.processing(file_ann=file_ann,quantile=quantile,combat=combat,cutoff=cutoff,intersection=intersection,batch_target=batch_target,par_prior=par_prior)
        self.__transcriptome_data, self.__clinical_data, self.__prognosis_data = dat.get_data()
        
    
    def deconvolution(self, df_ref="", DEGnumber=150, z=True, pretrimming=True,sep="_", num=1):
        """
        Estimate immune cell pupulation by deconvolution

        ----------    
        parameter : df_ref = path or pd.DataFrame / reference files
                    DEGnumber = int / extract DEGnumber / cell genes
                    z = boolean / convert z score 
                    pretrimming = boolean / take prognosis and transcriptome index intersection
                    sep = str / ref's separation str
                    num = int / iter of deconvolution
        
        ----------        
        """
        # load df file
        try:
            df_ref = pd.read_csv(df_ref,index_col=0)
        except:
            df_ref = df_ref
        # deconvolution
        self.__immune_population = deconvolution(df_mix=self.__transcriptome_data,df_ref=df_ref,DEGnumber=DEGnumber,z=z,pretrimming=pretrimming,sep=sep,num=num)
    
    def tensor_dec(self):
        """
        conduct tensor decomposition (sample * immune pop * transcript)
        
        """
        rsm = self.__tensor_dec(self.__transcriptome_data, self.__immune_population)
        self.__tensor_rsm = rsm
        print("Decomposed")
        
    def add_immune(self, method="IQR", iqrco=0.5, alpha=0.05):
        """
        add immune outlier information to prognosis data from immune population
        
        conduct cph fitting of all immunes
        
        """

        immune_outlier = group_sep(self.__immune_population,method=method,iqrco=iqrco, alpha=alpha).T
        prognosis_data = self.__prognosis_data.loc[:,["OS_Time","OS_Status"]]
        prognosis_data, immune_outlier = self.__index_intersection(prognosis_data, immune_outlier)
        prognosis_data = pd.concat([prognosis_data, immune_outlier],axis=1,sort=False)
        
        # fitting
        cph = fit.coxph_fitter.CoxPHFitter()
        cph.fit(prognosis_data, duration_col="OS_Time", event_col="OS_Status")
        cph.print_summary()
        self.__fit = cph.summary
        self.__prognosis_data = prognosis_data
    
    def add_tensor(self, method="IQR", iqrco=0.5, alpha=0.05):
        tensor_outlier = group_sep(self.__tensor_rsm.T,method=method,iqrco=iqrco, alpha=alpha).T
        prognosis_data = self.__prognosis_data.loc[:,["OS_Time","OS_Status"]]
        prognosis_data, tensor_outlier = self.__index_intersection(prognosis_data, tensor_outlier)
        prognosis_data = pd.concat([prognosis_data, tensor_outlier],axis=1,sort=False)
        
        # fitting
        cph = fit.coxph_fitter.CoxPHFitter()
        cph.fit(prognosis_data, duration_col="OS_Time", event_col="OS_Status")
        cph.print_summary()
        self.__fit = cph.summary
        self.__prognosis_data = prognosis_data
    
    def gene_impact(self, genes=[], method="median", iqrco=0.5, alpha=0.05, plot=False):
        """
        survey single gene impact to survival

        genes : list of interested gene names
        """
        if len(self.__transcriptome_data_sep)==0:
            if len(genes)>0:
                transcriptome_data = copy.deepcopy(self.__transcriptome_data.loc[genes,:])
            else:
                transcriptome_data = copy.deepcopy(self.__transcriptome_data)
            transcript_outlier = group_sep(transcriptome_data, method=method).T
            if len(genes)==0:
                self.__transcriptome_data_sep = transcript_outlier
            del transcriptome_data
        else:
            transcript_outlier = copy.deepcopy(self.__transcriptome_data_sep.loc[:,genes])

        prognosis_data = self.__prognosis_data.loc[:,["OS_Time","OS_Status"]]
        prognosis_data, transcript_outlier = self.__index_intersection(prognosis_data, transcript_outlier)
        prognosis_data = pd.concat([prognosis_data, transcript_outlier],axis=1,sort=False)
        
        # return fitting values
        genes = list(transcript_outlier.columns)
        res_log = pd.DataFrame(index=range(3))
        for i in tqdm(range(len(genes))):
            try:
                df_temp = prognosis_data.iloc[:,[0,1,i+2]]
                df1 = df_temp[df_temp.iloc[:,2]==0]
                df2 = df_temp[df_temp.iloc[:,2]==1]

                # log-rank test
                results = logrank_test(df1["OS_Time"], df2["OS_Time"], df1["OS_Status"], df2["OS_Status"])
                p_log = float(results.summary["p"])

                # 一般化wilcoxon
                results = logrank_test(df1["OS_Time"], df2["OS_Time"], df1["OS_Status"], df2["OS_Status"], weightings = "wilcoxon")
                p_wil = float(results.summary["p"])

                # fleming-harrington 検定
                results = logrank_test(df1["OS_Time"], df2["OS_Time"], df1["OS_Status"], df2["OS_Status"], weightings = "fleming-harrington",p=0,q=1)
                p_fh = float(results.summary["p"])

                # fleming-harrington 検定
                #results = logrank_test(df1["OS_Time"], df2["OS_Time"], df1["OS_Status"], df2["OS_Status"], weightings = "fleming-harrington",p=0.5,q=0.5)
                #p_fhh = float(results.summary["p"])

                #res_log[list(df_all.columns)[i+2]] = [p_log,p_wil,p_fh,p_fhh]
                res_log[genes[i]] = [p_log,p_wil,p_fh]
            except:
                pass
            if plot:
                plot_once(df_temp,genes[i])
                plt.ylim(0,1)
                plt.show()
        return res_log



    def plot_KM(self, target=["Neutrophils"]):
        plot_curves(self.__prognosis_data, target=target)
        
    ### method
    # processing
    def __index_intersection(self,df,df2):
        df_ind = list(df.index)
        df2_ind = list(df2.index)
        new_ind = list(set(df_ind)&set(df2_ind))
        
        df = df.loc[new_ind,:]
        df2 = df2.loc[new_ind,:]
        return df, df2
    
    def __tensor_dec(self,transcriptome_data, immune_population):
        data_lst = [transcriptome_data.T, immune_population.T]
        X = self.__tensor.generate_tensor(data_lst)
        a, b = self.__tensor.decompose()
        res = self.__tensor.get_svector()
        res0 = pd.DataFrame(res[0])
        res0.columns = [i+"_transcript" for i in res0.columns]
        res1 = pd.DataFrame(res[1])
        res1.columns = [i+"_immune" for i in res1.columns]
        res = pd.concat([res0,res1],axis=1,sort=False)
        res.index = transcriptome_data.columns
        return res
    
### plot modules
def plot_curves(df,target=[""]):
    lst = ["OS_Time","OS_Status"]+target
    df = df.loc[:,lst]
    cph = fit.coxph_fitter.CoxPHFitter()
    cph.fit(df, duration_col="OS_Time", event_col="OS_Status")
    cph.print_summary()
    if len(target)==1:
        plot_once(df,target[0])
    else:
        cph.plot_partial_effects_on_outcome(target, all_list(target))
    plt.title("Kaplan-Meier Plot")
    plt.xlabel("Time (days)")
    plt.ylabel("Survival Rate (/)")
    plt.show()
    return

def plot_once(df,target):
    ax = None
    for i, group in df.groupby(target):
        kmf = KMF()
        kmf.fit(group['OS_Time'], event_observed=group['OS_Status'],
                label = target + ':' + str(i))
        if ax is None:
            ax = kmf.plot()
        else:
            ax = kmf.plot(ax=ax)

def all_list(lis):
    length = len(lis)
    number = length**2
    res = ["0"*(length-len(format(i,"b"))) + format(i,"b") for i in range(number)]
    res = [list(i) for i in res]
    return res