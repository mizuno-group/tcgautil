# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 16:37:19 2020

Processing module

@author: Katsuhisa Morita
"""
import collections
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from scipy.stats import rankdata
from combat.pycombat import pycombat
import codecs


class processing():
    def __init__(self):
        self.__transcriptome_data = pd.DataFrame()
        self.__clinical_data = pd.DataFrame()
        self.__prognosis_data = pd.DataFrame()
    
    # setter / getter    

    def set_data(self, transcriptome_data, clinical_data):
        self.__transcriptome_data = transcriptome_data
        self.__clinical_data = clinical_data
        return

    def get_data(self):
        return self.__transcriptome_data, self.__clinical_data, self.__prognosis_data
        
    # main
    def processing(self, file_ann, quantile=True, combat=True, cutoff=1.0, intersection=True, batch_target=["plate","gender"], par_prior=False):
        """
        Processing of Transcriptome
        
        (1) trim/imputation
        (2) quantile normalization
        
        if combat:
            (3) batch correction (ComBat) by pycombat
            (4) trimming (second)
            (5) quantile normalization
        
        (Last) annotation gene
        
        Processing of Prognosis data
        
        generate (Overall Survival Time, Overall Survival Status) matrix from clinical data matrix
        
        """
        
        transcriptome_data = self.__transcriptome_data
        clinical_data = self.__clinical_data
        
        ### Transcriptome data processing
        # FPKM to TPM conversion
        transcriptome_data = self.__fpkm2tpm(transcriptome_data)
        # Imputing
        transcriptome_data = self.__array_imputer(transcriptome_data,cutoff=cutoff)
        """ 210910 remove this QN step """
        # Quantile Normalization
        #if quantile:
            #transcriptome_data = self.__quantile(transcriptome_data)
        #else:
            #print("Not quantile normalized")
        # Combat correction
        if combat:
            clinical_data = clinical_data.dropna(axis=0, subset=batch_target)
            temp, clinical_data = self.__index_intersection(transcriptome_data.T, clinical_data)
            transcriptome_data = temp.T
            for i in batch_target:
                print("Conducting ComBat : {}".format(i))
                batch_list = self.__batchmaker(list(clinical_data.loc[:,i]))
                temp = pycombat(transcriptome_data, batch_list, par_prior=par_prior)
                if len(temp)==0:
                    print("combat error {}".format(i))
                else:
                    transcriptome_data = temp
                    del temp
            # Imputing
            transcriptome_data = self.__array_imputer(transcriptome_data,cutoff=cutoff)
            # Quantile Normalization
            if quantile:
                transcriptome_data = self.__quantile(transcriptome_data)            
        else:
            print("Not ComBat correction")
            
        if len(file_ann)>0:
            transcriptome_data = self.__gene_annotation(transcriptome_data, file_ann)
        else:
            print("gene ref file is not assigned")
        
        ### clinical data processing
        prognosis_data = self.__prognosis_processing(clinical_data)
        prognosis_data = prognosis_data.loc[prognosis_data.loc[:,"OS_Time"]!=0,:]
        
        ### take intersection index
        if intersection:
            temp, prognosis_data = self.__index_intersection(transcriptome_data.T, prognosis_data)
            transcriptome_data = temp.T
        
        ### return
        self.__transcriptome_data = transcriptome_data
        self.__clinical_data = clinical_data
        self.__prognosis_data = prognosis_data
    
    def processing2(self, file_ann, quantile=True, combat=True, cutoff=1.0, intersection=True, batch_target=["plate","gender"], par_prior=False):
        """
        Processing of Transcriptome
        
        (1) trim/imputation
        
        if combat:
            (2) batch correction (ComBat) by pycombat
            (3) trimming (second)
            (4) quantile normalization
        
        (Last) annotation gene
        
        Processing of Prognosis data
        
        generate (Overall Survival Time, Overall Survival Status) matrix from clinical data matrix
        
        """
        
        transcriptome_data = self.__transcriptome_data
        clinical_data = self.__clinical_data
        
        ### Transcriptome data processing
        # FPKM to TPM conversion
        transcriptome_data = self.__fpkm2tpm(transcriptome_data)
        # Imputing
        transcriptome_data = self.__array_imputer(transcriptome_data,cutoff=cutoff)
        """ 210910 remove this QN step """
        # Quantile Normalization
        #if quantile:
            #transcriptome_data = self.__quantile(transcriptome_data)
        #else:
            #print("Not quantile normalized")
        # Combat correction
        if combat:
            clinical_data = self.__batch_selection(batch_target=batch_target,threshold=2)
            temp, clinical_data = self.__index_intersection(transcriptome_data.T, clinical_data)
            transcriptome_data = temp.T
            for i in batch_target:
                print("Conducting ComBat : {}".format(i))
                batch_list = self.__batchmaker(list(clinical_data.loc[:,i]))
                temp = pycombat(transcriptome_data, batch_list, par_prior=par_prior)
                if len(temp)==0:
                    print("combat error {}".format(i))
                else:
                    transcriptome_data = temp
                    del temp
            # Imputing
            transcriptome_data = self.__array_imputer(transcriptome_data,cutoff=cutoff)
            # Quantile Normalization
            if quantile:
                transcriptome_data = self.__quantile(transcriptome_data)   
        else:
            print("Not ComBat correction")
        if len(file_ann)>0:
            transcriptome_data = self.__gene_annotation(transcriptome_data, file_ann)
        else:
            print("gene ref file is not assigned")
        
        
        ### clinical data processing
        prognosis_data = self.__prognosis_processing(clinical_data)
        prognosis_data = prognosis_data.loc[prognosis_data.loc[:,"OS_Time"]!=0,:]
        
        ### take intersection index
        if intersection:
            temp, prognosis_data = self.__index_intersection(transcriptome_data.T, prognosis_data)
            transcriptome_data = temp.T
        
        ### return
        self.__transcriptome_data = transcriptome_data
        self.__clinical_data = clinical_data
        self.__prognosis_data = prognosis_data
            
    # method
    def __fpkm2tpm(self, data):
        """
        convert fpkm of opened TCGA data into tpm
        index: Ensemble Gene ID
        column: File ID
        
        Parameters
        ----------
        data: dataframe
            dataframe after "opener"
            
        Returns
        ----------
        res: TPM file
        
        """
        x = data.values
        sums = np.c_[np.sum(x,axis=0)].T
        return pd.DataFrame(1000000*x/sums,index=data.index,columns=data.columns)
    
    def __array_imputer(self,df,cutoff=1.0,threshold=0.9,strategy="median"):
        """
        imputing nan and trim the values less than 1
        
        Parameters
        ----------
        df: a dataframe
            a dataframe to be analyzed
        
        cutoff: float
            determine cutoff value of expression
        
        threshold: float, default 0.9
            determine whether imupting is done or not dependent on ratio of not nan
            
        strategy: str, default median
            indicates which statistics is used for imputation
            candidates: "median", "most_frequent", "mean"
        
        Returns
        ----------
        res: a dataframe
        
        """
        df = df.where(df > cutoff)
        df = df.replace(0,np.nan)
        thresh = int(threshold*float(len(list(df.columns))))
        df = df.dropna(thresh=thresh)
        imr = SimpleImputer(strategy=strategy)
        imputed = imr.fit_transform(df.values.T) # impute in columns
        df_res = pd.DataFrame(imputed.T,index=df.index,columns=df.columns)
        return df_res
    
    def __quantile(self, df, method="median"):
        """
        quantile normalization of dataframe (variable x sample)
        
        Parameters
        ----------
        df: dataframe
            a dataframe subjected to QN
        
        method: str, default "median"
            determine median or mean values are employed as the template    
        """
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
    
    def __index_intersection(self,df,df2):
        df_ind = list(df.index)
        df2_ind = list(df2.index)
        new_ind = list(set(df_ind)&set(df2_ind))
        
        df = df.loc[new_ind,:]
        df2 = df2.loc[new_ind,:]
        return df, df2
    
    def __gene_annotation(self, df, file_ann):
        #print(df)
        df.index = [str(i) for i in df.index]
        # same gene median
        df = self.__samegene_median(df)
        
        # annotation
        with codecs.open(file_ann, "r", "Shift-JIS", "ignore") as file:
            df_ref = pd.read_csv(file, dtype=str)
        df_ref.index = [str(i) for i in df_ref.index]
        lst = list(zip(list(df_ref.iloc[:,0]), list(df_ref.iloc[:,1])))
        dic = dict(lst)
        index_new = []
        a = index_new.append
        for i in list(df.index):
            try:
                a(dic.get(i,"nan"))
            except:
                a("nan")
        df.index = index_new
        df["index"] = list(df.index)
        try:
            df = df.drop("nan")
        except:
            pass
        try:
            df = df.dropna(how='any',axis=0)
        except:
            pass
        
        # same gene median
        df = self.__samegene_median(df)
        return df
        
    def __samegene_median(self, df):
        print(df)
        df2 = pd.DataFrame()
        dup = df.index[df.index.duplicated(keep="first")]
        gene_list = pd.Series(dup).unique().tolist()
        if len(gene_list) != 0:
            for gene in gene_list:
                new = df.loc[gene].median()
                df2[gene] = new
            df = df.drop(gene_list)
            df = pd.concat([df,df2.T])
        return df
    
    def __batch_selection(self,batch_target=['plate','gender'],threshold=2):
        """
        remove samples which hold rare batch
        """
        clinical_data = self.__clinical_data
        clinical_data = clinical_data.dropna(axis=0, subset=batch_target) # remove missing sample
        final_ban_idx = set()
        for t in batch_target:
            batch_list = clinical_data[t].tolist()
            l = list(collections.Counter(batch_list).items()) # ex : [('white', 141), ('asian', 5), ('not reported', 1)]
            print(t,":",l)
            ban_batch = []
            for x in l:
                if x[1] < threshold:
                    ban_batch.append(x[0])
                else:
                    pass
            ban_idx = clinical_data[clinical_data[t].isin(ban_batch)].index.tolist()
            final_ban_idx = final_ban_idx.union(ban_idx)
        final_target_idx = list(set(clinical_data.index)-final_ban_idx)
        final_target_clinical = clinical_data.loc[final_target_idx]
        print(len(final_ban_idx),"was removed because of its batch size")
        print("")
        return final_target_clinical
        
    def __batchmaker(self, lis):
        """
        genarate batch list from str list
        """
        sample_unique = []
        batch = []
        ap1 = sample_unique.append
        ap2 = batch.append
        for i in lis:
            if i in sample_unique:
                number = sample_unique.index(i)
                ap2(number)
            else:
                ap1(i)
                ap2(len(sample_unique)-1)
        return batch
    
    def __prognosis_processing(self,clinical_data):
        prognosis = pd.DataFrame(index=clinical_data.index, columns=["OS_Time","OS_Status"])
        prog_time = list()
        prog_status = list()
        TF = list()
        
        status = list(clinical_data.loc[:,"status"])
        death = list(clinical_data.loc[:,"days2death"])
        follow = clinical_data.loc[:,"days2follow"]
        
        
        for i in range(len(clinical_data.index)):
            if status[i]=="Alive" or status[i]=="Not Reported":
                try:
                    prog_time.append(float(follow[i]))
                    prog_status.append(0)
                    TF.append(True)
                except:
                    TF.append(False)
                    
            if status[i]=="Dead":
                try:
                    prog_time.append(float(death[i]))
                    prog_status.append(1)
                    TF.append(True)
                except:
                    TF.append(False)
        prognosis = prognosis.loc[TF,:]
        prognosis["OS_Time"] = [float(i) for i in prog_time]
        prognosis["OS_Status"] = prog_status
        print("drop {} samples by clinical data missing".format(len(TF)-sum(TF)))
        return prognosis