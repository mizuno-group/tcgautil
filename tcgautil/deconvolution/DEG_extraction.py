# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 18:11:03 2020

@author: Katsuhisa Morita
"""

import numpy as np
import pandas as pd
from scipy import stats as st
import statsmodels.stats.multitest as sm

class reference():
    ### init ###
    def __init__(self):
        self.df_all=pd.DataFrame()
        self.df_target=pd.DataFrame()
        self.df_else=pd.DataFrame()
        self.df_logFC=pd.DataFrame()
        self.df_CV=pd.DataFrame()
        self.seps=[]
        self.__pickup_genes=[]
        self.__pickup_genes_df=pd.DataFrame()
        self.__method_dict={'ttest':self.__DEG_extraction_qval}
    
    ### main ###    
    def prepare(self,df,sep=[0,0,0,1,1,1,2,2,2]):
        """
        df : dataframe or file path
        sep : sample separation by conditions
        
        """
        self.__set_df_all(df)
        self.__make_seplist(sep=sep)
        self.__same_gene_median()
        
        
    def DEG_extraction(self,method='ttest',number=50,q_limit=0.1,limit_CV=0.3):
        self.__pickup_genes = []
        pickup_genes_list = []
        ap = pickup_genes_list.append
        for i,sep in enumerate(self.seps):
            print(i)
            self.__df_separate(sep)
            self.__logFC()
            self.__calc_CV()
            self.__method_dict[method](q_limit=q_limit)
            pickup_genes = self.__selection(number=number,limit_CV=limit_CV)
            ap(pickup_genes)
        self.__pickup_genes_df=pd.DataFrame(pickup_genes_list).T
    
    def condition_number(self,loop_range=[50,200],method='ttest',q_limit=0.1,limit_CV=0.3,columns_sep='.'):        
        res_list = []
        ap = res_list.append
        for i,sep in enumerate(self.seps):
            print(i)
            self.__df_separate(sep)
            self.__logFC()
            self.__calc_CV()
            self.__method_dict[method](q_limit=q_limit)
            pickup_genes_list = self.__selection_loop(loop_range=loop_range,limit_CV=limit_CV)
            ap(pickup_genes_list)
        
        condition_numbers = []
        ap = condition_numbers.append
        best_number=0
        for l in range(len(range(loop_range[0],loop_range[1]+1))):
            pickup_genes = []
            for v in res_list:
                pickup_genes = pickup_genes + v[l]
            
            pickup_genes = [i for i in pickup_genes if str(i)!='nan']
            pickup_genes = list(set(pickup_genes))
            
            df_res = self.df_all.loc[pickup_genes,:]
            df_res.columns=[i.split(columns_sep)[0] for i in list(df_res.columns)]
            df_res = df_res.groupby(level=0,axis=1).median()
            condition_number = np.linalg.cond(df_res)
            if len(condition_numbers) == 0:
                best_number=range(loop_range[0],loop_range[1]+1)[l]
            else:    
                if min(condition_numbers) > condition_number:
                    best_number=range(loop_range[0],loop_range[1]+1)[l]
            ap(condition_number)
        print('pickup number = {}'.format(best_number))
        self.DEG_extraction(method=method,number=best_number,q_limit=q_limit,limit_CV=limit_CV)
        return condition_numbers
    
    ### in/out put ###
    def get_res(self):
        self.__pickup_genes=[i for i in self.__pickup_genes if str(i)!='nan']
        self.__pickup_genes=list(set(self.__pickup_genes))
        return self.__pickup_genes
    
    def get_res_df(self):
        return self.__pickup_genes_df
    
    def __set_df_all(self,df):
        # input df or file path
        try:
            df = pd.read_csv(df,index_col=0)
        except:
            pass
        
        if min(df.min())==0:
            df = df+1
        else:
            pass
        # 対数が取っていない場合対数を取る
        if max(df.max()) > 100:
            df = np.log2(df)
        else:
            pass
        self.df_all = df
        
    ### DEG method ###
    def __DEG_extraction_qval(self,q_limit=0.1,**kwargs):
        p_vals = [st.ttest_ind(self.df_target.iloc[i,:],self.df_else.iloc[i,:],equal_var=False)[1] for i in range(len(self.df_target.index))]
        p_vals = [float(str(i).replace('nan','1')) for i in p_vals]
        q_vals = sm.multipletests(p_vals, alpha=0.1, method='fdr_bh')[1]
        TF=[True if i<q_limit else False for i in list(q_vals)]
        print("extracted genes number = {}".format(TF.count(True)))
        self.df_target=self.df_target.loc[TF,:]
        self.df_else=self.df_else.loc[TF,:]
        
        return
        
    ### processing ###
    def sepmaker(self,df,delimiter='.'):
        self.__set_df_all(df)
        df = self.df_all
        samples = list(df.columns)
        sample_unique = []
        seps=[]
        ap1 = sample_unique.append
        ap2 = seps.append
        for i in samples:
            if i.split(delimiter)[0] in sample_unique:
                number = sample_unique.index(i.split(delimiter)[0])
                ap2(number)
            else:
                ap1(i.split(delimiter)[0])
                ap2(len(sample_unique)-1)
                
        return seps, sample_unique

    def __df_separate(self,sep):
        df = self.df_all
        df.columns=[str(i) for i in sep]
        self.df_target=df.loc[:,df.columns.str.contains('1')]
        self.df_else=df.loc[:,df.columns.str.contains('0')]
        
    def __make_seplist(self,sep=[0,0,0,1,1,1,2,2,2]):
        res = [[0 if v!=i else 1 for v in sep] for i in list(range(max(sep)+1))]
        self.seps=res  

    def __logFC(self):
        # calculate df_target / df_else logFC
        df_logFC = self.df_target.T.median() - self.df_else.T.median()
        df_logFC = pd.DataFrame(df_logFC)
        self.df_logFC = df_logFC
    
    def __calc_deviation(self):
        df_CV = pd.DataFrame(index=self.df_target.index)
        df_CV.loc[:,'CV'] = st.variation(self.df_target.T)
        df_CV = df_CV.replace(np.inf,np.nan)
        df_CV = df_CV.replace(-np.inf,np.nan)
        df_CV = df_CV.dropna()
        self.df_CV=df_CV
    
    def __calc_CV(self):
        df_CV = np.std(self.df_target.T) / np.mean(self.df_target.T)
        df_CV.index = self.df_target.index
        df_CV = df_CV.replace(np.inf,np.nan)
        df_CV = df_CV.replace(-np.inf,np.nan)
        df_CV = df_CV.dropna()
        self.df_CV=pd.DataFrame(df_CV)
        
    def __intersection(self):
        lis1 = list(self.df_logFC.index)
        lis2 = list(self.df_CV.index)
        self.df_logFC = self.df_logFC.loc[list(set(lis1)&set(lis2)),:]
        self.df_CV = self.df_CV.loc[list(set(lis1)&set(lis2)),:]

    def __same_gene_median(self):
        df = self.df_all
        df2 = pd.DataFrame()
        dup = df.index[df.index.duplicated(keep="first")]
        gene_list = pd.Series(dup).unique().tolist()
        if len(gene_list) != 0:
            for gene in gene_list:
                new = df.loc[gene].median()
                df2[gene] = new
        df = df.drop(gene_list)
        df = pd.concat([df,df2.T])
        self.df_all = df

    def __selection(self,number=50,limit_CV=0.1):
        self.__intersection()
        df_logFC=self.df_logFC.sort_values(by=0,ascending=False)
        df_CV=self.df_CV.loc[list(df_logFC.index),:]
        genes=list(df_logFC.index)
        pickup_genes=[]
        ap = pickup_genes.append
        i=0
        while len(pickup_genes)<number:
            if len(genes)<i+1:
                pickup_genes = pickup_genes+[np.nan]*number
                print('not enough genes picked up')
            elif df_CV.iloc[i,0] < limit_CV and df_logFC.iloc[i,0] > 1:
                ap(genes[i])
            i+=1
        else:
            self.__pickup_genes = self.__pickup_genes + pickup_genes
            return pickup_genes

    def __selection_loop(self,loop_range=[50,200],limit_CV=0.1):
        self.__intersection()
        df_logFC=self.df_logFC.sort_values(by=0,ascending=False)
        df_CV=self.df_CV.loc[list(df_logFC.index),:]
        genes=list(df_logFC.index)
        
        pickup_genes_list = []
        ap_list=pickup_genes_list.append
        for v in range(loop_range[0],loop_range[1]+1):
            pickup_genes=[]
            ap = pickup_genes.append
            i=0
            while len(pickup_genes)<v:
                if len(genes)<i+1:
                    pickup_genes = pickup_genes+[np.nan]*v
                elif df_CV.iloc[i,0] < limit_CV and df_logFC.iloc[i,0] > 1:
                    ap(genes[i])
                i+=1
            else:
                ap_list(pickup_genes)
        return pickup_genes_list