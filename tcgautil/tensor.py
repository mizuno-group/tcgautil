# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:05:34 2019

Handler of tensor generated from matrices

@author: tadahaya
"""
import tensorly as tl
import numpy as np
import pandas as pd
from tensorly.decomposition import tucker

class TDFE():
    def __init__(self):
        self.X = np.array([[],[]])
        self.decomposed = dict()
        self.data = dict()
        self.ndata = 0
        self.res = dict()
        self.idx = []
        self.sample_name = []

        
    def generate_tensor(self,data_list:list,dump=True):
        """
        generate a tensor from matrices

        Parameters
        ----------
        data_list: list
            a list of matrices with common samples in row and features in column
            CAUTION: so far, only 2 matrices are accepted
            
        dump: boolean
            indicate whether summation is calculated to dump the generated tensor

        """
        # normalize features by L2 norm before composition
        data_list = [self._ts_norm(x) for x in data_list]
        try:
            self.sample_name = list(data_list[0].index)
            self.idx = list(map(lambda x: list(x.columns),data_list))
            data_list = list(map(lambda x: x.values,data_list))
        except AttributeError:
            pass
        for i,v in enumerate(data_list):
            self.data[i] = v
        self.ndata = len(data_list)
        l_shape = [v.shape for v in data_list]
        ndata = set([v[0] for v in l_shape])
        if len(ndata)==1:
            ndata = ndata.pop()
        else:
            raise ValueError("!! Data length (row) should be the same !!")
        n_feats = [v[1] for v in l_shape]
        tpl = tuple([ndata] + n_feats)
        if len(tpl)!=3:
            raise NotImplementedError
        else:
            if dump:
                X = np.zeros((n_feats[0],n_feats[1]))
                for i in range(ndata):
                    X += np.c_[self.data[0][i]].dot(np.c_[self.data[1][i]].T)
            else:
                X = self._init_X(tpl)
                for i in range(ndata):
                    X[i] = np.c_[self.data[0][i]].dot(np.c_[self.data[1][i]].T)
        self.X = X
        return X


    def decompose(self,X=None):
        """
        decompose generated tensor with SVD

        """
        if X is None:
            X = self.X
        U,Lmd,Vh = np.linalg.svd(X,full_matrices=False)
        self.decomposed[0] = U
        self.decomposed[1] = Vh.T
        return U, Vh.T
                
            
    def get_svector(self):
        """
        get sample singular vectors

        """
        for i in range(self.ndata):
            temp = self.data[i].dot(self.decomposed[i])
            col = ["PC{}".format(j + 1) for j in range(temp.shape[1])]
            temp = pd.DataFrame(temp,index=self.sample_name,columns=col)
            self.res[i] = temp
        return self.res


    def conv_matrix(self,value:list,idx:list,dim_label:list):
        """
        convolution of n-mode expanded matrix into a tensor
        based on the indicated dimension labels
        
        Parameters
        ----------
        value: list
            a list of values of the expanded mode
            such as expression data in omics data
        
        idx: list
            a list of indices that contain whole dimension labels
        
        dim_label: list
            a list of lists that contain labels of each dimension


        Returns
        ----------
        tensor: tensor
            numpy tensor object

        label dict: nested list
            for checking data labels

        """
        # sort index
        lst = idx.copy()
        for dl in dim_label[::-1]:
            temp = []
            for v in dl:
                temp2 = []
                for l in lst:
                    if v in l:
                        temp2.append(l)
                temp += temp2
            lst = temp    
            
        # generate a tensor
        dic = dict(zip(idx,value))
        sorted_val = [dic[v] for v in lst]
        len_dim = [len(v) for v in dim_label]
        tshape = len_dim + [len(value[0])]
        tensor = np.array(sorted_val).reshape(tuple(tshape))

        # generate a label dict
        def splitter(l:list,n:int):
            for idx in range(0,len(l),n):
                yield l[idx:idx + n]
        
        for v in len_dim[::-1]:
            lst = list(splitter(lst,v))

        return tensor,lst[0]


    def _ts_norm(self,X):
        return X/np.linalg.norm(X,axis=0)

    
    def _init_X(self,tpl):
        return np.array(np.zeros(shape=tpl))
