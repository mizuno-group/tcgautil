# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 16:35:31 2020

load module

@author: Katsuhisa Morita
"""

import pandas as pd
import numpy as np
import json
import csv
from pathlib import Path
import gzip

class load():
    # init
    def __init__(self):
        self.__transcriptome_data = pd.DataFrame()
        self.__clinical_data = pd.DataFrame()
        self.__metadata = pd.DataFrame()
        
    # setter
    
    # getter
    def get_data(self):
        return self.__transcriptome_data, self.__clinical_data, self.__metadata
    
    # main
    def load(self, transcript=None, meta=None, sample=None, clinical=None):
        """
        load files from indicated path
        
        and process to dataframe for easy handling
        
        """
        # load files
        transcriptome_data = self.__transcript_opener(transcript)
        meta_data, clinical_data = self.__summarize_meta(meta,sample,clinical)
        
        # converting index names
        transcriptome_data = self.__convert(transcriptome_data,meta_data,axis=1) # replace sample labeling
        
        # return data
        self.__transcriptome_data =  transcriptome_data
        self.__clinical_data = clinical_data
        self.__metadata = meta_data
        
    # methods    
    def __transcript_opener(self, url):
        """
        open TCGA data and generate FPKM count data
        index: Ensemble Gene ID
        column: File ID

        Parameters
        ----------
        url: str
            path of the folder after unpacking DL file
            
        Returns
        ----------
        res: FPKM file
        
        """
        print("file loading")
        lst = [(url + "\\" + v.name + "\\" + [w for w in v.glob("*.txt.gz")][0].name,v.name) for v in list(Path(url).glob("*")) if v.is_dir()]
        data = pd.DataFrame()
        for i,v in enumerate(lst):
            if i==0:
                with gzip.open(v[0], "rt", "utf_8") as f:
                    reader = csv.reader(f, delimiter="\t")
                    dat = pd.DataFrame([l for l in reader])
                data = pd.DataFrame(list(dat.iloc[:,1]),index=list(dat.iloc[:,0]),columns=[v[1]])
            else:
                with gzip.open(v[0], "rt", "utf_8") as f:
                    reader = csv.reader(f, delimiter="\t")
                    dat = pd.DataFrame([l for l in reader])
                    dat = pd.DataFrame(list(dat.iloc[:,1]),index=list(dat.iloc[:,0]),columns=[v[1]])
                data = pd.concat([data,dat],axis=1,join="inner")
        lst_ind = list(map(lambda x: x.split(".")[0],list(data.index)))
        data.index = lst_ind    
        print("completed")
        return data.astype("float")
    
    def __summarize_meta(self, metadata=None, sample=None, clinical=None):
        """
        prepare df for converting main data with metadata.cart.20XX-XX-XX.json and clinical.tsv
        
        """
        # metadata prep
        with open(metadata) as f:
            df = json.load(f)
        file_id = [df[i]["file_id"] for i in range(len(df))]
        barcode = [df[i]["associated_entities"][0]["entity_submitter_id"] for i in range(len(df))]
        case_id = [df[i]["associated_entities"][0]["case_id"] for i in range(len(df))]
        df_metadata = pd.DataFrame({"barcode":barcode,"case id":case_id},index=file_id)
        
        # samplesheet summary
        df_sample = pd.read_csv(sample,index_col=0,delimiter="\t")
        lst_id = list(df_sample.index)
        lst_type = list(df_sample["Sample Type"])
        id2type = dict(zip(lst_id,lst_type))
        
        # batch prep
        case2barcode = dict(zip(case_id,barcode))
        case2id = dict(zip(case_id,file_id))
        df_clinical = pd.read_csv(clinical,index_col=0,delimiter="\t")
        conved = [case2barcode[v] for v in list(df_clinical.index)]
        ids = [case2id[v] for v in list(df_clinical.index)]
        # sample type
        types = [id2type[v] for v in ids]
        # plate
        lst_plate = [v.split("-")[5] for v in conved]
        component = list(set(lst_plate))
        dic_plate = dict(zip(component,list(range(len(component)))))
        lst_plate = [dic_plate[v] for v in lst_plate]
        # center
        lst_center = [v.split("-")[6] for v in conved]
        component = list(set(lst_center))
        dic_center = dict(zip(component,list(range(len(component)))))
        lst_center = [dic_center[v] for v in lst_center]
        
        ### charactaristic data
        # age
        lst_age = [np.nan if "--" in str(v) else int(v) for v in list(df_clinical["age_at_index"])]
        # gender
        lst_gender = [0 if v=="male" else 1 for v in list(df_clinical["gender"])]
        # race
        lst_race = list(df_clinical["race"])
        # diagnosis
        lst_diag = list(df_clinical["primary_diagnosis"])
        # stage
        lst_stage = list(df_clinical["tumor_stage"])
        # site
        lst_site = list(df_clinical["site_of_resection_or_biopsy"])
        # vital
        lst_vital = list(df_clinical["vital_status"])
        # days to death
        lst_days = list(df_clinical["days_to_death"])
        lst_days = [v if v!="--" else np.nan for v in lst_days]
        # days to last follow up
        lst_follow = list(df_clinical["days_to_last_follow_up"])
        lst_follow = [v if v!="--" else np.nan for v in lst_follow]
        # export    
        df_clinical = pd.DataFrame({"age":lst_age,"center":lst_center,"gender":lst_gender,
                                    "plate":lst_plate,"race":lst_race,"diagnosis":lst_diag,
                                    "stage":lst_stage,"site":lst_site,"type":types,"fileID":ids,
                                    "status":lst_vital,"days2death":lst_days,
                                    "days2follow":lst_follow},index=conved)
        # keep unique index
        df_clinical = self.__unique_index(df_clinical)
        return df_metadata, df_clinical
    
    def __convert(self, before,ref,axis=1,separator=" /// ",position=0):
        """
        converts index or columns into new one
        returns converted dataframe
        Parameters
        ----------
        before: dataframe
            target dataframe with ID in index or columns
        ref: dataframe
            index: old, column 0: new
        axis: 0 or 1, default 1
            determine which to be converted, index or column
            0: index
            1: column
            
        separator: str, default " /// "
            separator for gene alias
            
        position: int, default 0
            position of symbol after separation
            
        """
        # conversion
        if axis==0:
            df = before.copy()
            l_id = list(ref.index)
            l_id = list(map(lambda x: str(x),l_id))
            l_symbol = list(ref.iloc[:,0])        
            l_symbol = list(map(lambda x: str(x),l_symbol))
            l_symbol2 = []
            ap = l_symbol2.append
            for v in l_symbol:
                spl = v.split(separator)
                if len(spl)==1:
                    ap(v)
                else:
                    ap(spl[position])
            dic = dict(zip(l_id,l_symbol2))
            before = list(map(lambda x: str(x),list(df.index)))
            new = []
            ap = new.append
            for v in before:
                try:
                    ap(dic[v])
                except KeyError:
                    print("{}: Error".format(v))
                    ap("")
            df.index = new
            df = df.sort_index()
        else:
            df = before.copy().T
            l_id = list(ref.index)
            l_id = list(map(lambda x: str(x),l_id))
            l_symbol = list(ref.iloc[:,0])        
            l_symbol = list(map(lambda x: str(x),l_symbol))
            l_symbol2 = []
            ap = l_symbol2.append
            for v in l_symbol:
                spl = v.split(separator)
                if len(spl)==1:
                    ap(v)
                else:
                    ap(spl[position])
            dic = dict(zip(l_id,l_symbol2))
            before = list(map(lambda x: str(x),list(df.index)))
            new = []
            ap = new.append
            for v in before:
                try:
                    ap(dic[v])
                except KeyError:
                    print("{}: Error".format(v))
                    ap("")
            df.index = new
            df = df.T
        del_nan = [v for v in list(df.index) if v is not np.nan]
        df = df.loc[del_nan,:]
        return df
    
    def __unique_index(self, df):
        dup = df.index[df.index.duplicated(keep="first")]
        df2 = pd.DataFrame()
        multi_list = pd.Series(dup).unique().tolist()
        if len(multi_list) != 0:
            for multi in multi_list:
                new = df.loc[multi].iloc[0,:]
                df2[multi] = new
            df = df.drop(multi_list)
            df = pd.concat([df,df2.T])
        return df