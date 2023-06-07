# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:05:34 2019

@author: tadahaya
"""
import unittest
import pandas as pd
import os
import sys
import math

from enan.binom import BT

class SampleTest(unittest.TestCase):
    CLS_VAL = 'none'

    # called when test class initialization
    @classmethod
    def setUpClass(cls):
        if sys.flags.debug:
            print('> setUpClass method is called.')
        cls.CLS_VAL = '> setUpClass : initialized!'
        if sys.flags.debug:
            print(cls.CLS_VAL)

    # called when test class end
    @classmethod
    def tearDownClass(cls):
        if sys.flags.debug:
            print('> tearDownClass method is called.')
        cls.CLS_VAL = '> tearDownClass : released!'
        if sys.flags.debug:
            print(cls.CLS_VAL)

    # called when a test method runs
    def setUp(self):
        if sys.flags.debug:
            print(os.linesep + '> setUp method is called.')
        self.smpl = BT()

    # called when a test method ends
    def tearDown(self):
        if sys.flags.debug:
            print(os.linesep + '> tearDown method is called.')

    def _df_checker(self,df):
        if type(df)!=pd.core.frame.DataFrame:
            return False
        elif df.shape[0]==0:
            return False
        else:
            head = df.head(1)
            judge = math.isnan(head.iat[0,0])
            return not judge

    def _sr_checker(self,sr):
        if type(sr)!=pd.core.series.Series:
            return False
        if sr.shape[0]==0:
            return False
        else:
            head = sr.head(1)
            judge = math.isnan(head.iat[0])
            return not judge

    def test_calc(self):
        # prepare test patterns
        test_patterns = [
            ("fdr_bh","greater",None), # (arg1, arg2, ..., expected result)
            ("fdr_bh","two-sided",None), # (arg1, arg2, ..., expected result)
            ("fdr_bh","less",None), # (arg1, arg2, ..., expected result)
            ("fdr_bh","greater",3) # (arg1, arg2, ..., expected result)
            ]

        ref,obj = self.smpl.generate_test_data()
        self.smpl.fit(ref)

        ### loop for sweeping all conditions
        for tcorr,tmode,tfocus in test_patterns:
            with self.subTest(correction=tcorr,mode=tmode,focus=tfocus):
                self.assertTrue(self._df_checker(self.smpl.calc(data=obj,
                                                 correction=tcorr,mode=tmode,focus=tfocus)))
