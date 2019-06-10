#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 20:37:04 2019

@author: wzhan
"""

import unittest 
import pandas as pd
import os
from lib import OOP
from sklearn.model_selection import train_test_split

ROOT_DIR = os.getcwd()

path_reproducible = os.path.join(ROOT_DIR, 
                './data/df_reproduceCheck.csv')
path_missing = os.path.join(ROOT_DIR, 
                './data/df_missingCheck.csv')
path_newFeature = os.path.join(ROOT_DIR, 
                './data/df_newFeatureCheck.csv')

df1 = pd.read_csv(path_reproducible)
df2 = pd.read_csv(path_missing)
df3 = pd.read_csv(path_newFeature)

X1_train, X1_test, y1_train, y1_test = train_test_split(df1.drop('is_bad',axis=1),
                                                    df1['is_bad'],test_size=0.3,
                                                    random_state=101)

X2_train, X2_test, y2_train, y2_test = train_test_split(df2.drop('is_bad',axis=1),
                                                    df2['is_bad'],test_size=0.3,
                                                    random_state=101)

X3_train, X3_test, y3_train, y3_test = train_test_split(df3.drop('is_bad',axis=1),
                                                    df3['is_bad'],test_size=0.3,
                                                    random_state=101)

class PredictTestCase(unittest.TestCase):
    '''
    The function to do unit test on reproducible, handle missing value, handle 
    new category level at prediction time
    '''
    
    def test_reproducible(self):
        '''
        The function to test the model is reproducible or not.
        '''
        model1 = OOP.model()
        model1.fit(X1_train, y1_train)
        y_pred1 = model1.predict(X1_test)
        
        model2 = OOP.model()
        model2.fit(X1_train, y1_train)
        y_pred2 = model2.predict(X1_test)
        
        self.assertTrue((y_pred1 == y_pred2).all())
    
    def test_missingValue(self):
        '''
        The function to test the model can handle missing value or not.
        '''
        model = OOP.model()
        model.fit(X1_train, y1_train)
        model.predict(X2_test)
        
    def test_newFeature(self,):
        '''
        The function to test the model can handle new category level feature 
        or not. 
        '''
        model = OOP.model()
        model.fit(X1_train, y1_train)
        model.predict(X3_test)        
        
        
if __name__ == '__main__':
    unittest.main()