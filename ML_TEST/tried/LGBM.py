#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 12:03:30 2019

@author: wzhan
"""

#%% Import package 
import OOP
import matplotlib.pyplot as plt 
import os 


# Preprocessing data 
## Read the data from csv file  
ROOT_DIR = os.getcwd()
path = os.path.join(ROOT_DIR, 'DR_Demo_Lending_Club_reduced.csv')

## Understand the distribution of classes 
Data = OOP.Data(path)
Data.describe_data()

## Drop Id and zip_code feature that is not helping predition 
lst = ['Id', 'zip_code']
Data.dropFeatures(lst)

## Find the feature that missing data and fix it 
missing = Data.find_missing()
droplst = []
implst = []

### Drop the feature if missing percentage more than 50%
### Complement the feature is missing value less than 50% 
for i, v in missing.iteritems():
    if v > 0.7 * len(Data.df):
        droplst.append(i)
    elif v > 0:
        implst.append((i, v))

Data.dropFeatures(droplst)
Data.fillmissing(implst)
miss = Data.find_missing()


### Oversample the dataset 
#df_train, df_val, df_test = Data.OverSample_n_Split()
#%% Feature Selection 

import pandas as pd

lst = ['addr_state']
Data.df.drop(lst, axis = 1, inplace = True)

## Category feature 

feature_select = OOP.Feature_Select(Data.df, Data.df.is_bad)
feature_select.emp_length_replace()

df = feature_select.df 
feature_select.df = pd.get_dummies(df)


#%%
feature_select = OOP.Feature_Select(Data.df, Data.df.is_bad)


# View feature 'home_ownership'
feature_select.viewCate('home_ownership')
feature_select.OHEncoder('home_ownership')

# View feature 'policy_code'
feature_select.viewCate('policy_code')
feature_select.OHEncoder('policy_code')


# View feature 'verification_status'
feature_select.viewCate('verification_status')
t_map = {'VERIFIED - income': 2, 'VERIFIED - income source': 1, 'not verified': 0}
feature_select.OrdEncoder('verification_status', t_map)

# View feature 'purpose_cat'
feature_select.viewCate('purpose_cat')
feature_select.FHasher('purpose_cat', 5)

# View feature 'initial_list_status'
feature_select.viewCate('initial_list_status')
# Transfer initial_list_status
feature_select.OHEncoder('initial_list_status')


# View feature 'pymnt_plan'
feature_select.viewCate('pymnt_plan')
# Transfer 'pymnt_plan' to numeric feature
feature_select.OHEncoder('pymnt_plan')

# View feature 'addr_state'
feature_select.viewCate('addr_state')
# The mean value of target label is almost same for different state, let's drop 

lst = ['addr_state']
feature_select.df.drop(lst, axis = 1, inplace = True)


#%%
## Numeric Feature 

# Replace the feature of emp_length
feature_select.emp_length_replace()

# View all the numeric features using violin plot 
feature_select.viewNumeric()

# Drop the feature that has high correlation(>0.8) with another feature
# The reason to drop collections_12_mths_ex_med is the value of this feature is 
# all zero 
feature_select.heatmap()

droplst = ['f', 'y', 'MORTGAGE', 'open_acc', 'purpose_cat4', 'purpose_cat3', 'collections_12_mths_ex_med']
feature_select.df.drop(droplst, axis = 1, inplace = True)


#%%
# Run this session only when you ned to reduce dimensionaly of data 
feature_select.PCA()
#%%

Data.df = feature_select.df
Data.y = feature_select.df.is_bad
lst = ['is_bad']
Data.X = Data.df.drop(lst, axis = 1)

#%%
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
import numpy as np

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        print("Train Result:\n")
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_train, clf.predict(X_train))))
        print("Classification Report: \n {}\n".format(classification_report(y_train, clf.predict(X_train))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, clf.predict(X_train))))

        res = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
        print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
        print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))
        
    elif train==False:
        print("Test Result:\n")        
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, clf.predict(X_test))))
        print("Classification Report: \n {}\n".format(classification_report(y_test, clf.predict(X_test))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, clf.predict(X_test))))
        
#%% Create a model 
## Test on model.fit, model.predict, model.predict_proba, model.evaluate, model.plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

model = OOP.model()

df = Data.df
#df_train, df_val, df_test = Data.OverSample_n_Split()
X_train, X_test, y_train, y_test = train_test_split(df.drop('is_bad',axis=1),df['is_bad'],test_size=0.15,random_state=101)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test=sc.transform(X_test)
sm = SMOTE(random_state=12, ratio = 1.0)
x_train_r, y_train_r = sm.fit_sample(X_train, y_train)

#y_train = df_train.is_bad 
#lst = ['is_bad']
#X_train = df_train.drop(lst, axis = 1)
#
#y_test = df_test.is_bad 
#lst = ['is_bad']
#X_test = df_test.drop(lst, axis = 1)
#
#y_val = df_val.is_bad 
#lst = ['is_bad']
#X_val = df_val.drop(lst, axis = 1)

model.fit(x_train_r, y_train_r)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test) 
score = model.evaluate(X_test, y_test)
model.plot_confusion_matrix(y_test, y_pred, classes=[0,1], cmap=plt.cm.Blues, title='Confusion Matrix')
print_score(model.logModel, x_train_r, y_train_r, X_test, y_test, train=False)

#%% Create a model 
model = OOP.model()

## Test on model.tune_parameters
best_para = model.tune_parameters(Data.X, Data.y)
print(best_para)

df_train, df_val, df_test = Data.OverSample_n_Split()

y_train = df_train.is_bad 
lst = ['is_bad']
X_train = df_train.drop(lst, axis = 1)

y_test = df_test.is_bad 
lst = ['is_bad']
X_test = df_test.drop(lst, axis = 1)

y_val = df_val.is_bad 
lst = ['is_bad']
X_val = df_val.drop(lst, axis = 1)

model.fit(X_train, y_train, best_para)
score1 = model.evaluate(Data.X_test, Data.y_test)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test) 
score = model.evaluate(X_test, y_test)
model.plot_confusion_matrix(y_test, y_pred, classes=[0,1], cmap=plt.cm.Blues, title='Confusion Matrix')

