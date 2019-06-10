#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:22:37 2019

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

## Drop the feature if missing percentage more than 70%
## Complement the feature is missing value less than 70% 
for i, v in missing.iteritems():
    if v > 0.7 * len(Data.df):
        droplst.append(i)
    elif v > 0:
        implst.append((i, v))

Data.dropFeatures(droplst)
Data.fillmissing(implst)
miss = Data.find_missing()

#%%
# Data Cleaning and Feature selection 
## Category Feature  
feature_select = OOP.Feature_Select(Data.df, Data.df.is_bad)

### View feature 'home_ownership'
feature_select.viewCate('home_ownership')
feature_select.OHEncoder('home_ownership')

### View feature 'policy_code'
feature_select.viewCate('policy_code')
feature_select.OHEncoder('policy_code')

### View feature 'verification_status'
feature_select.viewCate('verification_status')
t_map = {'VERIFIED - income': 2, 'VERIFIED - income source': 1, 'not verified': 0}
feature_select.OrdEncoder('verification_status', t_map)

### View feature 'purpose_cat'
feature_select.viewCate('purpose_cat')
feature_select.FHasher('purpose_cat', 5)

### View feature 'initial_list_status'
feature_select.viewCate('initial_list_status')
# Transfer initial_list_status
feature_select.OHEncoder('initial_list_status')

### View feature 'pymnt_plan'
feature_select.viewCate('pymnt_plan')
# Transfer 'pymnt_plan' to numeric feature
feature_select.OHEncoder('pymnt_plan')

### View feature 'addr_state'
feature_select.viewCate('addr_state')
### The mean value of target label is almost same for different state, let's drop 
lst = ['addr_state']
feature_select.df.drop(lst, axis = 1, inplace = True)

#%%
# Data Cleaning and Feature selection 
## Numeric Feature 

### Replace the feature of emp_length
feature_select.emp_length_replace()

### View all the numeric features using violin plot 
feature_select.viewNumeric()

# Drop the feature that has high correlation(>0.8) with another feature
# The reason to drop collections_12_mths_ex_med is the value of this feature is 
# all zero 
feature_select.heatmap()

droplst = ['f', 'y', 'MORTGAGE', 'open_acc', 'purpose_cat4', 'purpose_cat3', 'collections_12_mths_ex_med']
feature_select.df.drop(droplst, axis = 1, inplace = True)

#%%
# Principal Component Analysis 
# Run this session only when you need to reduce dimensionaly of data 
feature_select.PCA()
#%%
## Update the Data.df with the dataset after feature selection and data clean
Data.df = feature_select.df
Data.y = feature_select.df.is_bad
lst = ['is_bad']
Data.X = Data.df.drop(lst, axis = 1)

#%% 
# Train model 
## Test on model.fit, model.predict, model.predict_proba, model.evaluate, model.plot_confusion_matrix

model = OOP.model()
df = Data.df
### Split dataset into train, validation, test 
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

### fit the train dataset 
model.fit(X_train, y_train)
### Predict the test data set 
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test) 
### Evaluation the model 
score = model.evaluate(X_test, y_test)
### Plot confusion matrix 
model.plot_confusion_matrix(y_test, y_pred, classes=[0,1], cmap=plt.cm.Blues, title='Confusion Matrix')

#%% 
# Find the best parameter using KFold cross validation 
model = OOP.model()

# Find the best parameter 
best_para = model.tune_parameters(Data.X, Data.y, stratified = True)
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
score1 = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test) 
score = model.evaluate(X_test, y_test)
model.plot_confusion_matrix(y_test, y_pred, classes=[0,1], cmap=plt.cm.Blues, title='Confusion Matrix')

