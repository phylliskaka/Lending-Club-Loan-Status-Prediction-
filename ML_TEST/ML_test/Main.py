#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:22:37 2019

@author: wzhan
"""
#%% Import package 
from lib import OOP
import matplotlib.pyplot as plt 
import os 

#%%
plt.close('all')

# Preprocessing data 
## Read the data from csv file  
ROOT_DIR = os.getcwd()
path = os.path.join(ROOT_DIR, './data/DR_Demo_Lending_Club_reduced.csv')

## Understand the distribution of classes 
Data = OOP.Data(path)
Data.describe_data()

## Drop Id is unique and zip_code is unordered as well as very detailed numeric
## feature that will not help prediction. We are dropping states, a less detailed
## descriptor of geography as intuitively, it doesn't seem to be very highly
## related to loan defaulting behaviour right now. We'll add it later if needed
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
    elif v > 0.01 * len(Data.df):
        implst.append((i, v))

Data.dropFeatures(droplst)
Data.df.info()

# Remove samples of columns with 0%<_<1% missing data
Data.drop_less_miss()
Data.df.info()

#Fill in all missing data and check
Data.fillmissing(implst)
miss = Data.find_missing()

#Convert all binary features via. labelencoder
Data.convert_binary_features()

#%%
# Data Cleaning and Feature selection 
## Category Feature  
feature_select = OOP.Feature_Select(Data.df.copy(), Data.df.is_bad.copy())
feature_select.df.info()

### View feature 'home_ownership'
feature_select.viewCate('home_ownership')
feature_select.OHEncoder('home_ownership')
feature_select.df.info()

### View feature 'policy_code'
feature_select.viewCate('policy_code')
feature_select.OHEncoder('policy_code')

### View feature 'verification_status'
feature_select.viewCate('verification_status')
t_map = {'VERIFIED - income': 2, 'VERIFIED - income source': 1, 'not verified': 0}
feature_select.OrdEncoder('verification_status', t_map)

### View feature 'purpose_cat'
feature_select.viewCate('purpose_cat')
feature_select.OHEncoder('purpose_cat')

### View feature 'addr_state'
feature_select.viewCate('addr_state')
### The mean value of target label is almost same for different state, let's drop 
lst = ['addr_state']
feature_select.df.drop(lst, axis = 1, inplace = True)

## Numeric Feature 
### convert the feature of emp_length from object to numeric
feature_select.object2float(['emp_length'])

##Drop any constant columns
feature_select.remove_constant_cols()

### View all the numeric features using violin plot 
feature_select.viewNumeric()

feature_select.heatmap()

plt.close('all')
#%%
# Principal Component Analysis 
# Run this session only when you need to reduce dimensionaly of data 
feature_select.PCA()
#%%
## Update the Data.df with the dataset after feature selection and data clean
Data.df = feature_select.df.copy()
Data.y = feature_select.df.is_bad.copy()
lst = ['is_bad']
Data.X = Data.df.drop(lst, axis = 1)

#%% 
# Train model 
## Test on model.fit, model.predict, model.predict_proba, model.evaluate, model.plot_confusion_matrix

### Split dataset into train, validation, test 
df_train,df_test=Data.binary_train_test_split(0.2)


lst = ['is_bad']
y_train = df_train.is_bad.copy()
X_train = df_train.drop(lst, axis = 1).copy()

y_test = df_test.is_bad.copy()
X_test = df_test.drop(lst, axis = 1).copy()

model = OOP.model(random_state=2)
### fit the train dataset 
model.fit(X_train, y_train)
### Predict the test data set 
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test) 
### Evaluation the model
score = model.evaluate(X_train, y_train)
score = model.evaluate(X_test, y_test)

del model
#%% Do k-fold CV and grid_search

# Find the best parameter using KFold cross validation 
model = OOP.model()

# Find the best parameter 
best_para = model.tune_parameters(X_train, y_train, n_folds = 5)

print(best_para)

model.fit(X_train, y_train, best_para)
score1 = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test) 