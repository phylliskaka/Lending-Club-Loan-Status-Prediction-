#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 10:09:29 2019

@author: wzhan
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

import os 
# Read the data using pandas 
ROOT_DIR = os.getcwd()
path = os.path.join(ROOT_DIR, 'DR_Demo_Lending_Club_reduced.csv')
data = pd.read_csv(path)
print(data.columns)


#%%
lst = ['Id']
data.drop(lst, axis = 1, inplace = True )
#%%
# Deal with missing values that missing percentage is larger than 50%
print('Percent of missing "mths_since_last_record" records is %.2f%%' %((data['mths_since_last_record'].isnull().sum()/data.shape[0])*100))
print('Percent of missing "mths_since_last_delinq" records is %.2f%%' %((data['mths_since_last_delinq'].isnull().sum()/data.shape[0])*100))

## Missing 91.6% and 63.16% in these two columns. It means using these two features is not wise. 
## We will ignore this variable in our model 

## Drop two features 
lst = ['mths_since_last_record', 'mths_since_last_delinq']
data.drop(lst, axis = 1, inplace = True)

#%%
# Deal with missing values that missing percentage is smaller than 50%
print('Percent of missing "collections_12_mths_ex_med" records is %.2f%%' %((data['collections_12_mths_ex_med'].isnull().sum()/data.shape[0])*100))
print('Percent of missing "revol_util" records is %.2f%%' %((data['revol_util'].isnull().sum()/data.shape[0])*100))

print('Percent of missing "annual_inc" records is %.2f%%' %((data['annual_inc'].isnull().sum()/data.shape[0])*100))
print('Percent of missing "delinq_2yrs" records is %.2f%%' %((data['delinq_2yrs'].isnull().sum()/data.shape[0])*100))

print('Percent of missing "inq_last_6mths" records is %.2f%%' %((data['inq_last_6mths'].isnull().sum()/data.shape[0])*100))
print('Percent of missing "open_acc" records is %.2f%%' %((data['open_acc'].isnull().sum()/data.shape[0])*100))

print('Percent of missing "pub_rec" records is %.2f%%' %((data['pub_rec'].isnull().sum()/data.shape[0])*100))
print('Percent of missing "total_acc" records is %.2f%%' %((data['total_acc'].isnull().sum()/data.shape[0])*100))

#%%
# Since the missing value of all feature are less 0.5%, so we can 
# just input with the most frequent value 
data_new = data.copy()
data_new['collections_12_mths_ex_med'].fillna(data['collections_12_mths_ex_med'].value_counts().idxmax(), inplace = True)
data_new['revol_util'].fillna(data['revol_util'].value_counts().idxmax(), inplace = True)
data_new['annual_inc'].fillna(data['annual_inc'].value_counts().idxmax(), inplace = True)
data_new['delinq_2yrs'].fillna(data['delinq_2yrs'].value_counts().idxmax(), inplace = True)
data_new['inq_last_6mths'].fillna(data['inq_last_6mths'].value_counts().idxmax(), inplace = True)
data_new['open_acc'].fillna(data['open_acc'].value_counts().idxmax(), inplace = True)
data_new['pub_rec'].fillna(data['pub_rec'].value_counts().idxmax(), inplace = True)
data_new['total_acc'].fillna(data['total_acc'].value_counts().idxmax(), inplace = True)
data_new['emp_length'].fillna(data['emp_length'].value_counts().idxmax(), inplace = True)

#%%
# Understand the category features 
sns.barplot('addr_state', 'is_bad', data = data_new)
plt.show()
## it looks like State dont have some effect on the data 

#%% 
sns.barplot('home_ownership', 'is_bad', data = data_new)
plt.show()
## it looks like home_ownership dont have strong relation to prediction label

#%%
sns.barplot('initial_list_status', 'is_bad', data = data_new)
plt.show()
## it looks like initial_list_status dont have strong relation to prediction label

#%%
sns.barplot('pymnt_plan', 'is_bad', data = data_new)
plt.show()
## it looks like pymnt_plan have strong relation to prediction label

#%%
sns.barplot('policy_code', 'is_bad', data = data_new)
plt.show()
data_new.drop('policy_code')
## it looks like policy_code dont have strong relation to prediction label

#%%
sns.barplot('verification_status', 'is_bad', data = data_new)
plt.show()
## it looks like verification_status dont have strong relation to prediction label

#%%
sns.barplot('purpose_cat', 'is_bad', data = data_new)
plt.show()
## it looks like purpose_cat have strong relation to prediction label

#%%
# Drop the category feature that not helping predication 
lst = ['verification_status', 'policy_code', 'policy_code', 'initial_list_status', 'home_ownership', 'addr_state', 'zip_code', 'emp_length']
data_new.drop(lst, axis = 1, inplace = True)
print(data_new.columns)
#%%
# Create new features from 'purpose cat' and 'pymnt_plan'

def cats_replace(data_cat):
    '''
    Return a new column of data according to data.purpose_cat
    '''
    cats = ['medical', 'debt consoildation', 'credit card', 'car', 'wedding', 'house',
           'educational', 'mahor purchase', 'home improvement', 'vacation', 'moving', 
           'small business', 'renewable energy', 'other']
    cats_new = []
    for i in range(len(data_cat)):
        cat = data_cat.iloc[i]
        if cat in cats:
            cats_new.append(0)
        else:
            cats_new.append(1)
    return cats_new

data_new['purpose_new_cat'] = cats_replace(data_new.purpose_cat)

data_1 = pd.get_dummies(data_new, columns =['pymnt_plan'])
lst = ['pymnt_plan_y', 'purpose_cat']
data_1.drop(lst, axis = 1, inplace = True)
#%%
# Drop the predictions label from data 
y = data.is_bad
x = data_1.drop('is_bad', axis = 1)

#%%
# understand distribution of label 
plt.figure(figsize = [10, 5])
plt.subplot(121)
plt.pie( x = y.value_counts(), labels = y.unique())
plt.title('the percentage of labels')
plt.subplot(122)
plt.bar(x = [0.2, 1], height = y.value_counts(), width = 0.6)
plt.title('the number of labels')
plt.show()

# It's unbalanced dataset. Hence, we can use F1 as metrics. 

#%%
# standardization of data
data_dia = y
data_content = x 
data_n_2 = (data_content - data_content.mean())/(data_content.std())
data_temp = pd.concat([data_dia, data_n_2.iloc[:, 8:13]], axis = 1)
data_temp = pd.melt(data_temp,id_vars="is_bad",
                    var_name="features",
                    value_name='value')
plt.figure(figsize =(10,10))
sns.violinplot(x = 'features', y = 'value', hue = 'is_bad', data = data_temp, split = True, inner = 'quart')
plt.xticks(rotation = 90)

#%%
# Using PCA to reduce the dimensionality of data 
from sklearn.preprocessing import StandardScaler 
conv = StandardScaler()
std_data = conv.fit_transform(x)

# use PCA to reduce dimensionality 
from sklearn.decomposition import PCA 
pca = PCA(n_components=13, svd_solver='full')
transformed_data = pca.fit_transform(std_data)
print(transformed_data.shape)
print(pca.explained_variance_ratio_*100)
print(pca.explained_variance_)

threshold = 0.95
for_test = 0
order = 0 
for index, ratio in enumerate(pca.explained_variance_ratio_):
    if threshold > for_test:
        for_test += ratio 
    else:
        order = index + 1 
        break 

print('the first %d features could represent 95 percents of the viarance' % order)
print(pca.explained_variance_ratio_[:order].sum())
com_col = ['com'+ str(i+1) for i in range(order)]
com_col.append('others')
com_value = [i for i in pca.explained_variance_ratio_[:order]]
com_value.append(1-pca.explained_variance_ratio_[:order].sum())
plt.figure(figsize=[4,4])
plt.pie(x = com_value, labels = com_col)
plt.title('the first 12 components')
plt.show()

#%%
# using regularization for logistic regression, use gridSearchCV to search
# for the best type of regularization and corrisponding paramenter
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
 
X_train, X_test, y_train, y_test = train_test_split(transformed_data, y, 
                                                    test_size = 0.2)
logistic_reg = LogisticRegression()
para_grid = {
        'penalty': ['l1', 'l2'], 
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000 ]}

CV_log_reg = GridSearchCV(estimator= logistic_reg, param_grid=para_grid, n_jobs=-1)
CV_log_reg.fit(X_train, y_train)
best_para = CV_log_reg.best_params_
print('The best parameters are: ', best_para)

#%% 
# the helper function for ploting confusion matrix and get accuracy 
from sklearn.metrics import confusion_matrix 
def plot_confusion_matrix(label, pred, classes=[0,1], cmap=plt.cm.Blues, title='Confusion Matrix'):
    con_m = confusion_matrix(label,pred)
    plt.imshow(con_m, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    thres = con_m.max()/2
    for j in range(con_m.shape[0]):
        for i in range(con_m.shape[1]):
            plt.text(i,j, con_m[j,i],
                     horizontalalignment = 'center',
                     color='white' if con_m[i,j]>thres else 'black')
    plt.ylabel('true label')
    plt.xlabel('predicted label')
    plt.xticks(classes, classes)
    plt.yticks(classes, classes)
    plt.tight_layout()
    
def print_accuracy(label, y_pred):
    tn, fp, fn, tp = confusion_matrix(label, y_pred).ravel()
    print('Accuracy rate = %.2f' %((tp+tn)/(tn+fp+fn+tp)))

    
#%%
# now using the best parmeters to log the regression model 
logistic_reg = LogisticRegression(C =best_para['C'], penalty=best_para['penalty'])
logistic_reg.fit(X_train, y_train)
y_pred = logistic_reg.predict(X_test)

# result 
plot_confusion_matrix(y_test, y_pred)
plt.show()
print_accuracy(y_test, y_pred)

