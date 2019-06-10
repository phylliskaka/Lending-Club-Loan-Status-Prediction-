#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 10:00:58 2019

@author: wzhan
"""
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import log_loss, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
import sys

class Data:
    '''
    This is a class for Data Preprocessing.
    
    Attributes:
        path(str): the string object to store the path to dataset.
        df(DataFrame): the dataframe object to store dataset. 
    '''

    def __init__(self, path):
        '''
        The constructor for Data class.
        
        Parameters:
            path(str): the string object to store the path to dataset.
        '''
         
        self.path = path
        self.df = pd.read_csv(self.path)
        self.df.describe()
        
    def describe_features(self):
        '''
        The method to describe the features name of a Data instance.
        '''
        
        lst = ['is_bad']
        data_content = self.df.drop(lst, axis = 1)
        col = data_content.columns 
        print(col)
        # Free memory 
        del data_content, col, lst
        gc.collect()
    
    def OverSample_n_Split(self, var_rat = 0.15, test_rat = 0.15):
        '''
        The method to randomly oversample the dataset and split it into train,
        validation, and test dataset.
        
        Parameters: 
        var_rat(float): The ratio you want your validation dataset to be. 
        test_rat(float): The ratio you want your test dataset to be.
        
        Return: 
        df_train(DataFrame): training dataset 
        df_val(DataFrame): validation dataset 
        df_test(DataFrame): test dataset 
        '''
        
        # Seperate the dataset according to labels
        df_1 = self.df[self.df['is_bad'] == 1]
        df_0 = self.df[self.df['is_bad'] == 0]
        
        bal_ratio = int(np.round((len(df_0)/len(df_1))))
        tt_ratio = var_rat + test_rat
        lenth_test_1 = len(df_1)
        lenth_test_0 = len(df_0)
        
        # Split the dataset into train, val, test 
        val_1 = df_1.iloc[:int(lenth_test_1 * var_rat)]
        val_0 = df_0.iloc[:int(lenth_test_0 * var_rat)]
        test_1 = df_1.iloc[int(lenth_test_1 * var_rat):\
                         int(lenth_test_1 * tt_ratio)]
        test_0 = df_0.iloc[int(lenth_test_0 * var_rat):\
                         int(lenth_test_0 * tt_ratio)]
        train_1 = df_1.iloc[int(lenth_test_1 * tt_ratio):]
        train_0 = df_0.iloc[int(lenth_test_0 * tt_ratio):]
        
        # Oversample classes 1 
        df_val = pd.concat(bal_ratio*[val_1]+[val_0],axis=0).sample(frac=1)\
                                                    .reset_index(drop=True)
        df_test = pd.concat(bal_ratio*[test_1]+[test_0],axis=0).sample(frac=1)\
                                                    .reset_index(drop=True)
        df_train = pd.concat(bal_ratio*[train_1]+[train_0],axis=0)\
                                        .sample(frac=1).reset_index(drop=True)
        
        self.df = pd.concat([df_train, df_val, df_test], axis = 0)\
                                                    .reset_index(drop=True)
        self.y = self.df.is_bad
        lst = ['is_bad']
        self.X = self.df.drop(lst, axis = 1)
        # Free memory 
        del df_1, df_0, bal_ratio, tt_ratio, lenth_test_1, lenth_test_0, \
        val_1, val_0, test_1, test_0, train_1, train_0
        gc.collect()
        
        return df_train, df_val, df_test

    
    def describe_data(self):
        '''
        The method to understand the distribution of dataset in different 
        classes. It draws dataset in pie chart and bar chart. 
        '''
        
        # Understand distribution of classes 
        plt.figure(figsize = [10, 5])
        plt.subplot(121)
        plt.pie(x = self.df.is_bad.value_counts(), labels = self.df.\
                is_bad.unique())
        plt.title('the percentage of labels')
        plt.subplot(122)
        plt.bar(x = [0.2, 1], height = self.df.is_bad.value_counts(), \
                width = 0.6)
        plt.title('the number of labels')
        plt.show()
    
    
    def find_missing(self):
        '''
        The method to find the missing value percentage in all features.
        
        Return:
            missing(Series): it stores the number of missing value in all 
            features 
        '''
        
        # find the feature that missing value and the missing percentage 
        missing = self.df.isnull().sum()
        for i, v in missing.iteritems():
            # find the feature that missing value and the missing percentage 
            print('Percent of missing ' + i + ' records is %.2f%%' %((\
            self.df[i].isnull().sum()/self.df.shape[0])*100))
        return missing 
    
    def fillmissing(self, features):
        '''
        The method to fill the missing value in dataset. 
        
        Parameters:
            features(list): the list that stores n tuple of (feature_name, 
            number of missing value). 
        '''
        
        df = self.df
        for feature, v in features:
            if v < 0.3 * len(self.df): 
                self.df[feature].fillna(df[feature].value_counts().idxmax(),\
                       inplace = True)
            else:
                self.df[feature].fillna(df[feature].median(), inplace = True)
        # Free up memory 
        del df
        gc.collect()

    def dropFeatures(self, lst):
        '''
        The method to drop features in the lst 
        
        Paramters: 
            lst(list): the list that contains the feature need to be dropped.
        '''
        
        self.df.drop(lst, axis = 1, inplace = True)
        
    def drop_less_miss(self):
        '''
        The method to remove samples of columns with 0%<_<1% missing data
        '''
        n_miss_cols=list(len(self.df) - self.df.count())
        col_list = list(self.df.columns)
        tot_samples = len(self.df.index)
        cols_less_miss = [col_list[i] for i in range(len(col_list)) if \
                    ((n_miss_cols[i]!=0) and \
                     (n_miss_cols[i]<0.01*tot_samples))]
        print('Columns with 0%<_<1% missing samples are \n',cols_less_miss)
        self.df.dropna(subset=cols_less_miss,inplace=True)
        
    def convert_binary_features(self):
        '''
        The method to convert all binary features via. Label Encoder
        '''
        cntr = 0
        for col in self.df:
            if self.df[col].dtype == 'object':
                if len(list(self.df[col].unique())) <= 2:     
                    le = LabelEncoder()
                    self.df[col] = le.fit_transform(self.df[col])
                    cntr += 1
                    print(col)
                    
        print('%d columns were label encoded.' % cntr)
        
    def binary_train_test_split(self,test_frac=0.2):
        '''
        The method to returns python data_frames after equally splitting both 
        classes
        '''
        df_0 = self.df[self.df.is_bad == 0].sample(frac = 1)
        df_1 = self.df[self.df.is_bad == 1].sample(frac = 1)
    
        test_0,train_0 = df_0[:int(test_frac*len(df_0))],\
                            df_0[int(test_frac*len(df_0)):]
        test_1,train_1 = df_1[:int(test_frac*len(df_1))],\
                            df_1[int(test_frac*len(df_1)):]
        test_df = pd.concat([test_0,test_1]).reset_index(drop=True)
        train_df = pd.concat([train_0,train_1]).reset_index(drop=True)
        
        return train_df,test_df
        
    


class Feature_Select:
    '''
    This is the class for feature selection 
    
    Attributes:
        df(DataFrame): the dataset that need to do feature selection.
    '''
    
    def __init__(self, df, y):
        '''
        The constructor for Feature_Select class 
        
        Parameters:
            df(DataFrame): the dataset that need to do feature selection.
            y(Panda Series): the label of the dataset 
        '''
        
        self.df = df
                                  
    def viewCate(self, feature):
        '''
        The method to view the distribution of category feature in bar plot 
        
        Parameter:
            feature(str): feature name that need to be viewed  
        '''
        plt.figure()
        sns.barplot(feature, 'is_bad', data = self.df)
        plt.show()
    
    def viewNumeric(self):
        '''
        The method to view the distribution of numeric features in volin plot 
        
        '''
    
        data_dia = self.df.is_bad
        lst = ['is_bad']
        # get the dataset without label 
        data_content = self.df.drop(lst, axis = 1)
        # Normalization of data
        data_n_2 = (data_content - data_content.mean())/(data_content.std())
        n_numeric = len(data_content.columns)
        # View the feature distribution using seaborn volin plot 
        for i in range(int(n_numeric/5)):
            data_temp = pd.concat([data_dia, data_n_2.iloc[:, 5*i:5*i+5]],\
                                  axis = 1).reset_index(drop=True)
            data_temp = pd.melt(data_temp,id_vars="is_bad",
                                var_name="features",
                                value_name='value')
            plt.figure(figsize =(10,10))
            sns.violinplot(x = 'features', y = 'value', hue = 'is_bad',\
                           data = data_temp, split = True, inner = 'quart')
            plt.xticks(rotation = 90)  
    

    def heatmap(self):
        '''
        The method to view te heatmap of dataset.
        
        '''

        f, ax = plt.subplots(figsize=(18,18))
        sns.heatmap(self.df.corr(), annot=True, linewidths=0.5, fmt='.1f',\
                    ax=ax)
            
    def remove_constant_cols(self):
        '''
        The method to remove any columns filled with the same value all over.
        '''
        const_cols = list((self.df.loc[:, (self.df == self.df.iloc[0]).all()])\
                        .columns)
        print('Removing following constant columns \n',const_cols)
        self.df.drop(const_cols, axis = 1, inplace = True)
        #self.df=self.df.loc[:, (self.df != self.df.iloc[0]).any()]
        
    def object2float(self,cols):
        '''
        The method to replace the numeric features of dtype object or char 
        with corresponding float objects. 
        '''
        for col in cols:
            len_new = []
            len_old = self.df[col]
            for i in range(len(len_old)):
                l = len_old.iloc[i]
                try:
                    len_new.append(float(l))
                except ValueError:
                    len_new.append(0)
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    raise

            self.df[col+'_new'] = len_new
            # Free up memory 
            del len_new, len_old
            gc.collect()
            
        self.df.drop(cols, axis = 1, inplace = True)
         
    def FHasher(self, feature, n_features):
        '''
        The method to transfer nominal category feature, that has many classes,
        to numeric feature using feature hashing function.
        
        Parameters: 
            feature(str): the name of feature that need to transfer. 
            n_features: the number of dimension in transfered feature 
            
        '''
        
        from sklearn.feature_extraction import FeatureHasher
        
        fh = FeatureHasher(n_features, input_type='string')
        hashed_feature = fh.fit_transform(self.df[feature]).toarray()
        hashed_feature_label = [feature + str(i) for i in range(n_features)]
        hashed_feature_df = pd.DataFrame(hashed_feature,\
                                         columns = hashed_feature_label)
        lst = [feature]
        self.df.drop(lst, axis = 1, inplace = True)
        self.df = pd.concat([hashed_feature_df, self.df], axis =1)\
                                            .reset_index(drop=True)
        # Free up memory 
        del hashed_feature, hashed_feature_label, hashed_feature_df, lst
        gc.collect()
        
        
    def OrdEncoder(self, feature, t_map):
        '''
        The method to transfer ordinal categorical feature to numeric feature. 
        
        Parameters:
        feature(str): the name of the feature that need to transfer 
        t_map(dict): a dictionary tha store the map between category feature 
        and numeric feature. such t_map = {'VERIFIED - income': 2, 
        'VERIFIED - income source': 1, 'not verified': 0}
        ''' 
        
        self.df[feature] = self.df[feature].map(t_map)
        
        
    def OHEncoder(self, feature):
        '''
        The method to transfer nominal categorical feature to one-hot encoded
        numeric feature.
        
        '''
        
        self.df=pd.get_dummies(self.df.copy(),columns=[feature])
        
    def PCA(self):
        '''
        The method to perform Principle Component Analysis on dataset. It will 
        draw the principle components in pie chart. 
        '''
        
        # Using PCA to reduce the dimensionality of data 
        from sklearn.preprocessing import StandardScaler 
        conv = StandardScaler()
        lst = ['is_bad']
        X = self.df.drop(lst, axis = 1)
        std_data = conv.fit_transform(X)
        
        # use PCA to reduce dimensionality 
        from sklearn.decomposition import PCA 
        pca = PCA(n_components=len(self.df.columns)-1, svd_solver='full')
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
        
        print('the first %d features could represent 95% of the variance'\
              % order)
        print(pca.explained_variance_ratio_[:order].sum())
        com_col = ['com'+ str(i+1) for i in range(order)]
        com_col.append('others')
        com_value = [i for i in pca.explained_variance_ratio_[:order]]
        com_value.append(1-pca.explained_variance_ratio_[:order].sum())
        plt.figure(figsize=[4,4])
        plt.pie(x = com_value, labels = com_col)
        plt.title('Principal components')
        plt.show()
        
        X = pd.DataFrame(data = transformed_data)
        self.df = pd.concat([self.df.is_bad, X], axis = 1)\
                                    .reset_index(drop=True)
        # Free up memory 
        del X, std_data, transformed_data, threshold, for_test, order, lst
        gc.collect()
    
          
        
class model:
    '''
    This is a clas for Logistic Regression Model
    
    Attribute:
        logModel: A logistic model instance.
        class_weight: the parameter of the self.logistic model 
        dual: the parameter of the self.logistic model 
        intercept_scaling=intercept_scaling: the parameter of the self.logistic model 
        max_iter: the parameter of the self.logistic model 
        n_jobs: the parameter of the self.logistic model 
        penalty: the parameter of the self.logistic model 
        random_state: the parameter of the self.logistic model 
        verbose: the parameter of the self.logistic model 
        warm_start: the parameter of the self.logistic model 
        
    '''

    def __init__(self,C=1, class_weight={1:5},dual=False,\
                  fit_intercept=True,intercept_scaling=1,\
                  max_iter=2000,n_jobs=1, penalty='l2',\
                  random_state=2, solver='liblinear',\
                  tol=1e-10, verbose=0, warm_start=False):
        '''
        The constructor for model class.
 
        '''
        self.class_weight=class_weight
        self.dual=dual
        self.intercept_scaling=intercept_scaling
        self.max_iter=max_iter
        self.n_jobs=n_jobs
        self.penalty=penalty
        self.random_state=random_state
        self.verbose=verbose
        self.warm_start=warm_start
        
        self.logModel= LogisticRegression(C=C, class_weight=class_weight,\
                                          fit_intercept=fit_intercept,\
                                          intercept_scaling=intercept_scaling,\
                                          max_iter=max_iter,n_jobs=n_jobs,\
                                          penalty=penalty,verbose=verbose,\
                                          random_state=random_state,\
                                          solver=solver,tol=tol,dual=dual,\
                                          warm_start=warm_start)
    
    def fit(self, X_train, y_train, best_para = None):
        '''
        The method to fit the train dataset. 
        
        Parameters:
        X_train(DataFrame): the DataFrame to store the train dataset 
        y_train(Pandas Series):the DataFrame to store label 
        best_para(dict): A dictionary includes the best parameters for model. 
                         if None, it will use defaut parameters
        ''' 
        
        X_train_copy, y_train_copy = X_train.copy(), y_train.copy()
        if not(hasattr(self, 'col')):
            self.col = X_train_copy.columns
            
        if not(hasattr(self, 'scaler')):
            self.scaler = StandardScaler()
            X_train_copy[self.col] = self.scaler.fit_transform\
                                    (X_train_copy[self.col])
        else:
            X_train_copy[self.col] = self.scaler.transform\
                                    (X_train_copy[self.col])
        X_train_copy = X_train_copy[self.col]


        if type(best_para) == type(None):
            self.logModel.fit(X_train_copy, y_train_copy)
        else:
            #Re-initialize the logmodel
            self.__init__(C = best_para['C'], 
                            solver = best_para['solver'],\
                            fit_intercept = best_para['fit_intercept'],\
                            tol = best_para['tol'],\
                            class_weight=self.class_weight,\
                            dual=self.dual,\
                            intercept_scaling=self.intercept_scaling,\
                            max_iter=self.max_iter,n_jobs=self.n_jobs,\
                            penalty=self.penalty,\
                            random_state=self.random_state,\
                            verbose=self.verbose, warm_start=self.warm_start)
            self.logModel.fit(X_train_copy, y_train_copy)
        
        # Free up memory 
        del X_train_copy, y_train_copy
        gc.collect()
    
    def predict(self, X_test):
        '''
        The method to predict the class label for dataset of X_test.
        
        Parameters:
            X_test(DataFrame): the DataFrame to store test dataset.
        
        Return:
            y_pred(list): a list of integer that contains the prediction label 
            for X_test.
        '''
        
        X_test_copy = X_test.copy()
        X_test_copy.dropna(inplace=True)
        if len(X_test_copy)!=len(X_test):
            print('There was missing data in X. Use X.index to find correct y')
        X_test_copy[self.col] = self.scaler.transform(X_test_copy[self.col])
        X_test_copy = X_test_copy[self.col]
        y_pred =self.logModel.predict(X_test_copy)
        y_pred_arr = np.asarray(y_pred)
        
        # Free up memory 
        del X_test_copy, y_pred
        gc.collect()
        return y_pred_arr

    
    def predict_proba(self, X_test):
        '''
        The method to predict the probability for dataset of X_test.
        
        Parameters:
            X_test(DataFrame): the DataFrame to store test dataset.
        
        Return:
            y_pred_proba(list): a list of float that contains the prediction 
            probability. 
        '''
        
        X_test_copy = X_test.copy()
        X_test_copy.dropna(inplace=True)
        if len(X_test_copy)!=len(X_test):
            print('There was missing data in X. Use X.index to find correct y')
        X_test_copy[self.col] = self.scaler.transform(X_test_copy[self.col])
        X_test_copy = X_test_copy[self.col]

        y_pred_proba = self.logModel.predict_proba(X_test_copy)
        y_pred_proba_arr = np.asarray(y_pred_proba)
        
        # Free up memory
        del X_test_copy, y_pred_proba
        gc.collect()
        return y_pred_proba_arr
      
    
    def evaluate(self, X_test, y_test):
        '''
        The method to evaluate the model performance on X_test dataset
        
        Parameters: 
            X_test(DataFrame): the DataFrame to store test dataset.
            y_test(Pandas Series): the Pandas Series to store test label 
        
        Return:
            score(dict): a dictionary to store f1_score, logloss, accuracy. 
        '''
        
        X_test_copy,y_test_copy = X_test.copy(), y_test.copy()
        X_test_copy.dropna(inplace=True)
        y_test_copy = y_test_copy[X_test_copy.index]
        X_test_copy[self.col] = self.scaler.transform(X_test_copy[self.col])
        X_test_copy = X_test_copy[self.col]

        score = {}
        y_pred = self.predict(X_test_copy)
        y_pred_proba = self.predict_proba(X_test_copy)
        
        f1 = f1_score(y_test_copy, y_pred)
        logloss = log_loss(y_test_copy, y_pred_proba)
        accuracy= accuracy_score(y_test_copy, y_pred)
        score['f1_score'] = f1 
        score['logloss'] = logloss
        score['accuracy'] = accuracy
        print("Classification Report: \n {}\n".format\
              (classification_report(y_test_copy, y_pred)))
        print("Confusion Matrix: \n {}\n".format\
              (confusion_matrix(y_test_copy, y_pred)))
        
        # Free up memory 
        del X_test_copy,y_test_copy, y_pred, y_pred_proba
        gc.collect()
        return score 
    
    def tune_parameters(self, X_train, y_train, n_folds = 5, seed=47):
        '''
        The method to find the best parameters of logistic regression model 
        using KFold cross validation. 
        
        Parameters: 
            X_train(DataFrame): the DataFrame to store the train dataset 
            y_train(Pandas Series):the DataFrame to store label 
            n_folds(int): the number of k to split your dataset in KFold split 
        
        Return: 
        best_paras(dict): A dictionary includes the best parameters for model
                              during all the KFold cross validation search. 
        '''
        
        X_train_copy, y_train_copy = X_train.copy(), y_train.copy()
         
        para_grid = {
                'tol':[0.1, 0.001], 
                'fit_intercept': [True, False], 
                'solver': ['lbfgs', 'liblinear', 'sag', 'saga'],
                'C': [0.001, 0.1,100]}
        

        if not(hasattr(self, 'col')):
            self.col = X_train_copy.columns
            
        if not(hasattr(self, 'scaler')):
            self.scaler = StandardScaler()
            X_train_copy[self.col] = self.scaler.fit_transform\
                                    (X_train_copy[self.col])
        else:
            X_train_copy[self.col] = self.scaler.transform\
                                    (X_train_copy[self.col])
        X_train_copy = X_train_copy[self.col]

        CV_log_reg = GridSearchCV(cv = n_folds, estimator= self.logModel, 
                          param_grid=para_grid, n_jobs=-1)
        CV_log_reg.fit(X_train_copy, y_train_copy)
        best_para = CV_log_reg.best_params_
        # Free up memory 
        del X_train_copy, y_train_copy, para_grid
        gc.collect()
        return best_para
