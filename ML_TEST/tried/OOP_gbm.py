#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 20:57:50 2019

@author: wzhan
"""

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
from sklearn.metrics import confusion_matrix 
#from sklearn.linear_model import LogisticRegression 


class Data:
    ## Describe data 
    ## Understand distribution of classes 
    ## Find the missing value feature with percentage 
    ## Implement missing value or drop it 
    ## Understand the distribution of category feature in different class 
    ## Transfer category feature to numerical feature 
    ## Understand the distribution of numerical feature in different class 
    ## Drop irrelavent feature 
    
    def __init__(self, path):
        # Read the dataset using Pandas 
         
        self.path = path
        self.df = pd.read_csv(self.path)
        self.df.describe()
        
#        # Seperate X and y from Dataset 
#        self.y = self.df.is_bad
#        lst = ['is_bad']
#        self.X = self.df.drop(lst, axis = 1)
        
        
    def describe_features(self):
        # Describe the features dataset has 
        lst = ['is_bad']
        data_content = self.df.drop(lst, axis = 1)
        col = data_content.columns 
        print(col)
        del data_content, col, lst
        gc.collect()
        
    def OverSample_n_Split(self, var_rat = 0.15, test_rat = 0.15):
        df_1=self.df[self.df['is_bad']==1]
        df_0=self.df[self.df['is_bad']==0]
        
        bal_ratio=int(np.round((len(df_0)/len(df_1))))
        tt_ratio = var_rat + test_rat
        lenth_test_1=len(df_1)
        lenth_test_0=len(df_0)
        
        #Do train test split
        val_1=df_1.iloc[:int(lenth_test_1 * var_rat)]
        val_0=df_0.iloc[:int(lenth_test_0 * var_rat)]
        test_1=df_1.iloc[int(lenth_test_1 * var_rat):int(lenth_test_1 * tt_ratio)]
        test_0=df_0.iloc[int(lenth_test_0 * var_rat):int(lenth_test_0 * tt_ratio)]
        train_1=df_1.iloc[int(lenth_test_1 * tt_ratio):]
        train_0=df_0.iloc[int(lenth_test_0 * tt_ratio):]
        
        #Repeat class 1 samples
        df_val=pd.concat(bal_ratio*[val_1]+[val_0],axis=0).sample(frac=1).reset_index(drop=True)
        df_test=pd.concat(bal_ratio*[test_1]+[test_0],axis=0).sample(frac=1).reset_index(drop=True)
        df_train = pd.concat(bal_ratio*[train_1]+[train_0],axis=0).sample(frac=1).reset_index(drop=True)
        
        self.df = pd.concat([df_train, df_val, df_test], axis = 0).reset_index(drop=True)
        self.y = self.df.is_bad
        lst = ['is_bad']
        self.X = self.df.drop(lst, axis = 1)
        
        del df_1, df_0, bal_ratio, tt_ratio, lenth_test_1, lenth_test_0, \
        val_1, val_0, test_1, test_0, train_1, train_0
        gc.collect()
        return df_train, df_val, df_test

    
    def describe_data(self):
        # understand distribution of classes 
        plt.figure(figsize = [10, 5])
        plt.subplot(121)
        plt.pie(x = self.df.is_bad.value_counts(), labels = self.df.is_bad.unique())
        plt.title('the percentage of labels')
        plt.subplot(122)
        plt.bar(x = [0.2, 1], height = self.df.is_bad.value_counts(), width = 0.6)
        plt.title('the number of labels')
        plt.show()
    
    
    def find_missing(self):
        # find the feature that missing value and the missing percentage 
        missing = self.df.isnull().sum()
        for i, v in missing.iteritems():
            print('Percent of missing ' + i + ' records is %.2f%%' %((\
            self.df[i].isnull().sum()/self.df.shape[0])*100))
        return missing 
    
    def fillmissing(self, features):
        # Fill the missing value with most frequent value 
        df = self.df
        for feature in features:
            self.df[feature].fillna(df[feature].value_counts().idxmax(), inplace = True)
        del df
        gc.collect()

    def dropFeatures(self, lst):
        self.df.drop(lst, axis = 1, inplace = True)
    

class Feature_Select:
    
    def __init__(self, df, y):
        self.df = df
                                  
    def viewCate(self, feature):
        sns.barplot(feature, 'is_bad', data = self.df)
        plt.show()
    
    def viewNumeric(self):
        # standardization of data
        data_dia = self.df.is_bad
        lst = ['is_bad']
        data_content = self.df.drop(lst, axis = 1)
        data_n_2 = (data_content - data_content.mean())/(data_content.std())
        n_numeric = len(data_content.columns)
        for i in range(int(n_numeric/5)):
            data_temp = pd.concat([data_dia, data_n_2.iloc[:, 5*i:5*i+5]], axis = 1).reset_index(drop=True)
            data_temp = pd.melt(data_temp,id_vars="is_bad",
                                var_name="features",
                                value_name='value')
            plt.figure(figsize =(10,10))
            sns.violinplot(x = 'features', y = 'value', hue = 'is_bad', data = data_temp, split = True, inner = 'quart')
            plt.xticks(rotation = 90)  
            
        if 5* i + 5 < n_numeric:
            # Draw last few features 
            data_temp = pd.concat([data_dia, data_n_2.iloc[:, 5*i + 5 : n_numeric]]).reset_index(drop=True)
            data_temp = pd.melt(data_temp,id_vars="is_bad",
                                    var_name="features",
                                    value_name='value')
            plt.figure(figsize =(10,10))
            sns.violinplot(x = 'features', y = 'value', hue = 'is_bad', data = data_temp, split = True, inner = 'quart')
            plt.xticks(rotation = 90)  
    

    def heatmap(self):
        f, ax = plt.subplots(figsize=(18,18))
        sns.heatmap(self.df.corr(), annot=True, linewidths=0.5, fmt='.1f', ax=ax)
            
#    
    def emp_length_replace(self):
        '''
        Return a new column of data according to data.emp_length
        '''
        len_new = []
        len_old = self.df.emp_length
        for i in range(len(len_old)):
            l = len_old.iloc[i]
            if l=='na':
                len_new.append(0)
            else:
                len_new.append(int(l))
        self.df['emp_length_new'] = len_new
        lst = ['emp_length']
        self.df.drop(lst, axis = 1, inplace = True)
        
        del len_new, len_old, lst
        gc.collect()
        
    def FHasher(self, feature, n_features):
        from sklearn.feature_extraction import FeatureHasher
        
        fh = FeatureHasher(n_features, input_type='string')
        hashed_feature = fh.fit_transform(self.df[feature]).toarray()
        hashed_feature_label = [feature + str(i) for i in range(n_features)]
        hashed_feature_df = pd.DataFrame(hashed_feature, columns = hashed_feature_label)
        lst = [feature]
        self.df.drop(lst, axis = 1, inplace = True)
        self.df = pd.concat([hashed_feature_df, self.df], axis =1).reset_index(drop=True)

        del hashed_feature, hashed_feature_label, hashed_feature_df, lst
        gc.collect()
        
        
    def OrdEncoder(self, feature, t_map):
        self.df[feature] = self.df[feature].map(t_map)
        
        
    def OHEncoder(self, feature):
        
        from sklearn.preprocessing import OneHotEncoder, LabelEncoder
        
        gen_le = LabelEncoder()
        gen_labels = gen_le.fit_transform(self.df[feature])
        self.df['gen_feature'] = gen_labels

        ohe = OneHotEncoder(categories = 'auto')
        gen_feature_arr = ohe.fit_transform(self.df[['gen_feature']]).toarray()
        gen_feature_labels = list(gen_le.classes_)
        gen_features = pd.DataFrame(gen_feature_arr, columns = gen_feature_labels)
        lst = [feature, 'gen_feature']
        self.df.drop(lst, axis = 1, inplace = True)
        df = self.df.copy()
        self.df = pd.concat([df, gen_features], axis = 1).reset_index(drop=True)

        del gen_le, gen_labels, gen_feature_arr, gen_feature_labels, gen_features, df, lst
        gc.collect()
        
        
    def PCA(self):
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
        
        print('the first %d features could represent 95 percents of the viarance' % order)
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
        self.df = pd.concat([self.df.is_bad, X], axis = 1).reset_index(drop=True)
        del X, std_data, transformed_data, threshold, for_test, order, lst
        gc.collect()
    
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
        
class model:

    def __init__(self):
        '''
        Input: 
        Output: 
        '''
        self.model = GradientBoostingClassifier(
                random_state = 1,
                learning_rate=0.05,
                max_depth=8,
                verbose=1)
        
    
    def fit(self, X_train, y_train):
        '''
        Input:
        X : pd.DataFrame
        y : np.ndarray
        Output 
        None
        '''
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        '''
        Input: 
        X : pd.DataFrame
        Output: 
        y : np.ndarray
        '''
        y_pred =self.model.predict(X_test)
        return y_pred
    
    def predict_proba(self, X_test):
        '''
        Input: 
        X : pd.DataFrame 
        Output:
        y : np.ndarray
        '''
        y_pred_proba = self.model.predict_proba(X_test)
        return y_pred_proba
      
    
    def evaluate(self, X_test, y_test):
        '''
        Input: 
        X : pd.DataFrame
        y : np.ndarray 
        Output: 
        dict: such as {'f1_score': 0.8, 'logloss': 0.7}
        '''
        from sklearn.metrics import log_loss, f1_score, accuracy_score      
        score = {}
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        f1_score = f1_score(y_test, y_pred)
        logloss = log_loss(y_test, y_pred_proba)
        accuracy= accuracy_score(y_test, y_pred)
        score['f1_score'] = f1_score 
        score['logloss'] = logloss
        score['accuracy'] = accuracy
        
        return score 
    
    def tune_parameters(self, X_train, y_train, X_val, y_val):
        '''
        Input: 
        X : pd.DataFrame
        y : np.ndarray 
        Output: 
        dict: such as {'tol': 0.02, 'fit_intercept': False, 'solver': 'sag'}
        '''
        from sklearn.model_selection import KFold
        from sklearn.model_selection import GridSearchCV
          
        para_grid = {
                'tol':[0.1, 0.01, 0.001, 0.001], 
                'fit_intercept': [True, False], 
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'penalty': ['l1', 'l2'], 
                'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000 ]}

        kf =  KFold(n_splits = 5, shuffle = True, random_state = 1)
        best_paras = []
        for (train, test), i in zip(kf.split(X, y), range(5)):
            CV_log_reg = GridSearchCV(estimator= self.logModel, 
                                      param_grid=para_grid, n_jobs=-1)
            CV_log_reg.fit(X.iloc[train], y.iloc[train])
            best_paras.append(CV_log_reg.best_params_)
        return best_paras
    

    def plot_confusion_matrix(self, y_true, y_pred, classes=[0,1], cmap=plt.cm.Blues, title='Confusion Matrix'):
        
        con_m = confusion_matrix(y_true, y_pred)
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
        

    
    