# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:27:16 2015
@author: franciscojavierarceo
"""
import scipy
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, decomposition
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
print '***********************************************************************'
print '******************************  BEGIN  ********************************'
print '***********************************************************************'
sys.path.append('/Users/franciscojavierarceo/GitHub/xgboost/wrapper')
import xgboost as xgb
execfile("/Users/franciscojavierarceo/GitHub/Python/My_Functions.py")
os.chdir("/Users/franciscojavierarceo/ProjectWork/MIS/GDW")	
df1 = pd.read_csv('1000StaffComplaints.csv',sep=',')
varchar = df1['CaseSummary']

def Build_STDM(docs, **kwargs):
    ''' Build Spares Term Document Matrix '''
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd
    vectorizer = CountVectorizer(**kwargs)
    sparsematrix= vectorizer.fit_transform(docs)
    vocab = vectorizer.vocabulary_.keys()
    return sparsematrix, vocab

# Call the function on e-mail messages. The token_pattern is set so that terms are only
# words with two or more letters (no numbers or punctuation)
xs, vocab = Build_STDM(varchar)
print xs[:,1:10]
print vocab[0:10]

xs = xs[:,1:10]
y = np.hstack((np.ones(100/2),np.zeros(100/2))).reshape((100,1))
betas = scipy.sparse.linalg.inv(xs.T.dot(xs)).dot(xs.T).dot(y)
bdf= pd.DataFrame()
bdf['Variables'] = vocab[1:10]
bdf['betas'] = betas

print bdf

logistic = linear_model.LogisticRegression(penalty='l2',tol=0.0001,
                                           fit_intercept=True,intercept_scaling=1)
MyModel = logistic.fit(xs,y)
betas = MyModel.coef_.ravel()
ys = logistic.predict_proba(xs)[:,0]
roc_plot(y,ys,'My ROC')