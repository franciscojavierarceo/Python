# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 10:32:36 2015

@author: franciscojavierarceo
"""

import os 
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model

v = DictVectorizer(sparse=True)

dftmp = pd.DataFrame(['Male', 'Male', 'Male', 'Male', 'NaN', 'Male', 'Male', 'Female',
       'NaN', 'Male'])
dftmp.fillna('Missing', inplace=True)
       
label = ['6. Call Later', '8. Interested', '6. Call later', '6. Call Later',
       '6. Call later', '6. Call later', '6. Call later', '6. Call later',
       '6. Call Later', '8. Interested']
y = []
for i in label:
    if i == '8. Interested':
        y.append(1)
    else:
        y.append(0)

Gender = pd.DataFrame(dftmp)
GenderDict = Gender.T.to_dict().values()
GenderXs = v.fit_transform(GenderDict)
print GenderXs,y

logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(GenderXs, y)

probs = pd.DataFrame(logreg.predict_proba(GenderXs))[0]
preds = pd.DataFrame(logreg.predict(GenderXs))

print probs,preds,y

print 'Residuals:',(y-probs)
# To run the file, call this:
# execfile('/Users/franciscojavierarceo/MyPrograms/Python/SparseMatrixBuildExample.py')
