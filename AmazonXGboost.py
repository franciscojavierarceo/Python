import os, sys
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print '***********************************************************************'
print '******************************  BEGIN  ********************************'
print '***********************************************************************'
sys.path.append('/Users/franciscojavierarceo/GitHub/xgboost/wrapper')
import xgboost as xgb
execfile("/Users/franciscojavierarceo/GitHub/My_Functions.py")
os.chdir("/Users/franciscojavierarceo/K/Amazon/Input")	
df1 = pd.read_csv('train.csv',sep=',')
# df2 = pd.read_csv('test.csv',sep=',')
labels = df1['ACTION'].values
xs = df1[df1.columns[1:]].values
trnflt = np.random.uniform(size=len(labels)) <=0.7
tstflt = trnflt==False
xstrn = xs[trnflt]
xstst = xs[tstflt]
xstrn = xgb.DMatrix(xstrn,label=labels[trnflt])
xstst = xgb.DMatrix(xstst,label=labels[tstflt])
depthval = 10
param = {'max_depth':depthval,'eta':0.4,'silent':1,
         'objective':'binary:logistic'}
num_round = 200
xgbout = xgb.cv(param, xstrn, num_round, nfold=5,
                metrics={'auc'}, seed = 0)
Iteration= []
TrainER = []
TrainSE = []
TestER = []
TestSE = []
for i in xgbout:
    Iteration.append(int(i.split()[0].split('[')[1].split(']')[0]))
    TestER.append(1-float(i.split()[1].split(':')[1].split('+')[0]))
    TestSE.append(float(i.split()[1].split(':')[1].split('+')[1]))
    TrainER.append(1-float(i.split()[2].split(':')[1].split('+')[0]))
    TrainSE.append(float(i.split()[2].split(':')[1].split('+')[1]))

Trn1 = np.array(TrainER)-np.array(TrainSE)
Trn2 = np.array(TrainER)+np.array(TrainSE)
Tst1 = np.array(TestER)-np.array(TestSE)
Tst2 = np.array(TestER)+np.array(TestSE)
plt.plot(Iteration, TrainER,'rs-',label='Train Error')
plt.ylim(0,1)
plt.grid()
plt.plot(Iteration,Trn1,'r--')
plt.plot(Iteration,Trn2,'r--')
plt.plot(Iteration,TestER, 'bs-',label='Test Error')
plt.plot(Iteration,Tst1,'b--')
plt.plot(Iteration,Tst2,'b--')
plt.legend(loc='upper right')
plt.show()
change = []
for i, idx in enumerate(TrainER):
	change.append(TrainER[i] - TrainER[i-1])
	if i > 0 and np.round(change[i],3)==0:
		print "CV Reached Lower Bound"
		break

param = {'max_depth':depthval, 'eta':0.4, 'silent':1,
         'objective':'binary:logistic' }
watchlist  = [(xstst,'eval'), (xstrn,'train')]
num_round = len(change)
bst = xgb.train(param, xstrn, num_round, watchlist)
probtrn= bst.predict(xstrn, ntree_limit=depthval)
probtst = bst.predict(xstst, ntree_limit=depthval)
ytst = labels[tstflt]
ytrn = labels[trnflt]
roc_perf(ytrn,probtrn,ytst,probtst)
print '***********************************************************************'
print '******************************  END  **********************************'
print '***********************************************************************'
