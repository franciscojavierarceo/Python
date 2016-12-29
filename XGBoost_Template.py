import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.linear_model import enet_path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors.kde import KernelDensity
from sklearn.metrics import roc_curve, auc
import numpy as np
import scipy as sparse
import pandas as pd

def roc_plot(actual,pred,ttl):
    fpr, tpr, thresholds = roc_curve(actual, pred)
    roc_auc = auc(fpr, tpr)
    print("The Area Under the ROC Curve : %f" % roc_auc)
    # Plot ROC curve
    plt.clf()
    plt.plot(fpr, tpr, color='red',label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve'+'\n'+ttl)
    plt.legend(loc="lower right")
    plt.show()

def roc_perf(atrn,ptrn,atst,ptst):
    fprtrn, tprtrn, thresholds = roc_curve(atrn, ptrn)
    fprtst, tprtst, thresholdstst = roc_curve(atst, ptst)
    roc_auctrn= auc(fprtrn, tprtrn)
    roc_auctst = auc(fprtst, tprtst)
    print("The Training Area Under the ROC Curve : %f" % roc_auctrn)
    print("The Test Area Under the ROC Curve : %f" % roc_auctst)
    # Plot ROC curve
    plt.clf()
    plt.plot(fprtrn, tprtrn, color='red',label='Train AUC = %0.2f' % roc_auctrn)
    plt.plot(fprtst, tprtst, color='blue',label='Test AUC = %0.2f' % roc_auctst)
    plt.plot([0, 1], [0, 1], 'k')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def myauc(actual,pred):
    fpr, tpr, thresholds = metrics.roc_curve(actual, pred)
    return metrics.auc(fpr, tpr)

def replacemissingFunc(df):
    print "The input data has",df.shape[0], "rows and",df.shape[1],"columns"
    for varname in df.columns:
        if df[varname].dtype == np.float:
            df[varname+'_missing']= 0
            df.loc[pd.isnull(df[varname]),varname+'_missing']  = 1
            df[varname] =  df[varname].fillna(np.mean(df[varname]))
        if df[varname].dtype==np.object:
            df[varname] = df[varname].fillna('MISSING')
    print "The output data has",df.shape[0], "rows and",df.shape[1],"columns"
    return df

def buildMatrix(df):
    numcols = []
    catcols = []
    for i, col in enumerate(df.columns):
        if df[col].dtype == "object" and c!='text':
            catcols.append(col)
            if i == 1:
                xs_cat = pd.get_dummies(df[col], prefix="",sparse=True)
            if i > 1:
                xs_cat = scipy.sparse.hstack([xs_cat, pd.get_dummies(df[col], prefix="",sparse=True)])
        if df[col].dtype in (np.float,np.int):
            numcols.append(col)
    
    xs = scipy.sparse.hstack([df[numcols],xs_cat])


df = replacemissingFunc(df)
ys = df[:,'y']
xdf_cols = [col for col in df.columns if col!='y']
xdf = df[cols]
xs = buildMatrix(xdf)

xstrn, xstst, ytrn, ytst = train_test_split(xs, ys, test_size=0.20, random_state=420)
dtrain = xgb.DMatrix(xstrn, label = ytrn)
dtest = xgb.DMatrix(xstst, label = ytst)

ntrees = 1000
seedval= 0
lossf = 'auc'
depthval=3
etaval = 0.01
# Ntrees, Eta, and Maxdepth
xgb_vars = product((200,500,1000),(0.01,0.03,0.05),(2,3,5,9))
xgb_vars[0]

param = {'eta':etaval,'max_depth':depthval,'silent':0,
     'objective':'binary:logistic'}
xgbout = xgb.cv(param, dtrain, ntrees, nfold=10, metrics={lossf}, seed=seedval)

xgb_history = []
for i in xgb_vars:    
    param = {'eta':i[1],'max_depth':i[2],'silent':1,'objective':'binary:logistic'}
    xgbout = xgb.cv(param, xstrn,ntrees,nfold=i[0], metrics={lossf},seed=seedval)
    xgb_history.append(xgbout)

TrainER = xgbout['train-auc-mean']
TrainSE = xgbout['train-auc-std']
TestER = xgbout['test-auc-mean']
TestSE = xgbout['test-auc-std']
Iteration= range(len(TestSE))

Trn1 = np.array(TrainER)-np.array(TrainSE)
Trn2 = np.array(TrainER)+np.array(TrainSE)
Tst1 = np.array(TestER)-np.array(TestSE)
Tst2 = np.array(TestER)+np.array(TestSE)

# Round the nearest 10th (decimal point)
minyval = round(np.min([Trn1,Tst1,Tst1,Tst2]),1)

plt.plot(Iteration, TrainER,'rs-',label='Train Error')
plt.ylim(minyval,1)
plt.grid()
plt.plot(Iteration,Trn1,'r--')
plt.plot(Iteration,Trn2,'r--')
plt.plot(Iteration,TestER, 'bs-',label='Test Error')
plt.plot(Iteration,Tst1,'b--')
plt.plot(Iteration,Tst2,'b--')
plt.legend(loc='upper right')
plt.show()

param = {'max_depth':3,'eta':0.01,'silent':0,'objective':'binary:logistic'}
dtrain = xgb.DMatrix(xstrn, label = ytrn)
dtest = xgb.DMatrix(xstst, label = ytst)
watchlist  = [(dtest,'eval'), (dtrain,'train')]
bst = xgb.train(param, dtrain, np.argmax(TestER), watchlist)

# Providing Model predictions and taking the argmax for each class
probtrn= bst.predict(dtrain)
probtst = bst.predict(dtest)
predclss = [i.argmax() for i in probtrn]

roc_perf(ytrn,probtrn,ytst,probtst)

import operator
importance = bst.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))
df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()
#df =  df.ix[0:-20,:]
ldf = pd.merge(df,cdf,left_on='feature',right_on='feature',how='left')
print ldf.sort_values(by='fscore',ascending=False).head(20)
plt.figure()
ldf['fscore'].plot()
ldf = ldf.iloc[-20:,:]
ldf.plot(kind='barh', x='Word', y='fscore', legend=False, figsize=(12, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.show()