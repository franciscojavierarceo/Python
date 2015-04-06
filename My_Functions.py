# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 15:57:19 2014

@author: farceo
"""
# To clear variables use "%reset"

import numpy as np
import pandas as pd
import networkx as nx
import pulp
import os
import glob
import sys
import pylab
import sklearn as sk
import random 
from sklearn.linear_model import enet_path

def summarize(x):
    import pandas as pd
    for i in mydf.columns:
        print i
        if isinstance(np.array(mydf[i]),int)==True:
            print np.mean(mydf[i])

def dim(mydf):
    import pandas as pd
    out = mydf.shape
    return out

def ndistinct(x):
    import numpy as np
    out = len(np.unique(x))
    print "There are",out,"distinct values."

def gini(actual,pred,weight=None):
    import scipy
    import numpy as np
    import pandas as pd
    pdf= pd.DataFrame(scipy.vstack([actual,pred]).T,columns=['Actual','Predicted'],)
    pdf= pdf.sort(columns='Predicted')
    if weight is None:
        pdf['Weight'] = 1.0
  
    pdf['CummulativeWeight'] = np.cumsum(pdf['Weight'])
    pdf['CummulativeWeightedActual'] = np.cumsum(pdf['Actual']*pdf['Weight'])
    TotalWeight = sum(pdf['Weight'])
    Numerator = sum(pdf['CummulativeWeightedActual']*pdf['Weight'])
    Denominator = sum(pdf['Actual']*pdf['Weight']*TotalWeight)
    Gini = 1.0 - 2.0 * Numerator/Denominator
    return Gini 

def normgini(actual,pred,Val=None):
    return gini(actual,pred,weight=Val) / gini(actual,actual,weight=Val)

def mylift(actual,pred,weight=None,n=10,xlab='Predicted Decile',MyTitle='Model Performance Lift Chart'):
    import scipy
    import numpy as np
    import pandas as pd
    import pylab
    pdf= pd.DataFrame(scipy.vstack([actual,pred]).T,columns=['Actual','Predicted'],)
    pdf= pdf.sort(columns='Predicted')
    if weight is None:
        pdf['Weight'] = 1.0
  
    pdf['CummulativeWeight'] = np.cumsum(pdf['Weight'])
    pdf['CummulativeWeightedActual'] = np.cumsum(pdf['Actual']*pdf['Weight'])
    TotalWeight = sum(pdf['Weight'])
    Numerator = sum(pdf['CummulativeWeightedActual']*pdf['Weight'])
    Denominator = sum(pdf['Actual']*pdf['Weight']*TotalWeight)
    Gini = 1.0 - 2.0 * Numerator/Denominator
    NormalizedGini = Gini/ gini(pdf['Actual'],pdf['Actual'])
    GiniTitle = 'Normalized Gini = '+ str(round(NormalizedGini,4))
    
    pdf['PredictedDecile'] = np.round(pdf['CummulativeWeight']*n /TotalWeight + 0.5,decimals=0)
    pdf['PredictedDecile'][pdf['PredictedDecile'] < 1.0] = 1.0
    pdf['PredictedDecile'][pdf['PredictedDecile'] > n] = n 
    
    pdf['WeightedPrediction'] = pdf['Predicted']*pdf['Weight']
    pdf['WeightedActual'] = pdf['Actual']*pdf['Weight']
    lift_df = pdf.groupby('PredictedDecile').agg({'WeightedPrediction': np.sum,'Weight':np.sum,'WeightedActual':np.sum,'PredictedDecile':np.size})
    nms = lift_df.columns.values
    nms[1] = 'Count'
    lift_df.columns = nms
    lift_df['AveragePrediction'] = lift_df['WeightedPrediction']/lift_df['Weight']
    lift_df['AverageActual'] = lift_df['WeightedActual']/lift_df['Weight']
    lift_df['AverageError'] = lift_df['AverageActual']/lift_df['AveragePrediction']
    
    d = pd.DataFrame(lift_df.index)
    p = lift_df['AveragePrediction']
    a = lift_df['AverageActual']
    pylab.plot(d,p,label='Predicted',color='blue',marker='o')
    pylab.plot(d,a,label='Actual',color='red',marker='d')
    pylab.legend(['Predicted','Actual'])
    pylab.title(MyTitle +'\n'+GiniTitle)
    pylab.xlabel(xlab)
    pylab.ylabel('Actual vs. Predicted')
    pylab.grid()
    pylab.show()
    
def deciles(var):
    import numpy as np
    import pandas as pd
    out = []
    decile = [i * 10 for i in range(0,11)]
    for i in decile:
        out.append(np.percentile(var,i))
    
    outdf= pd.DataFrame()
    outdf['Decile'] = decile
    outdf['Value'] = out
    return outdf
    
def auc(actual,pred):
    from sklearn import metrics
    import scipy
    import numpy as np
    import pandas as pd
    fpr, tpr, thresholds = metrics.roc_curve(actual, pred)
    return metrics.auc(fpr, tpr)

def roc_plot(actual,pred):
    import pylab as pl
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(actual, pred)
    roc_auc = auc(fpr, tpr)
    print "Area under the ROC curve : %f" % roc_auc
    # Plot ROC curve
    pl.clf()
    pl.plot(fpr, tpr, color='red',label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('ROC Curve')
    pl.legend(loc="lower right")
    pl.show()
    
def histogram(xvar,nbins=50):
    import matplotlib.pyplot as plt
    plt.hist(xvar,bins=nbins)
    plt.grid()
    plt.show()
    
    
def denplot(xvar,xlbl='Variable'):
    import matplotlib.pyplot as plt
    from sklearn.neighbors.kde import KernelDensity
    import numpy as np
    xvar = np.array(xvar)
    xvar = xvar.reshape(-1,1)
    kde = KernelDensity(bandwidth=0.2).fit(xvar)
    x = np.linspace(xvar.min(), xvar.max(), len(xvar)).reshape(-1, 1)
    density = np.exp(kde.score_samples(x))
    plt.plot(x, density)
    plt.plot(xvar, xvar * 0, 'ok', alpha=.03)
    plt.ylim(-.001, .035)
    plt.xlabel(xlbl)
    plt.grid()
    plt.ylabel("Density")
    plt.show()
    
def cdfplot(xvar):
    import matplotlib.pyplot as plt
    from sklearn.neighbors.kde import KernelDensity
    import numpy as np
    sortedvals=np.sort( xvar)
    yvals=np.arange(len(sortedvals))/float(len(sortedvals))
    plt.plot( sortedvals, yvals )
    plt.grid()
    plt.show()

def ptable(df,var,asc=False):
    import numpy as np
    import pandas as pd
    outdf = df.groupby(var).count().reset_index().ix[:,0:2]
    outdf.columns = [outdf.columns[0],'Count']
    outdf = outdf.sort(columns='Count',ascending=asc)
    return outdf


def barplot(df,var,MyTitle="",aval=0.9):
    # Taken from a pandas summary file
    import numpy as np
    import matplotlib.pyplot as plt
    out = ptable(df,var,asc=True)
    indxvl = np.arange(len(out[var]))
    fig = plt.figure(figsize=(5.5,3),dpi=100)
    ax = fig.add_subplot(111)
    ax.grid(True,which='both')
    bar = ax.barh(indxvl, out['Count'], xerr=0, align='center', alpha=aval)
    plt.yticks(indxvl, out[var])
    plt.xlabel('')
    fnt = {'family':'normal','weight':'bold','size':10}
    plt.rc('font',**fnt)
    plt.title(MyTitle)
    fig.tight_layout()    
    plt.show()

def scatplot(x,y,colors='blue',MyTitle='',size=1):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.scatter(x, y, s=size, c=colors, alpha=0.5)
    plt.show()
    plt.grid()
