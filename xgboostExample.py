import sys
sys.path.append('/Users/franciscojavierarceo/GitHub/xgboost/wrapper/')
import os
# This is necessary for running in anaconda, which is annoying
os.chdir('/Users/franciscojavierarceo/GitHub/xgboost/wrapper/')
import xgboost as xgb
import numpy as np
import scipy.sparse 

dtrain = xgb.DMatrix('/Users/franciscojavierarceo/Data/xgboost/agaricus.txt.train')
dtest = xgb.DMatrix('/Users/franciscojavierarceo/Data/xgboost/agaricus.txt.test')
# Parameters in xgboost
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }

# specify validations set to watch performance
watchlist  = [(dtest,'eval'), (dtrain,'train')]
num_round = 20
bst = xgb.train(param, dtrain, num_round, watchlist)

print "XGboost worked, yay!"

# This is prediction
preds = bst.predict(dtest)
labels = dtest.get_label()
print ('error=%f' % (  sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))
bst.save_model('0001.model')
# dump model
bst.dump_model('dump.raw.txt')
# dump model with feature map
bst.dump_model('dump.nice.txt','/Users/franciscojavierarceo/Data/xgboost/featmap.txt')
# save dmatrix into binary buffer
dtest.save_binary('dtest.buffer')
bst.save_model('xgb.model')
# load model and data in 
bst2 = xgb.Booster(model_file='xgb.model')
dtest2 = xgb.DMatrix('dtest.buffer')
preds2 = bst2.predict(dtest2)
# assert they are the same
assert np.sum(np.abs(preds2-preds)) == 0
# build dmatrix from scipy.sparse
print ('start running example of build DMatrix from scipy.sparse CSR Matrix')
labels = []
row = []; col = []; dat = []
i = 0
for l in open('/Users/franciscojavierarceo/Data/xgboost/agaricus.txt.train'):
    arr = l.split()
    labels.append( int(arr[0]))
    for it in arr[1:]:
        k,v = it.split(':')
        row.append(i); col.append(int(k)); dat.append(float(v))
    i += 1
csr = scipy.sparse.csr_matrix( (dat, (row,col)) )
dtrain = xgb.DMatrix( csr, label = labels )
watchlist  = [(dtest,'eval'), (dtrain,'train')]
bst = xgb.train( param, dtrain, num_round, watchlist )

print ('start running example of build DMatrix from scipy.sparse CSC Matrix')
# we can also construct from csc matrix
csc = scipy.sparse.csc_matrix( (dat, (row,col)) )
dtrain = xgb.DMatrix(csc, label=labels)
watchlist  = [(dtest,'eval'), (dtrain,'train')]
bst = xgb.train( param, dtrain, num_round, watchlist )

print ('start running example of build DMatrix from numpy array')
# NOTE: npymat is numpy array, we will convert it into scipy.sparse.csr_matrix in internal implementation
# then convert to DMatrix
npymat = csr.todense()
dtrain = xgb.DMatrix(npymat, label = labels)
watchlist  = [(dtest,'eval'), (dtrain,'train')]
bst = xgb.train( param, dtrain, num_round, watchlist )
print '***************************************************************************'
print '********* Below is output of the Boost from Prediction File ***************'
print '***************************************************************************'
watchlist  = [(dtest,'eval'), (dtrain,'train')]
###
# advanced: start from a initial base prediction
###
print ('start running example to start from a initial prediction')
# specify parameters via map, definition are same as c++ version
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
# train xgboost for 1 round
bst = xgb.train( param, dtrain, 1, watchlist )
# Note: we need the margin value instead of transformed prediction in set_base_margin
# do predict with output_margin=True, will always give you margin values before logistic transformation
ptrain = bst.predict(dtrain, output_margin=True)
ptest  = bst.predict(dtest, output_margin=True)
dtrain.set_base_margin(ptrain)
dtest.set_base_margin(ptest)

print ('this is result of running from initial prediction')
bst = xgb.train( param, dtrain, 1, watchlist )
#=========================================================================================
# 						End 
#=========================================================================================