import os
import math
import theano
import seaborn
import itertools 
import numpy as np
import pandas as pd

import keras
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Activation
from keras.regularizers import l2, activity_l2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from matplotlib import cm


def build_data():
    x1 = np.arange(-100,100)
    x2 = np.arange(-100,100)
    xs = np.asarray(list(itertools.product(x1,x2)))
    ys = np.zeros( (len(xs),))

    # This defines our function (i.e., f(x) = y)
    clist = [-75, -50, -25, 25, 50, 75]
    olist = [10, 25, 50, 75, 100]
    for c, o in zip(clist, olist):
        flt = (xs[:,0]> c) & (xs[:,1] > c)
        ys[np.where(flt),] = o

    return xs, ys

def learn_gbm(X_train, y_train, X_test, y_test, ntrees=1000):
    estimator = ExtraTreesRegressor(n_estimators=ntrees, max_features='auto', 
                            random_state=420, verbose=False)
    estimator.fit(X_train, y_train)
    yprd_tst = estimator.predict(X_test)
    err = yprd_tst - y_test
    return yprd_tst, err

def learn_mlp(X_train, y_train, X_test, y_test, nhidden=10, n_neurons=200, nepochs=200):
    model = Sequential()
    # Initial layer
    model.add(Dense(n_neurons, input_dim=2, activation='relu'))
    # Creating nhidden number of layers 
    for i in range(nhidden):
        model.add(Dense(n_neurons, activation='relu', W_regularizer=l2(0.01),
                        activity_regularizer=activity_l2(0.01)))

    model.add(Dense(1))
    adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mse', optimizer=adam)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(X_train, y_train,
              nb_epoch=nepochs, batch_size=50,
              validation_data=(X_test, y_test),
              callbacks=[early_stopping])

    yprd_tstnn = model.predict(X_test)[:,0]
    errnn = yprd_tstnn - y_test
    return yprd_tstnn, errnn

def save3dfig(X, Y, Z, title, fileloc, cmapc=cm.hot):
    fig = plt.figure(figsize=(12,12))
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(X, Y, Z, cmap=cmapc)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(fileloc)

def main():
    # Create dataset
    xs, ys = build_data()
    # Split into test and training
    X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=0.5, random_state=420)

    # Learn GBM
    yprd_tst, err = learn_gbm(X_train, y_train, X_test, y_test, ntrees=1000)
    # Learn NN (MLP)
    yprd_tstnn, errnn = learn_mlp(X_train, y_train, X_test, y_test, 
                                nhidden=10, n_neurons=200, nepochs=200)

    # Collecting errors
    rmse_gbm = np.sqrt( ((yprd_tst - y_test)**2).sum())
    rmse_nns = np.sqrt( ((yprd_tstnn - y_test)**2).sum())

    print("The RMSE of the GBM is %0.3f" % rmse_gbm)
    print("The RMSE of the NN is %0.3f" % rmse_nns)
    print("The GBM/NN RMSE = %0.3f" % (rmse_gbm / rmse_nns) )

    sav3dfig(xs[:,0], xs[:,1], ys, 
        title='True Step Function in 2-d', 
        fileloc='./images/stepfunction_true.png')

    sav3dfig(X_test[:,0], X_test[:,1], yprd_tst, 
        title='Approximated Step Function in 2-d \n (Gradient Boosting)', 
        fileloc='./images/stepfunction_gbm.png')

    sav3dfig(X_test[:,0], X_test[:,1], yprd_tstnn, 
        title='Approximated Step Function in 2-d \n (Neural Network)', 
        fileloc='./images/stepfunction_mlp.png')

    sav3dfig(X_test[:,0], X_test[:,1], err, 
        title='Residuals of Learned Step Function in 2-d \n (Gradient Boosting)', 
        fileloc='./images/stepfunction_gbm_nn.png', cmapc=cm.RdBu_r)

    sav3dfig(X_test[:,0], X_test[:,1], errnn, 
        title='Residuals of Learned Step Function in 2-d \n (Neural Network)', 
        fileloc='./images/stepfunction_mlp_res.png', cmapc=cm.RdBu_r)

if __name__=='__main__':
    main()