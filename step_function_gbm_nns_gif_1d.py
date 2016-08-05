import os
import math
import theano
import seaborn
import imageio
import itertools
import numpy as np
import pandas as pd

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras

from PIL import Image
from images2gif import writeGif
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Activation
from keras.regularizers import l2, activity_l2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from matplotlib import cm

def build_data():
    xs = np.arange(-100,100, 0.1).reshape((2000))
    ys = np.zeros( (len(xs)))
    print(xs.shape, ys.shape)
    # This defines our function (i.e., f(x) = y)
    clist = [-75, -50, -25, 25, 50, 75]
    olist = [10, 25, 50, 75, 100]

    for c, o in zip(clist, olist):
        ys[np.where(xs> c),] = o

    return xs, ys

def learn_gbm(X_train, y_train, X_test, y_test, ntrees=10000):
    estimator = GradientBoostingRegressor(
                    n_estimators=ntrees,
                    random_state=420,
                    verbose=False,
                    learning_rate=0.2,
                    max_depth=2)

    estimator.fit(X_train[:, None], y_train[:, None])
    yprd_tst = estimator.predict(X_test[:, None])
    err = yprd_tst - y_test
    return yprd_tst, err

def learn_mlp(X_train, y_train, X_test, y_test, nhidden=10, n_neurons=200, nepochs=200):
    model = Sequential()
    # Initial layer
    model.add(Dense(n_neurons,
                    input_dim=np.minimum(X_train.shape[-1], len(X_train.shape)),
                    activation='relu'))
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

def save2dfig(X, Y, title, fileloc):
    plt.scatter(X, Y)
    plt.title(title)
    plt.savefig(fileloc)
    plt.close()

def runSimulation(n_iters=100):
    # Create dataset
    xs, ys = build_data()
    # Split into test and training
    X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=0.5, random_state=420)

    # Learn GBM
    yprd_tst, err = learn_gbm(X_train, y_train, X_test, y_test, ntrees=n_iters)
    # Learn NN (MLP)
    yprd_tstnn, errnn = learn_mlp(X_train, y_train, X_test, y_test,
                                nhidden=10, n_neurons=200, nepochs=n_iters)

    # Collecting errors
    rmse_gbm = np.sqrt( ((yprd_tst - y_test)**2).sum())
    rmse_nns = np.sqrt( ((yprd_tstnn - y_test)**2).sum())

    print("The RMSE of the GBM is %0.3f" % rmse_gbm)
    print("The RMSE of the NN is %0.3f" % rmse_nns)
    print("The GBM/NN RMSE = %0.3f" % (rmse_gbm / rmse_nns) )

    save2dfig(X_test, yprd_tst,
        title='Approximated Step Function in 1-d \n (Gradient Boosting Trees = %i)' % n_iters,
        fileloc='./images/stepfunction_1d_gbm_%s.png' % str(n_iters).zfill(4))

    save2dfig(X_test,  yprd_tstnn,
        title='Approximated Step Function in 1-d \n (Neural Network, Epochs = %i)' % n_iters,
        fileloc='./images/stepfunction_1d_mlp_%s.png' % str(n_iters).zfill(4))

    save2dfig(X_test,  err,
        title='Residuals of Learned Step Function in 1-d \n (Gradient Boosting, Trees = %i)' % n_iters,
        fileloc='./images/stepfunction_1d_gbmres_%s.png' % str(n_iters).zfill(4))

    save2dfig(X_test,  errnn,
        title='Residuals of Learned Step Function in 1-d \n (Neural Network, Epochs = %i)' % n_iters,
        fileloc='./images/stepfunction_1d_mlpres_%s.png' % str(n_iters).zfill(4))

def build_gif(model):
    file_names = []
    for fn in os.listdir('./images'):
        if fn.startswith(model) and fn.endswith('.png'):
            file_names.append(fn)

    images = [Image.open('./images/'+fn) for fn in file_names ]
    writeGif('images/%s.gif' % model.replace('_',''), images, duration=1.0)

def main():
    models = [
        'stepfunction_1d_gbm_',
        'stepfunction_1d_mlp_',
        'stepfunction_1d_gbmres',
        'stepfunction_1d_mlpres'
    ]
    for iter_val in range(1, 1000, 25):
        runSimulation(iter_val)

    for model in models:
        build_gif(model=model)

if __name__=='__main__':
    main()
