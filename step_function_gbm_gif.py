import os
import math
import keras
import seaborn
import imageio
import itertools 
import numpy as np
import pandas as pd

from PIL import Image
from images2gif import writeGif

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
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
    estimator = GradientBoostingRegressor(
                    n_estimators=ntrees, 
                    max_features='auto', 
                    random_state=420, 
                    verbose=0,
                    learning_rate=0.35)

    estimator.fit(X_train, y_train)
    yprd_tst = estimator.predict(X_test)
    err = yprd_tst - y_test
    return yprd_tst, err

def save3dfig(X, Y, Z, title, fileloc, cmapc, z1, z2):
    fig = plt.figure(figsize=(12,12))
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(X, Y, Z, cmap=cmapc)
    ax.set_zlim(z1, z2)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(fileloc)
    plt.close()

def runSimulation(n_iters=100):
    # Create dataset
    xs, ys = build_data()
    # Split into test and training
    X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=0.5, random_state=420)

    # Learn GBM
    yprd_tst, err = learn_gbm(X_train, y_train, X_test, y_test, ntrees=n_iters)
    # Collecting errors
    rmse_gbm = np.sqrt( ((yprd_tst - y_test)**2).sum())

    print("The RMSE of the GBM is %0.3f" % rmse_gbm)

    save3dfig(X_test[:,0], X_test[:,1], yprd_tst, 
        title='Approximated Step Function in 2-d \n (Gradient Boosting Trees = %i)' % n_iters, 
        fileloc='./images/stepfunction_gbm_%s.png' % str(n_iters).zfill(4),
        cmapc=cm.hot,
        z1 = 0, z2=100)

    save3dfig(X_test[:,0], X_test[:,1], err, 
        title='Residuals of Learned Step Function in 2-d \n (Gradient Boosting, Trees = %i)' % n_iters, 
        fileloc='./images/stepfunction_gbmres_%s.png' % str(n_iters).zfill(4),
        cmapc=cm.RdBu_r,
        z1 =-10, z2=10)

def build_gif(model):
    file_names = []
    for fn in os.listdir('./images'):
        if fn.startswith(model) and fn.endswith('.png'):
            file_names.append(fn)

    images = [Image.open('./images/'+fn) for fn in file_names ]
    writeGif('images/%s.gif' % model.replace('_',''), images, duration=1.0)

def main():
    models = [
        'stepfunction_gbm_',
        'stepfunction_gbmres',
    ]
    for iter_val in range(1, 1000, 25):
        runSimulation(iter_val)

    for model in models:
        build_gif(model=model)

if __name__=='__main__':
    main()
