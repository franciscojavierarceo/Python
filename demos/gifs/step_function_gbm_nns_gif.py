import os
import math
import keras
import theano
import seaborn
import imageio
import itertools
import numpy as np
import pandas as pd

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
    x1 = np.arange(-100, 100)
    x2 = np.arange(-100, 100)
    xs = np.asarray(list(itertools.product(x1, x2)))
    ys = np.zeros((len(xs),))

    # This defines our function (i.e., f(x) = y)
    clist = [-75, -50, -25, 25, 50, 75]
    olist = [10, 25, 50, 75, 100]
    for c, o in zip(clist, olist):
        flt = (xs[:, 0] > c) & (xs[:, 1] > c)
        ys[
            np.where(flt),
        ] = o

    return xs, ys


def learn_gbm(X_train, y_train, X_test, y_test, ntrees=1000):
    estimator = GradientBoostingRegressor(
        n_estimators=ntrees,
        max_features="auto",
        random_state=420,
        verbose=False,
        learning_rate=0.2,
    )

    estimator.fit(X_train, y_train)
    yprd_tst = estimator.predict(X_test)
    err = yprd_tst - y_test
    return yprd_tst, err


def learn_mlp(X_train, y_train, X_test, y_test, nhidden=10, n_neurons=200, nepochs=200):
    model = Sequential()
    # Initial layer
    model.add(Dense(n_neurons, input_dim=2, activation="relu"))
    # Creating nhidden number of layers
    for i in range(nhidden):
        model.add(
            Dense(
                n_neurons,
                activation="relu",
                W_regularizer=l2(0.01),
                activity_regularizer=activity_l2(0.01),
            )
        )

    model.add(Dense(1))
    adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss="mse", optimizer=adam)

    early_stopping = EarlyStopping(monitor="val_loss", patience=5)
    model.fit(
        X_train,
        y_train,
        nb_epoch=nepochs,
        batch_size=50,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
    )

    yprd_tstnn = model.predict(X_test)[:, 0]
    errnn = yprd_tstnn - y_test
    return yprd_tstnn, errnn


def save3dfig(X, Y, Z, title, fileloc, cmapc, z1, z2):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca(projection="3d")
    ax.plot_trisurf(X, Y, Z, cmap=cmapc)
    ax.set_zlim(z1, z2)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig(fileloc)
    pl.tclose()


def runSimulation(n_iters=100):
    # Create dataset
    xs, ys = build_data()
    # Split into test and training
    X_train, X_test, y_train, y_test = train_test_split(
        xs, ys, test_size=0.5, random_state=420
    )

    # Learn GBM
    yprd_tst, err = learn_gbm(X_train, y_train, X_test, y_test, ntrees=n_iters)
    # Learn NN (MLP)
    yprd_tstnn, errnn = learn_mlp(
        X_train, y_train, X_test, y_test, nhidden=10, n_neurons=200, nepochs=n_iters
    )

    # Collecting errors
    rmse_gbm = np.sqrt(((yprd_tst - y_test) ** 2).sum())
    rmse_nns = np.sqrt(((yprd_tstnn - y_test) ** 2).sum())

    print("The RMSE of the GBM is %0.3f" % rmse_gbm)
    print("The RMSE of the NN is %0.3f" % rmse_nns)
    print("The GBM/NN RMSE = %0.3f" % (rmse_gbm / rmse_nns))

    save3dfig(
        X_test[:, 0],
        X_test[:, 1],
        yprd_tst,
        title="Approximated Step Function in 2-d \n (Gradient Boosting Trees = %i)"
        % n_iters,
        fileloc="./images/stepfunction_gbm_%s.png" % str(n_iters).zfill(4),
        cmapc=cm.hot,
        z1=0,
        z2=100,
    )

    save3dfig(
        X_test[:, 0],
        X_test[:, 1],
        yprd_tstnn,
        title="Approximated Step Function in 2-d \n (Neural Network, Epochs = %i)"
        % n_iters,
        fileloc="./images/stepfunction_mlp_%s.png" % str(n_iters).zfill(4),
        cmapc=cm.hot,
        z1=-20,
        z2=140,
    )

    save3dfig(
        X_test[:, 0],
        X_test[:, 1],
        err,
        title="Residuals of Learned Step Function in 2-d \n (Gradient Boosting, Trees = %i)"
        % n_iters,
        fileloc="./images/stepfunction_gbmres_%s.png" % str(n_iters).zfill(4),
        cmapc=cm.RdBu_r,
        z1=-10,
        z2=10,
    )

    save3dfig(
        X_test[:, 0],
        X_test[:, 1],
        errnn,
        title="Residuals of Learned Step Function in 2-d \n (Neural Network, Epochs = %i)"
        % n_iters,
        fileloc="./images/stepfunction_mlpres_%s.png" % str(n_iters).zfill(4),
        cmapc=cm.RdBu_r,
        z1=-30,
        z2=30,
    )


def build_gif(model):
    file_names = []
    for fn in os.listdir("./images"):
        if fn.startswith(model) and fn.endswith(".png"):
            file_names.append(fn)

    images = [Image.open("./images/" + fn) for fn in file_names]
    writeGif("images/%s.gif" % model.replace("_", ""), images, duration=1.0)


def main():
    models = [
        "stepfunction_gbm_",
        "stepfunction_mlp_",
        "stepfunction_gbmres",
        "stepfunction_mlpres",
    ]
    for iter_val in range(1, 1000, 25):
        runSimulation(iter_val)

    for model in models:
        build_gif(model=model)


if __name__ == "__main__":
    main()
