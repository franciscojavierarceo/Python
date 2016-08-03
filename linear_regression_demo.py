import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn import linear_model



def simulateData(n=1e4, k=10):
    n = int(n)
    xs = np.random.random((n,k))
    betas = np.random.uniform(size=(k,1))
    epsilon = np.random.normal(size=n, scale=1.0, loc=0.).reshape((n,1))
    
    # Here I'm generating the data as just a cross product between the xs and coefficents
    yhat = xs.dot(betas)
    y = yhat + epsilon          # This adds noise to the actual data
    return xs, y, yhat


def main():
    xs, y, yhat = simulateData(n=1e4, k=10)

    print("True R-Squared = %0.4f" % ( 1-((y-yhat)**2).sum() /  ((y-y.mean())**2).sum()))

    # Fitting regression
    regr = linear_model.LinearRegression()
    regr.fit(xs, y)

    # This regression is just used to plot the line between the actual and predicted y
    regr2 = linear_model.LinearRegression()
    regr2.fit(yhat, y)

    print("Learned R-Squared = %0.4f" % ( 1-((y-regr.predict(xs))**2).sum() /  ((y-y.mean())**2).sum()))

    plt.figure(figsize=(12,8))
    plt.scatter(yhat, y, c='green')
    plt.plot(yhat, regr2.predict(yhat),color='blue', linewidth=2)
    plt.title("Plot of Prediction and Actual with Regression Line")
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    # This is just a conventional thing that allows you to run this nicely in the terminal
    main()