import os
import scipy
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve, auc, roc_auc_score


def json_parser(d: dict, path: str, delimiter: str = "."):
    """
    Parse a JSON recursively using a path and delimiter.
    This function recursively parses a JSON by using an input path
    splitting the delimiter and using that to extract
    the logic path to the data.
    Parameters
    ----------
    d: dict
        JSON or Python Dictionary object
    path: str
        Path to the data you want to extract
    delimiter: str
        Delimiter allowing you to split differently if you really need to
    Returns
    -------
        Whatever data is stored at the end of the path
    Examples
    --------
    mydict = {
        'node1': 1,
        'node2': 2,
        'node3': [
            3,
            4,
        ],
    }
    >>> json_parser(mydict, 'node1')
    1
    >>> json_parser(mydict, 'node3.1')
    4
    """
    assert isinstance(path, str), "Path must be a string"

    if delimiter not in path:
        path = int(path) if path.isdigit() else path
        if isinstance(d, list):
            n = len(d)
            assert (path + 1) <= len(d), (
                "List value outside of range, can only select %i elements" % n
            )
        else:
            assert path in d.keys(), (
                "'%s' is not an available key in current path. Please update." % path
            )

        return d[path]

    paths = path.split(delimiter)
    pkey, newpath = paths[0], ("%s" % delimiter).join(paths[1:])
    pkey = int(pkey) if pkey.isdigit() else pkey
    return json_parser(d[pkey], newpath, delimiter)


def pp(x):
    print(json.dumps(json.loads(x) if type(x) in (str, bytes) else x, indent=2))


def cdfplotdata(xdf: pd.DataFrame, xcol: str):
    """
    Helper function to create a summary data frame of the Cumulative Percentile
    of a continuous feature
    """
    tmp = pd.DataFrame(xdf[xcol].value_counts(normalize=True)).reset_index()
    tmp.columns = [xcol, "Percent"]
    tmp.sort_values(by=xcol, inplace=True)
    tmp.reset_index(drop=True, inplace=True)
    tmp["Cumulative Percent"] = tmp["Percent"].cumsum()
    tmp.set_index(xcol, inplace=True)
    return tmp["Cumulative Percent"]


def gini(actual: pd.Series, pred: pd.Series, weight: int = None):
    pdf = pd.DataFrame(
        scipy.vstack([actual, pred]).T,
        columns=["Actual", "Predicted"],
    )
    pdf = pdf.sort_values("Predicted")
    if weight is None:
        pdf["Weight"] = 1.0

    pdf["CummulativeWeight"] = np.cumsum(pdf["Weight"])
    pdf["CummulativeWeightedActual"] = np.cumsum(pdf["Actual"] * pdf["Weight"])
    TotalWeight = sum(pdf["Weight"])
    Numerator = sum(pdf["CummulativeWeightedActual"] * pdf["Weight"])
    Denominator = sum(pdf["Actual"] * pdf["Weight"] * TotalWeight)
    Gini = 1.0 - 2.0 * Numerator / Denominator
    return Gini



def mylift(
    actual,
    pred,
    weight=None,
    n=10,
    xlab="Predicted Decile",
    MyTitle="Model Performance Lift Chart",
    dualaxis=False,
    output=False,
):
    pdf = pd.DataFrame(scipy.vstack([actual, pred]).T, columns=["Actual", "Predicted"])
    pdf = pdf.sort_values("Predicted")
    if weight is None:
        pdf["Weight"] = 1.0

    pdf["CummulativeWeight"] = np.cumsum(pdf["Weight"])
    pdf["CummulativeWeightedActual"] = np.cumsum(pdf["Actual"] * pdf["Weight"])
    TotalWeight = sum(pdf["Weight"])
    Numerator = sum(pdf["CummulativeWeightedActual"] * pdf["Weight"])
    Denominator = sum(pdf["Actual"] * pdf["Weight"] * TotalWeight)
    Gini = 1.0 - 2.0 * Numerator / Denominator
    NormalizedGini = Gini / gini(pdf["Actual"], pdf["Actual"])
    GiniTitle = "Normalized Gini = " + str(round(NormalizedGini, 4))
    print(GiniTitle)

    pdf["PredictedDecile"] = np.round(
        pdf["CummulativeWeight"] * n / TotalWeight + 0.5, decimals=0
    )
    pdf["PredictedDecile"][pdf["PredictedDecile"] < 1.0] = 1.0
    pdf["PredictedDecile"][pdf["PredictedDecile"] > n] = n
    pdf["WeightedPrediction"] = pdf["Predicted"] * pdf["Weight"]
    pdf["WeightedActual"] = pdf["Actual"] * pdf["Weight"]
    lift_df = pdf.groupby("PredictedDecile").agg(
        {
            "WeightedPrediction": np.sum,
            "Weight": np.sum,
            "WeightedActual": np.sum,
            "PredictedDecile": np.size,
        }
    )
    nms = lift_df.columns.values
    nms[1] = "Count"

    lift_df.columns = nms
    lift_df["AveragePrediction"] = lift_df["WeightedPrediction"] / lift_df["Count"]
    lift_df["AverageActual"] = lift_df["WeightedActual"] / lift_df["Count"]
    lift_df["AverageError"] = lift_df["AverageActual"] / lift_df["AveragePrediction"]

    d = pd.DataFrame(lift_df.index)
    p = lift_df["AveragePrediction"]
    a = lift_df["AverageActual"]
    if dualaxis:
        fig, ax1 = plt.subplots(figsize=(12,8))
        ax2 = ax1.twinx()
        ax1.plot(d, p, label="Predicted", color="blue", marker="o")
        ax2.plot(d, a, label="Actual", color="red", marker="d")
        ax1.legend(["Predicted", "Actual"])
        ax1.title.set_text(MyTitle + "\n" + GiniTitle)
        ax1.set_xlabel(xlab)
        ax1.set_ylabel("Actual vs. Predicted")
        if isinstance(pred, pd.core.series.Series):
            ax2.set_ylabel(pred.name)
        ax1.grid()
        fig.show()
        if output:
            return lift_df
        else:
            return
    
    plt.plot(d, p, label="Predicted", color="blue", marker="o")
    plt.plot(d, a, label="Actual", color="red", marker="d")
    plt.legend(["Predicted", "Actual"])
    plt.title(MyTitle + "\n" + GiniTitle)
    plt.xlabel(xlab)
    plt.ylabel("Actual vs. Predicted")
    plt.grid()
    plt.show()
    
    if output:
        return lift_df


def roc_plot(actual, pred, ttl):
    fpr, tpr, thresholds = roc_curve(actual, pred)
    roc_auc = auc(fpr, tpr)
    print("The Area Under the ROC Curve : %f" % roc_auc)
    # Plot ROC curve
    plt.clf()
    plt.plot(fpr, tpr, color="red", label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], "k")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.grid()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve" + "\n" + ttl)
    plt.legend(loc="lower right")
    plt.show()


def roc_perf(atrn, ptrn, atst, ptst):
    fprtrn, tprtrn, thresholds = roc_curve(atrn, ptrn)
    fprtst, tprtst, thresholdstst = roc_curve(atst, ptst)
    roc_auctrn = auc(fprtrn, tprtrn)
    roc_auctst = auc(fprtst, tprtst)
    print("The Training Area Under the ROC Curve : %f" % roc_auctrn)
    print("The Test Area Under the ROC Curve : %f" % roc_auctst)
    # Plot ROC curve
    plt.clf()
    plt.plot(fprtrn, tprtrn, color="red", label="Train AUC = %0.2f" % roc_auctrn)
    plt.plot(fprtst, tprtst, color="blue", label="Test AUC = %0.2f" % roc_auctst)
    plt.plot([0, 1], [0, 1], "k")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.grid()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()


def cdfplot(xvar):
    sortedvals = np.sort(xvar)
    yvals = np.arange(len(sortedvals)) / float(len(sortedvals))
    plt.plot(sortedvals, yvals)
    plt.show()


def pt(df: pd.DataFrame, xvar: str, sortfreq: bool = True, ascending: bool = False):
    df = pd.concat([df[xvar].value_counts(), df[xvar].value_counts(True)], axis=1)
    df.columns = ["Count", "Percent"]
    df.sort_values(by="Count" if sortfreq else xvar, ascending=ascending, inplace=True)
    df["Cumulative Percent"] = df["Percent"].cumsum()
    return df


def ptbyx(df: pd.DataFrame, xvar: str, yvar: str) -> pd.DataFrame:
    tdf = df[[xvar, yvar]].groupby(by=xvar).agg({yvar: [len, np.sum, np.mean]})
    tdf.columns = tdf.columns.map("".join)  # I hate pandas multi-indexing
    tdf.rename(
        {yvar + "sum": "sum_y", yvar + "len": "count", yvar + "mean": "avg_y"},
        axis=1,
        inplace=True,
    )
    return tdf.reset_index()


def plotptbyx(df: pd.DataFrame, xvar: str, yvar: str):
    tmp = ptbyx(df, xvar, yvar)
    tmp[[xvar, "avg_y"]].plot(x=xvar, grid=True, figsize=(12, 8))
    plt.show()


def scatterplot(df: pd.DataFrame, xvar: str, yvar: str, regline: bool = True):
    plt.figure(figsize=(12, 8))
    plt.scatter(df[xvar], df[yvar])
    if regline:
        linear_regressor = LinearRegression()
        linear_regressor.fit(
            df[xvar].values.reshape(-1, 1), df[yvar].values.reshape(-1, 1)
        )
        Y_pred = linear_regressor.predict(df[xvar].values.reshape(-1, 1))
        plt.plot(df[xvar], Y_pred, color="red")
    plt.title("Scatter Plot & Regression line of \n%s and %s" % (xvar, yvar))
    plt.xlabel(xvar)
    plt.ylabel(yvar)
    plt.grid()


def build_stdm(docs, **kwargs):
    """Build Spares Term Document Matrix"""
    vectorizer = CountVectorizer(**kwargs)
    sparsematrix = vectorizer.fit_transform(docs)
    vocab = vectorizer.vocabulary_.keys()
    return sparsematrix, vocab


def pooled_regression(fulldata, trainfilter, cols, ydep):
    """
    fulldata:       :pd.DataFrame:
    trainfilter:    :bool:
    cols:           :list:
    ydep:           :str:
    """
    tmpdf = fulldata[cols + [ydep]].copy()
    training_mean = tmpdf[trainfilter][ydep].mean()
    training_var = tmpdf[trainfilter][ydep].var()

    traingroup = (
        tmpdf.ix[trainfilter]
        .groupby(cols)
        .agg({ydep: [len, "sum", "mean", "var", "max"]})
        .reset_index()
    )

    traingroup.columns = traingroup.columns.get_level_values(0)
    traingroup.columns = cols + [
        "%s_%s" % (ydep, x) for x in ["cnt", "sum", "mean", "var", "max"]
    ]

    # Here's the partial pooling component that shrinks
    # towards the global mean of the training data
    traingroup["%s_mean_pooled" % ydep] = (
        (traingroup["%s_cnt" % ydep] / training_var) * training_mean
        + (1.0 / traingroup["%s_var" % ydep]) * traingroup["%s_mean" % ydep]
    ) / (traingroup["%s_cnt" % ydep] / training_var + 1.0 / traingroup["%s_var" % ydep])

    traingroup.ix[
        traingroup["%s_cnt" % ydep] < 2, "%s_mean_pooled" % ydep
    ] = training_mean
    testdf = pd.merge(fulldata, traingroup, how="left", on=cols)
    testdf["%s_mean_pooled" % ydep] = testdf["%s_mean_pooled" % ydep].fillna(
        training_mean
    )
    return testdf[["jobId", "%s_mean_pooled" % ydep]]


def main():
    pass


if __name__ == "__main__":
    main()
