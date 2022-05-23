import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score


def liftchart(df: pd.DataFrame, actual: str, predicted: str, buckets: int = 10) -> None:
    # Bucketing the predictions (Deciling is the default)
    df["predbucket"] = pd.qcut(x=df[predicted], q=buckets)
    # Getting the performance
    aucperf = roc_auc_score(df[actual], df[predicted])
    sdf = (
        df[[actual, predicted, "predbucket"]]
        .groupby(by=["predbucket"])
        .agg({actual: [np.mean, sum, len], predicted: np.mean})
    )
    sdf.columns = sdf.columns.map("".join)  # I hate pandas multi-indexing
    sdf = sdf.rename(
        {
            actual + "mean": "Actual Default Rate",
            predicted + "mean": "Predicted Default Rate",
        },
        axis=1,
    )
    sdf[["Actual Default Rate", "Predicted Default Rate"]].plot(
        kind="line", style=".-", grid=True, figsize=(12, 8), color=["red", "blue"]
    )
    plt.ylabel("Default Rate")
    plt.xlabel("Decile Value of Predicted Default")
    plt.title(
        "Actual vs Predicted Default Rate sorted by Predicted Decile \nAUC = %.3f"
        % aucperf
    )
    plt.xticks(np.arange(sdf.shape[0]), sdf["Predicted Default Rate"].round(3))
    plt.show()


def main(n: int) -> None:
    # Generating the data
    np.random.seed(0)
    x_1 = np.random.poisson(lam=5, size=n)
    x_2 = np.random.poisson(lam=2, size=n)
    x_3 = np.random.poisson(lam=12, size=n)
    e = np.random.normal(size=n, loc=0, scale=1.0)

    # Setting the coefficient values to give us a ~5% default rate
    b_1, b_2, b_3 = -0.005, -0.03, -0.15
    ylogpred = x_1 * b_1 + x_2 * b_2 + x_3 * b_3 + e
    yprob = 1.0 / (1.0 + np.exp(-ylogpred))
    yclass = np.where(yprob >= 0.5, 1, 0)
    xs = np.hstack([x_1.reshape(n, 1), x_2.reshape(n, 1), x_3.reshape(n, 1)])
    # Adding an intercept to the matrix
    xs = sm.add_constant(xs)
    model = sm.Logit(yclass, xs)
    # All that work just to run .fit(), how terribly uninteresting
    res = model.fit()
    print(res.summary())

    # Using the model from before!
    pdf = pd.DataFrame(xs, columns=["intercept", "x1", "x2", "x3"])
    pdf["preds"] = res.predict(xs)
    pdf["actual"] = yclass
    # Finally, what we all came here to see
    liftchart(pdf, "actual", "preds", 10)
    # This is what it looks like when we have perfect information
    pdf["truth"] = pdf["actual"] + np.random.uniform(
        low=0, high=0.001, size=pdf.shape[0]
    )
    liftchart(pdf, "actual", "truth", 10)


if __name__ == "__main__":
    main(10000)
