"""
Feast + MLflow Demo: Feature Selection Experiments
Based on the blog post: Feast + MLflow + Kubeflow: A Unified AI/ML Lifecycle

This demo shows how to use MLflow to compare model performance across
different feature combinations from Feast, enabling data-driven feature selection.
"""
import mlflow
import mlflow.sklearn
from feast import FeatureStore
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from datetime import datetime
from itertools import combinations


def run_experiment(X_train, X_test, y_train, y_test, feature_cols, model_type="LogisticRegression"):
    """Run a single experiment with given features and model, logging to MLflow."""

    X_train_subset = X_train[feature_cols]
    X_test_subset = X_test[feature_cols]

    # Create run name from features
    feature_str = "+".join(feature_cols)
    run_name = f"{model_type}_{feature_str}"

    with mlflow.start_run(run_name=run_name):
        # Log feature configuration
        mlflow.log_param("features", feature_cols)
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_param("model_type", model_type)

        # Log individual features as tags for easier filtering
        for feat in feature_cols:
            mlflow.set_tag(f"feature_{feat}", "included")

        # Train model
        if model_type == "LogisticRegression":
            model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            model = RandomForestClassifier(random_state=42, n_estimators=100)

        model.fit(X_train_subset, y_train)

        # Evaluate
        y_pred = model.predict(X_test_subset)
        y_prob = model.predict_proba(X_test_subset)[:, 1]

        # Log metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_prob),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }

        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        return run_name, metrics


def main():
    print("=" * 60)
    print("Feast + MLflow Feature Selection Demo")
    print("=" * 60)

    # Initialize the feature store
    store = FeatureStore(repo_path=".")

    # Step 1: Retrieve all available features from Feast
    print("\n[Step 1] Retrieving features from Feast...")
    entity_df = pd.read_parquet("data/driver_labels.parquet")

    all_features = [
        "driver_hourly_stats:conv_rate",
        "driver_hourly_stats:acc_rate",
        "driver_hourly_stats:avg_daily_trips",
    ]

    feature_df = store.get_historical_features(
        entity_df=entity_df,
        features=all_features,
    ).to_df()

    feature_df = feature_df.dropna()
    print(f"Retrieved {len(feature_df)} samples with {len(all_features)} features")

    # Step 2: Prepare data
    print("\n[Step 2] Preparing train/test split...")
    feature_cols = ["conv_rate", "acc_rate", "avg_daily_trips"]
    X = feature_df[feature_cols]
    y = feature_df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training: {len(X_train)} samples, Test: {len(X_test)} samples")

    # Step 3: Run feature selection experiments
    print("\n[Step 3] Running feature selection experiments...")
    mlflow.set_experiment("feast-feature-selection")

    results = []

    # Try all feature combinations (1, 2, and 3 features)
    for num_features in range(1, len(feature_cols) + 1):
        for feature_subset in combinations(feature_cols, num_features):
            feature_list = list(feature_subset)

            # Test with Logistic Regression
            run_name, metrics = run_experiment(
                X_train, X_test, y_train, y_test,
                feature_list, "LogisticRegression"
            )
            results.append({"run": run_name, **metrics})
            print(f"  {run_name}: AUC={metrics['auc']:.4f}")

            # Test with Random Forest
            run_name, metrics = run_experiment(
                X_train, X_test, y_train, y_test,
                feature_list, "RandomForest"
            )
            results.append({"run": run_name, **metrics})
            print(f"  {run_name}: AUC={metrics['auc']:.4f}")

    # Print summary
    print("\n" + "=" * 60)
    print("Experiment Summary (sorted by AUC)")
    print("=" * 60)
    results_df = pd.DataFrame(results).sort_values("auc", ascending=False)
    print(results_df.to_string(index=False))

    print("\n" + "=" * 60)
    print(f"Total experiments: {len(results)}")
    print(f"Best model: {results_df.iloc[0]['run']} (AUC: {results_df.iloc[0]['auc']:.4f})")
    print("=" * 60)
    print("\nView results in MLflow UI: http://localhost:5000")
    print("Compare runs to see which features improve model performance!")

    # Step 4: Materialize features to online store
    print("\n[Step 4] Materializing features to online store...")
    store.materialize_incremental(end_date=datetime.utcnow())
    print("Features materialized successfully!")

    # Step 5: Test online feature retrieval
    print("\n[Step 5] Testing online feature retrieval...")
    test_driver_ids = [1039, 1029, 1015]

    for driver_id in test_driver_ids:
        features = store.get_online_features(
            features=all_features,
            entity_rows=[{"driver_id": driver_id}],
        ).to_dict()

        print(f"\nDriver {driver_id}:")
        for col in feature_cols:
            print(f"  {col}: {features[col][0]}")

if __name__ == "__main__":
    main()
