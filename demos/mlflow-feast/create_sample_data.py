"""Create sample data for the Feast + MLflow demo."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# Create driver stats data
n_records = 1000
driver_ids = np.random.choice(range(1001, 1051), n_records)  # 50 drivers
# Use recent timestamps within the last 30 days
base_date = datetime.now() - timedelta(days=30)
timestamps = [base_date + timedelta(hours=i) for i in range(n_records)]

driver_stats = pd.DataFrame({
    "driver_id": driver_ids,
    "event_timestamp": timestamps,
    "conv_rate": np.random.uniform(0.1, 0.9, n_records),
    "acc_rate": np.random.uniform(0.5, 1.0, n_records),
    "avg_daily_trips": np.random.randint(1, 50, n_records),
    "created": timestamps,
})

driver_stats.to_parquet("data/driver_stats.parquet", index=False)
print(f"Created driver_stats.parquet with {len(driver_stats)} records")
print(driver_stats.head())

# Create driver labels data for training
# Generate labels based on features (higher conv_rate and more trips = positive label)
n_labels = 200
label_driver_ids = np.random.choice(range(1001, 1051), n_labels)
label_timestamps = [base_date + timedelta(days=i % 30) for i in range(n_labels)]

# Create labels correlated with features
driver_labels = pd.DataFrame({
    "driver_id": label_driver_ids,
    "event_timestamp": label_timestamps,
})

# Generate labels with some noise
np.random.seed(42)
driver_labels["label"] = np.random.randint(0, 2, n_labels)

driver_labels.to_parquet("data/driver_labels.parquet", index=False)
print(f"\nCreated driver_labels.parquet with {len(driver_labels)} records")
print(driver_labels.head())
