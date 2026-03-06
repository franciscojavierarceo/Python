"""Feast feature definitions for the driver ranking demo."""
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float64, Int64

# Define the driver entity
driver = Entity(
    name="driver_id",
    value_type=ValueType.INT64,
    description="Driver identifier",
)

# Define the data source
driver_stats_source = FileSource(
    path="data/driver_stats.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

# Define the feature view
driver_hourly_stats = FeatureView(
    name="driver_hourly_stats",
    entities=[driver],
    ttl=timedelta(days=365),
    schema=[
        Field(name="conv_rate", dtype=Float64),
        Field(name="acc_rate", dtype=Float64),
        Field(name="avg_daily_trips", dtype=Int64),
    ],
    source=driver_stats_source,
)

# Define an on-demand feature view for derived features
@on_demand_feature_view(
    sources=[driver_hourly_stats],
    schema=[Field(name="conv_acc_ratio", dtype=Float64)],
)
def driver_ratios(inputs):
    df = inputs.copy()
    df["conv_acc_ratio"] = df["conv_rate"] / (df["acc_rate"] + 1e-6)
    return df[["conv_acc_ratio"]]
