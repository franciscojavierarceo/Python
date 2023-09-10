# This is an example feature definition file
import os
import pandas as pd
from datetime import timedelta, datetime

from feast import (
    Entity,
    FeatureView,
    Field,
    FileSource,
    RequestSource,
    ValueType,
)
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Int64, String

# Read data from parquet files. Parquet is convenient for local development mode. For
# production, you can use your favorite DWH, such as BigQuery. See Feast documentation
# for more info.
driver_hourly_stats = FileSource(
    path=os.path.abspath("data/driver_stats.parquet"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

driver_yesterdays_stats = FileSource(
    path=os.path.abspath("data/driver_stats_yesterday.parquet"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

driver_entities = FileSource(
    path=os.path.abspath("data/driver_entity_table.parquet"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)
# Define an entity for the driver. You can think of entity as a primary key used to
# fetch features.
driver = Entity(
    name="driver",
    join_keys=["driver_id"],
    # value_type=ValueType.INT64,
)

ssn = Entity(
    name="ssn",
    join_keys=["ssn"],
)

drivers_license = Entity(
    name="dl",
    join_keys=["dl"],
)

driver_ssn_entities_view = FeatureView(
    name="driver_ssn_entities",
    entities=[ssn],
    ttl=timedelta(days=365 * 10),
    schema=[
        Field(name="driver_id", dtype=Int64),
        Field(name="dl", dtype=String),
    ],
    online=True,
    source=driver_entities,
    tags={},
)

driver_dl_entities_view = FeatureView(
    name="driver_dl_entities",
    entities=[drivers_license],
    ttl=timedelta(days=365 * 10),
    schema=[
        Field(name="driver_id", dtype=Int64),
        Field(name="ssn", dtype=String),
    ],
    online=True,
    source=driver_entities,
    tags={},
)

# Our parquet files contain sample data that includes a driver_id column, timestamps and
# three feature column. Here we define a Feature View that will allow us to serve this
# data to our model online.
driver_hourly_stats_view = FeatureView(
    name="driver_hourly_stats",
    entities=[driver],
    ttl=timedelta(days=365 * 10),
    schema=[
        Field(name="conv_rate", dtype=Float32),
        Field(name="acc_rate", dtype=Float32),
        Field(name="avg_daily_trips", dtype=Int64),
    ],
    online=True,
    source=driver_hourly_stats,
    tags={},
)

driver_yesterdays_stats_view = FeatureView(
    name="driver_yesterdays_stats",
    entities=[driver],
    ttl=timedelta(days=2),
    schema=[
        Field(name="conv_rate", dtype=Float32),
        Field(name="acc_rate", dtype=Float32),
        Field(name="avg_daily_trips", dtype=Int64),
        Field(name="yesterdays_avg_daily_trips_lt_10", dtype=Int64),
        Field(name="yesterdays_acc_rate_lt_01", dtype=Int64),
        Field(name="yesterdays_conv_rate_gt_80", dtype=Int64),
    ],
    online=True,
    source=driver_yesterdays_stats,
    tags={},
)

input_request = RequestSource(
    name="input_request",
    schema=[
        Field(name="date_of_birth", dtype=Int64),
        Field(name="state", dtype=String),
    ],
)


def calculate_age(born):
    today = datetime.utcnow().date()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


@on_demand_feature_view(  # noqa
    sources=[
        driver_hourly_stats_view,
        input_request,
    ],
    schema=[
        Field(name="is_gt_18_years_old", dtype=Int64),
        Field(name="is_valid_state", dtype=Int64),
    ],
)
def transformed_onboarding(inputs: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df["is_valid_state"] = inputs["state"].str.lower().str.contains("sd").astype(int)
    df["is_gt_18_years_old"] = (
        pd.to_datetime(
            inputs["date_of_birth"],
            utc=True,
        )
        .apply(lambda x: calculate_age(x) >= 18)
        .astype(int)
    )
    return df


@on_demand_feature_view(  # noqa
    sources=[
        driver_ssn_entities_view,
    ],
    schema=[
        Field(name="is_previously_seen_ssn", dtype=Int64),
    ],
)
def ondemand_ssn_lookup(inputs: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df["is_previously_seen_ssn"] = (inputs["dl"].isnull() == False).astype(int)
    return df


@on_demand_feature_view(  # noqa
    sources=[
        driver_dl_entities_view,
    ],
    schema=[
        Field(name="is_previously_seen_dl", dtype=Int64),
    ],
)
def ondemand_dl_lookup(inputs: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df["is_previously_seen_dl"] = (inputs["ssn"].isnull() == False).astype(int)
    return df
