import cProfile

import subprocess
from datetime import datetime

import pandas as pd
from datetime import datetime

from feast import FeatureStore

entity_rows = [
    # {join_key: entity_value}
    {
        "driver_id": 1001,
        "val_to_add": 1000,
        "val_to_add_2": 2000,
    },
    {
        "driver_id": 1002,
        "val_to_add": 1001,
        "val_to_add_2": 2002,
    },
]

store = FeatureStore(repo_path=".")
store.materialize_incremental(end_date=datetime.now())

def odfv_pandas():
    features_to_fetch = [
        "transformed_conv_rate_fresh:conv_rate_plus_val1",
        "transformed_conv_rate_fresh:conv_rate_plus_val2",
    ]
    returned_features = store.get_online_features(
        features=features_to_fetch,
        entity_rows=entity_rows,
    ).to_dict()

def odfv_python():
    features_to_fetch = [
        "transformed_conv_rate_fresh_python:conv_rate_plus_val1",
        "transformed_conv_rate_fresh_python:conv_rate_plus_val2",
    ]
    returned_features = store.get_online_features(
        features=features_to_fetch,
        entity_rows=entity_rows,
    ).to_dict()


def main():
    print("running pandas odfv...")
    profiler = cProfile.Profile()
    profiler.enable()
    odfv_pandas()
    profiler.disable()
    profiler.dump_stats("odfv_pandas.prof")

    print("running python odfv...")
    profiler = cProfile.Profile()
    profiler.enable()
    odfv_python()
    profiler.disable()
    profiler.dump_stats("odfv_python.prof")
    print("...done")


if __name__ == "__main__":
    main()
