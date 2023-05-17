# Feast Demo

To build this demo do the following:
```bash
pyenv use 3.8
poetry shell
pip install -r requirements.txt
cd feature_repo
mkdir data
python generate_data.py
feast apply
CURRENT_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S")
feast materialize-incremental $CURRENT_TIME
```

The output should look like:

```bash
[main]% feast apply
~/.pyenv/versions/3.8.12/lib/python3.8/site-packages/feast/repo_config.py:233: RuntimeWarning: `entity_key_serialization_version` is either not specified in the feature_store.yaml, or is specified to a value <= 1.This serialization version may cause errors when trying to write fields with the `Long` data type into the online store. Specifying `entity_key_serialization_version` to 2 is recommended for new projects.
  warnings.warn(
/Users/francisco.arceo/.pyenv/versions/3.8.12/lib/python3.8/site-packages/flasgger/utils.py:5: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
  import imp
Created entity driver
Created feature view driver_hourly_stats

Created sqlite table feature_repo_driver_hourly_stats
```
```bash
[main]% feast materialize-incremental $CURRENT_TIME
~/.pyenv/versions/3.8.12/lib/python3.8/site-packages/feast/repo_config.py:233: RuntimeWarning: `entity_key_serialization_version` is either not specified in the feature_store.yaml, or is specified to a value <= 1.This serialization version may cause errors when trying to write fields with the `Long` data type into the online store. Specifying `entity_key_serialization_version` to 2 is recommended for new projects.
  warnings.warn(
Materializing 1 feature views to 2023-05-15 12:23:02-04:00 into the sqlite online store.

driver_hourly_stats from 2023-05-14 16:39:03-04:00 to 2023-05-15 12:23:02-04:00:
0it [00:00, ?it/s]
```
Then to serve it simply run
```
python app.py
```

And navigate to http://127.0.0.1:5000/apidocs/ 
