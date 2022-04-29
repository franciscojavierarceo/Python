# Feast Feature Store Framework

As outlined here: https://docs.feast.dev/getting-started/quickstart


To reproduce simply run:

```bash
[main]% cd feature_repo
[main]% feast apply
Created entity driver
Created feature view driver_hourly_stats

Created sqlite table feature_repo_driver_hourly_stats

[main]% CURRENT_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S")
[main]% feast materialize-incremental $CURRENT_TIME
Materializing 1 feature views to 2022-04-28 21:50:07-06:00 into the sqlite online store.

driver_hourly_stats from 2022-04-28 03:50:14-06:00 to 2022-04-28 21:50:07-06:00:
100%|████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 389.52it/s]

[main]% python print_features.py 
{
  'acc_rate': [0.039190519601106644, 0.9062243103981018],
  'avg_daily_trips': [693, 274],
  'conv_rate': [0.4921093285083771, 0.3993831276893616],
  'driver_id': [1004, 1005]
 }

```