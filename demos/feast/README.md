# Feast Feature Store Demo

This is a modification of [Feast's feature store demo](https://docs.feast.dev/getting-started/quickstart).

I also create a small Flask OpenAPI wrapper that lets you query
by an ID to fetch the features for that corresponding thing.

## Goals

The goal of this demo is to highlight the following:

1. Querying features online
2. Querying historical features at a point in time for analysis
3. Creating new features and pushing them to the feature store
4. Back-filling historical features as of a certain date

## Details 
This example uses Feast's pre-generated mock data for driver metrics.

To run the app simply do the following:

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
 
 [main]% export FLASK_APP=hello
 [main]% export FLASK_ENV=development
 [main]% flask run
 * Serving Flask app 'app' (lazy loading)
 * Environment: development
 * Debug mode: on
04/30/2022 09:44:08 AM INFO: * Running on http://127.0.0.1:5000 (Press CTRL+C to quit)
04/30/2022 09:44:08 AM INFO: * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 221-452-258
```

The data looks like
```
            event_timestamp  driver_id  conv_rate  acc_rate  avg_daily_trips                 created
0 2022-04-13 21:00:00+00:00       1005   0.409039  0.631994              359 2022-04-28 21:29:55.788
1 2022-04-13 22:00:00+00:00       1005   0.524565  0.317398              297 2022-04-28 21:29:55.788
2 2022-04-13 23:00:00+00:00       1005   0.300072  0.376808              295 2022-04-28 21:29:55.788
3 2022-04-14 00:00:00+00:00       1005   0.533681  0.984958              727 2022-04-28 21:29:55.788
4 2022-04-14 01:00:00+00:00       1005   0.999158  0.559122              745 2022-04-28 21:29:55.788
```

## Feast UI

If using feast version 0.21.0 (and potentially others) you can render a Feast UI by simply running

```
feast ui
```
