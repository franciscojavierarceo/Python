# A Feast Risk Feature Store Demo

This is a modification of [Feast's feature store demo](https://docs.feast.dev/getting-started/quickstart) 
to create an OpenAPI wrapper that allows you to browse the endpoints, simulate a user experience, and 
run a real-time machine learning model*.

*Note: this example uses a simple weighted sum of integers that stay between [0,1] for simplicity.
In practice, a real machine learning model can be swamped in but that is not the focus of this demonstration.*

## Goals

0. Build Feast using Poetry
1. Use Feast to query Batch and On-Demand features 
2. Materialize data into Feast's Online Store
3. Retrieving online features to run an online model
4. Simulate the a simple batch job to generate more features and materialize them
5. Run a batch model on some cadence to update a user experience

## Getting started
```bash
pyenv use 3.8
poetry shell
poetry install
python generate_data.py
python batch_job.py
```

Then you can run

```bash
cd feature_repo/
% feast apply
Created entity driver
Created feature view driver_yesterdays_stats
Created feature view driver_hourly_stats
Created on demand feature view transformed_onboarding

Created sqlite table feature_repo_driver_hourly_stats
Created sqlite table feature_repo_driver_yesterdays_stats
```

Then to insert records into the table you can run
```bash
% CURRENT_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S")
% feast materialize-incremental $CURRENT_TIME
Materializing 2 feature views to 2023-05-26 08:34:05-07:00 into the sqlite online store.

driver_yesterdays_stats from 2023-05-24 15:34:12-07:00 to 2023-05-26 08:34:05-07:00:
100%|████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 118.92it/s]
driver_hourly_stats from 2013-05-28 15:34:12-07:00 to 2023-05-26 08:34:05-07:00:
100%|███████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 3359.74it/s]
```
And you're done!

# Flask Demo

After getting Feast configured and materialized, you can then run a web server to demo our little Driver Risk application.

This can be done by running
```bash
[main]% python app.py
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Restarting with stat
```

Then navigate to http://127.0.0.1:5000/apidocs/ and you can see the endpoints.

## Data Details 
This example uses Feast's pre-generated mock data for driver metrics.

The data looks something like the sample below (the time stamp may differ)
```
            event_timestamp  driver_id  conv_rate  acc_rate  avg_daily_trips                 created
0 2022-04-13 21:00:00+00:00       1005   0.409039  0.631994              359 2022-04-28 21:29:55.788
1 2022-04-13 22:00:00+00:00       1005   0.524565  0.317398              297 2022-04-28 21:29:55.788
2 2022-04-13 23:00:00+00:00       1005   0.300072  0.376808              295 2022-04-28 21:29:55.788
3 2022-04-14 00:00:00+00:00       1005   0.533681  0.984958              727 2022-04-28 21:29:55.788
4 2022-04-14 01:00:00+00:00       1005   0.999158  0.559122              745 2022-04-28 21:29:55.788
```

I generate additional samples of this data using some very trivial scripts to simulate what happens in live customer-facing experiences and what happens behind the scenes through common batch jobs.

## Feast UI

This demo uses Feast 0.28.0 which means you can render a Feast UI by simply running.
```
feast ui
```

You'll notice that the feature views are nicely documented. 👍

## Requirements

This demo assumes you have Pyenv (2.3.10) and Poetry (1.4.1) installed on your machine as well as Python 3.8.

Note, I did this with 3.8.12

