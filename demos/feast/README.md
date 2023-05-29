# A Feast Risk Feature Store Demo

https://github.com/franciscojavierarceo/Python/assets/4163062/394f6f7b-34d1-4c4e-ae69-28c3ec3bf3e4

This is a modification of [Feast's feature store demo](https://docs.feast.dev/getting-started/quickstart) 
to create an OpenAPI wrapper that allows you to browse the endpoints, simulate a user experience, and 
run a real-time machine learning model*.

*Note: this example uses a simple weighted sum of integers that stay between [0,1] for simplicity.
In practice, a real machine learning model can be swamped in but that is not the focus of this demonstration.*

## Motiviation
I built this demo for a talk I gave at Tecton's apply(risk) conferene and the motivation was to show what
some real risk ml systems may look like. This is an extremely trivialized example but it helps highlight the 
different components.

The 5 non-technical (though arguably still technical) takeaways are:
1. It is good to separate feature retrieval, featurization, and model inference inference as much as possible
  - Notice that in the example the output of the features `feature_vector` is a true numpy vector and not non-numeric
2. In this example, the little `batch_job.py` script is trivial but in general data/feature pipelines are rich in business 
logic, context, and (typically) bugs, so it is worth emphasizing that the creation of batch features is non-trivial
3. Notice that this model has an explicit `is_valid_state` feature. What happens if the business enables a new state? 
The model would be wrong and it would unnecessarily penalize customers. This is bad and it's important to understand that
a model works within the context of a product experience.
4. This code and these features are untested. This is a terrible practice. Features should be 
rigorously tested for any production model that will make some sort of risk decision about your customer.
5. Feast is doing a lot of work here for us. The registry/metadata, materialization, online feature execution, and serving.
It is worth emphasizing this point.

## Goals

0. Build Feast using Poetry
1. Use Feast to query Batch and On-Demand features 
2. Materialize data into Feast's Online Store
3. Retrieving online features to run an online model
4. Simulate the a simple batch job to generate more features and materialize them
5. Run a batch model on some cadence to update a user experience

## Installation via PyEnv and Poetry

This demo assumes you have Pyenv (2.3.10) and Poetry (1.4.1) installed on your machine as well as Python 3.8.

```bash
pyenv use 3.8
poetry shell
poetry install
```
## Setting up the data and Feast

We start by generating some mock data via:
```bash
% python generate_data.py
% python batch_job.py
```

Then you can run apply the changes to feast's registry 
```bash
% cd feature_repo/
% feast apply
Created entity ssn
Created entity dl
Created entity driver
Created feature view driver_hourly_stats
Created feature view driver_dl_entities
Created feature view driver_yesterdays_stats
Created feature view driver_ssn_entities
Created on demand feature view transformed_onboarding
Created on demand feature view ondemand_ssn_lookup
Created on demand feature view ondemand_dl_lookup

Created sqlite table feature_repo_driver_dl_entities
Created sqlite table feature_repo_driver_hourly_stats
Created sqlite table feature_repo_driver_ssn_entities
Created sqlite table feature_repo_driver_yesterdays_stats
```

Then to insert records into the table you can run
```bash
% CURRENT_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S")
% feast materialize-incremental $CURRENT_TIME
Materializing 4 feature views to 2023-05-26 11:27:36-07:00 into the sqlite online store.

driver_hourly_stats from 2013-05-28 18:27:45-07:00 to 2023-05-26 11:27:36-07:00:
100%|████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 436.53it/s]
driver_dl_entities from 2013-05-28 18:27:46-07:00 to 2023-05-26 11:27:36-07:00:
100%|████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 531.06it/s]
driver_yesterdays_stats from 2023-05-24 18:27:46-07:00 to 2023-05-26 11:27:36-07:00:
100%|████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 329.82it/s]
driver_ssn_entities from 2013-05-28 18:27:46-07:00 to 2023-05-26 11:27:36-07:00:
100%|████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 508.89it/s]
```
And you're done! Your data from the parquet file has been inserted into Feast

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

