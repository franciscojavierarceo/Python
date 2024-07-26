# README

This is a light-weight example of a typical data science pipeline that would
be used for creating a quantitative stock trading application.

This "quantitative stock trading application" requires the following:
- Pulling data from a 3rd party API (Polygon) on a recurring schedule (daily)
- Creating a historical extract of raw data to build a model
    - The model attempts to predict the next day's Nasdaq open price based on
    features contstructed from the 7 of the top technology stocks
        - The features are various window aggregations of the open/close price - Storing the historical extract and the incrementals
- Creating a simple Neural Network to estimate the model
- Estimating the model and saving it
- Scoring the latest data that could be used in the "buy" application


There are two interesting things about this demo.

1. One single script is created to do the whole thing as a data scientist might
put it together.

2. A KFP is made from (1) and cleaned up.


# The Pipeline

The pipeilne has 4 major components

1. Fetch the data
2. Process the data
3. Train the model
4. Score the data

There is a fifth step that is ommitted at the moment but would be good to add
which would be to upload the data to Feast (the Feature Store) so that it could
be served online in some application.
