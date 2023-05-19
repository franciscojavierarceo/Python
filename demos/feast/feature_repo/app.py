from flask import Flask, jsonify, request
from feast import FeatureStore
from flasgger import Swagger
from datetime import datetime
import pandas as pd
import sqlite3
from ml import (
    calculate_onboarding_score,
    calculate_daily_score,
    make_risk_decision,
)

store = FeatureStore(repo_path=".")


def get_demo_historical_features():
    entity_df = pd.DataFrame.from_dict(
        {
            "driver_id": [1001, 1002, 1003, 1004],
            "event_timestamp": [
                datetime(2021, 4, 12, 10, 59, 42),
                datetime(2021, 4, 12, 8, 12, 10),
                datetime(2021, 4, 12, 8, 12, 10),
                datetime(2021, 4, 12, 8, 12, 10),
            ],
        }
    )
    retrieval_job = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "driver_hourly_stats:conv_rate",
            "driver_hourly_stats:acc_rate",
            "driver_hourly_stats:avg_daily_trips",
        ],
    )
    return jsonify(retrieval_job.to_df().to_dict())

def get_onboarding_features(state: str, ssn: str, dl: str):
    df = pd.DataFrame(pd.to_datetime([datetime.utcnow().date()]),
                      columns=['date_of_birth'])
    df['driver_id'] = 1
    df['state'] = 'NJ'
    df['ssn'] = '123-45-6789'
    df['dl'] = 'asdfpoijpasdf'

    feature_vector = store.get_online_features(
        features=[
            "transformed_onboarding:is_gt_18_years_old",
            "transformed_onboarding:is_valid_state",
            "transformed_onboarding:is_previously_seen_ssn",
            "transformed_onboarding:is_previously_seen_dl",
        ],
        entity_rows=[df.loc[0].to_dict()],
    ).to_dict()
    return feature_vector

def get_onboarding_score(state, ssn, dl):
    features = get_onboarding_features(state, ssn, dl)
    score = calculate_onboarding_score(features)
    print(f'the calculated onboarding risk score is {score} with features = {features}')
    return score

def get_daily_features(driver_id: int):
    rows = [{"driver_id": driver_id}]
    feature_vector = store.get_online_features(
        features=[
            "driver_yesterdays_stats:conv_rate",
            "driver_yesterdays_stats:acc_rate",
            "driver_yesterdays_stats:avg_daily_trips",
            "driver_yesterdays_stats:yesterdays_avg_daily_trips_lt_10",
            "driver_yesterdays_stats:yesterdays_acc_rate_lt_01",
            "driver_yesterdays_stats:yesterdays_conv_rate_gt_80",
        ],
        entity_rows=rows,
    ).to_dict()

    return feature_vector

def get_daily_score(driver_id: int):
    features = get_daily_features(driver_id)
    score = calculate_daily_score(features)
    print(f'the calculated daily risk score is {score} with features = {features}')
    return make_risk_decision(score)


app = Flask(__name__)
swagger = Swagger(app)

@app.route("/onboarding-risk-features/")
def onboarding():
    """Example endpoint returning features by id
    This is using docstrings for specifications.
    ---
    parameters:
      - name: state
        type: string
        in: query
        required: true
        default: NJ

      - name: ssn
        type: string
        in: query
        required: true
        default: 123-45-6789

      - name: dl
        type: string
        in: query
        required: true
        default: some-dl-number

    responses:
      200:
        description: A JSON of features
        schema:
          id: OnboardingFeatures
          properties:
            is_gt_18_years_old:
              type: array
              items:
                schema:
                  id: value
                  type: number
            is_valid_state:
              type: array
              items:
                schema:
                  id: value
                  type: number
            is_previously_seen_ssn:
              type: array
              items:
                schema:
                  id: value
                  type: number
            is_previously_seen_dl:
              type: array
              items:
                schema:
                  id: value
                  type: number
    """
    r = request.args
    feature_vector = get_onboarding_features(r.get("state"), r.get("ssn"), r.get("dl"))
    return jsonify(feature_vector)





@app.route("/onboarding-risk-score/")
def score_onboarding_risk():
    """Example endpoint returning features by id
    This is using docstrings for specifications.
    ---
    parameters:
      - name: state
        type: string
        in: query
        required: true
        default: NJ

      - name: ssn
        type: string
        in: query
        required: true
        default: 123-45-6789

      - name: dl
        type: string
        in: query
        required: true
        default: some-dl-number
    responses:
      200:
        description: A Decision about onboarding socre
        schema:
          id: Score
          properties:
            score:
              type: number
    """
    r = request.args
    score = get_onboarding_score(r.get("state"), r.get("ssn"), r.get("dl"))
    return jsonify(score)

@app.route("/onboarding-risk-decision/")
def decide_onboarding_risk():
    """Example endpoint returning features by id
    This is using docstrings for specifications.
    ---
    parameters:
      - name: state
        type: string
        in: query
        required: true
        default: NJ

      - name: ssn
        type: string
        in: query
        required: true
        default: 123-45-6789

      - name: dl
        type: string
        in: query
        required: true
        default: some-dl-number
    responses:
      200:
        description: A Decision about onboarding socre
        schema:
          id: Decision
          properties:
            decision:
              type: string
    """
    r = request.args
    score = get_onboarding_score(r.get("state"), r.get("ssn"), r.get("dl"))
    return jsonify(make_risk_decision(score))

@app.route("/daily-risk-features/<driver_id>/")
def driver_daily_features(driver_id: int):
    """Example endpoint returning features by id
    This is using docstrings for specifications.
    ---
    parameters:
      - name: driver_id
        in: path
        type: integer
        required: true
        default: 1001

    definitions:
      DriverId:
        type: object
        properties:
          id_name:
            type: integer

    responses:
      200:
        description: A JSON of features
        schema:
          id: DailyFeatures
          properties:
            acc_rate:
              type: array
              items:
                schema:
                  id: value
                  type: number
            conv_rate:
              type: array
              items:
                schema:
                  id: value
                  type: number
            avg_daily_trips:
              type: array
              items:
                schema:
                  id: value
                  type: number
            yesterdays_avg_daily_trips_lt_10:
              type: array
              items:
                schema:
                  id: value
                  type: number
            yesterdays_acc_rate_lt_01:
              type: array
              items:
                schema:
                  id: value
                  type: number
            yesterdays_conv_rate_gt_80:
              type: array
              items:
                schema:
                  id: value
                  type: number
            driver_id:
              type: array
              items:
                schema:
                  id: value
                  type: integer
            event_timestamp:
              type: array
              items:
                schema:
                  id: value
                  type: string
    """
    return jsonify(get_daily_features(driver_id))


@app.route("/daily-risk-score/<driver_id>/")
def driver_daily_risk(driver_id: int):
    """Example endpoint returning features by id
    This is using docstrings for specifications.
    ---
    parameters:
      - name: driver_id
        in: path
        type: integer
        required: true
        default: 1001

    definitions:
      DriverId:
        type: object
        properties:
          id_name:
            type: integer

    responses:
      200:
        description: A Decision about onboarding socre
        schema:
          id: Score
          properties:
            score:
              type: number
    """
    score = get_daily_score(driver_id)
    return jsonify(score)

@app.route("/daily-risk-decision/<driver_id>/")
def driver_daily_decision(driver_id: int):
    """Example endpoint returning features by id
    This is using docstrings for specifications.
    ---
    parameters:
      - name: driver_id
        in: path
        type: integer
        required: true
        default: 1001

    definitions:
      DriverId:
        type: object
        properties:
          id_name:
            type: integer

    responses:
      200:
        description: A Decision about onboarding socre
        schema:
          id: Decision
          properties:
            decision:
              type: string
    """
    score = get_daily_score(driver_id)
    return jsonify(make_decision(score))

@app.route("/historical-features/")
def historical():
    """Example endpoint returning all historical features
    This is using docstrings for specifications.
    ---
    responses:
      200:
        description: A JSON of features
        schema:
          id: HistoricalFeatures
          properties:
            acc_rate:
              type: array
              items:
                schema:
                  id: value
                  type: number
            conv_rate:
              type: array
              items:
                schema:
                  id: value
                  type: number
            avg_daily_trips:
              type: array
              items:
                schema:
                  id: value
                  type: number
            driver_id:
              type: array
              items:
                schema:
                  id: value
                  type: integer
            event_timestamp:
              type: array
              items:
                schema:
                  id: value
                  type: number
    """
    return get_demo_historical_features()


if __name__ == "__main__":
    app.run(debug=True)
