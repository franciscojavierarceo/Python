from flask import Flask, jsonify
from feast import FeatureStore
from flasgger import Swagger
from datetime import datetime
import pandas as pd

store = FeatureStore(repo_path=".")


def get_feature_vector(driver_id):
    rows = [{"driver_id": driver_id}]
    feature_vector = store.get_online_features(
        features=[
            "driver_hourly_stats:conv_rate",
            "driver_hourly_stats:acc_rate",
            "driver_hourly_stats:avg_daily_trips",
        ],
        entity_rows=rows,
    ).to_dict()

    return jsonify(feature_vector)


def get_historical_features():
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


app = Flask(__name__)
swagger = Swagger(app)


@app.route("/<driver_id>/")
def hello(driver_id: int):
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
          id: Features
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
                  type: string
    """
    return get_feature_vector(driver_id)


@app.route("/historical-features/")
def historical():
    """Example endpoint returning features by id
    This is using docstrings for specifications.
    ---
    responses:
      200:
        description: A JSON of features
        schema:
          id: Features
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
    """
    return get_historical_features()


if __name__ == "__main__":
    app.run(debug=True)
