from flask import Flask, jsonify
from feast import FeatureStore
from flasgger import Swagger


def get_feature_vector(id):
    rows = [{"driver_id": id}]
    store = FeatureStore(repo_path=".")
    feature_vector = store.get_online_features(
        features=[
            "driver_hourly_stats:conv_rate",
            "driver_hourly_stats:acc_rate",
            "driver_hourly_stats:avg_daily_trips",
        ],
        entity_rows=rows,
    ).to_dict()
    return jsonify(feature_vector)


app = Flask(__name__)
swagger = Swagger(app)


@app.route("/<id>/")
def hello(id):
    """Example endpoint returning features by id
    This is using docstrings for specifications.
    ---
    parameters:
      - name: id
        in: path
        type: integer
        required: true
        default: 1004

    definitions:
      Id:
        type: object
        properties:
          id_name:
            type: array
            items:
              $ref: '#/definitions/Color'
      Color:
        type: string
    responses:
      200:
        description: A json of features
        schema:
          $ref: '#/definitions/Id'
        examples:
          rgb: ['red', 'green', 'blue']
    """
    return get_feature_vector(id)


if __name__ == "__main__":
    app.run(debug=True)
