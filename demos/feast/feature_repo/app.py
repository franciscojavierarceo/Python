from flask import (
    Flask,
    jsonify,
    request,
    render_template,
)
from flasgger import Swagger
from datetime import datetime
from get_features import (
    get_onboarding_features,
    get_onboarding_score,
    get_daily_features,
    get_daily_score,
)
from ml import make_risk_decision

app = Flask(__name__)
swagger = Swagger(app)

@app.route("/")
def onboarding_page():
    return render_template("index.html")

@app.route("/home")
def home_page():
    return render_template("home.html")

@app.route("/onboarding-risk-features/", methods=["POST"])
def onboarding_features():
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

      - name: dob
        type: string
        in: query
        required: true
        default: 12-23-2000
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
    feature_vector = get_onboarding_features(r.get("state"), r.get("ssn"), r.get("dl"), r.get("dob"))
    return jsonify(feature_vector)


@app.route("/onboarding-risk-score/", methods=["POST"])
def onboarding_risk_score():
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

      - name: dob
        type: string
        in: query
        required: true
        default: 12-23-2000
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
    score = get_onboarding_score(r.get("state"), r.get("ssn"), r.get("dl"), r.get("dob"))
    return jsonify({"score": score})

@app.route("/onboarding-risk-decision/", methods=["POST"])
def onboarding_risk_decision():
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

      - name: dob
        type: string
        in: query
        required: true
        default: 12-23-2000
    responses:
      200:
        description: A Decision about onboarding socre
        schema:
          id: Decision
          properties:
            decision:
              type: string
            score:
              type: number
            model:
              type: string
    """
    r = request.args
    print(r)
    score = get_onboarding_score(r.get("state"), r.get("ssn"), r.get("dl"), r.get("dob"))
    return jsonify({"decision": make_risk_decision(score), "score": score, "model": "onboarding"})

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
def driver_daily_risk_score(driver_id: int):
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
    return jsonify({"score": score})

@app.route("/daily-risk-decision/<driver_id>/")
def driver_daily_risk_decision(driver_id: int):
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
            score:
              type: number
            model:
              type: string
    """
    score = get_daily_score(driver_id)
    return jsonify({"decision": make_risk_decision(score), "score": score, "model": "daily"})

if __name__ == "__main__":
    app.run(debug=True)
