import random
import pandas as pd
from feast import FeatureStore
from ml import (
    calculate_onboarding_score,
    calculate_daily_score,
    make_risk_decision,
)
from datetime import datetime

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


def get_onboarding_features(state: str, ssn: str, dl: str, dob: str):
    dob_clean = datetime.strptime(dob, "%m-%d-%Y")
    df = pd.DataFrame(pd.to_datetime([dob_clean.date()]), columns=["date_of_birth"])
    df["driver_id"] = random.randint(1005, 1020)
    df["state"] = state
    df["ssn"] = ssn
    df["dl"] = dl

    feature_vector = store.get_online_features(
        features=[
            "transformed_onboarding:is_gt_18_years_old",
            "transformed_onboarding:is_valid_state",
            "ondemand_ssn_lookup:is_previously_seen_ssn",
            "ondemand_dl_lookup:is_previously_seen_dl",
        ],
        entity_rows=[df.loc[0].to_dict()],
    ).to_dict()
    return feature_vector


def get_onboarding_score(state, ssn, dl, dob):
    features = get_onboarding_features(state, ssn, dl, dob)
    score = calculate_onboarding_score(features)
    print(
        f"\nthe calculated onboarding risk score is {score} with features = {features}\n"
    )
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
    print(f"\nthe calculated daily risk score is {score} with features = {features}\n")
    return score
