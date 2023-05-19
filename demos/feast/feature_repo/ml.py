def calculate_onboarding_score(features) -> float:
    # intercept
    score = 0.03
    coef = {
        "is_gt_18_years_old": 0.25,
        "is_valid_state": 0.25,
        "is_previously_seen_ssn": 0.25,
        "is_previously_seen_dl": 0.25,
    }
    for f in coef:
        score += coef[f] * features[f][0]

    return score

def calculate_daily_score(features) -> float:
    # intercept
    score = 0.01
    coef = {
        "conv_rate": 0.01,
        "acc_rate": 0.01,
        "avg_daily_trips": 0.001,
        "yesterdays_avg_daily_trips_lt_10": 0.20,
        "yesterdays_acc_rate_lt_01": 0.20,
        "yesterdays_conv_rate_gt_80": 0.20,
    }
    for f in coef:
        score += coef[f] * features[f][0]

    return score

def make_risk_decision(score, thresh=0.5) -> str:
    return 'Approved' if score < thresh else 'Decline'

