
def calculate_onboarding_score(features) -> int:
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

def make_risk_decision(score, thresh=0.5):
    return 'Approved' if score < thresh else 'Decline'

