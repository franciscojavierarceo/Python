import pandas as pd
from datetime import datetime, timedelta

def main():
    print('simulating running a batch job against sqlite3')
    td = datetime.utcnow() - timedelta(days=1)
    payload = {
        "created": td,
        "event_timestamp": td,
        "driver_id": 1001,
        "conv_rate": 0.3,
        "acc_rate": 0.4,
        "avg_daily_trips": 200,
    }
    df = pd.DataFrame([payload])
    df["yesterdays_avg_daily_trips_lt_10"] = (df['avg_daily_trips'] < 10).astype(int)
    df["yesterdays_acc_rate_lt_01"] = (df['acc_rate'] < 0.01).astype(int)
    df["yesterdays_conv_rate_gt_80"] = (df['conv_rate'] > 0.8).astype(int)

    export_file = "./feature_repo/data/driver_stats_yesterday.parquet"
    df.to_parquet(export_file, allow_truncated_timestamps=True)
    print(f"exporting data to {export_file}")

if __name__ == '__main__':
    main()
