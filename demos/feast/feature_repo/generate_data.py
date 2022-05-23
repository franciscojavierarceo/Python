import pandas as pd
import datetime


def generate_new_data() -> None:
    td = datetime.datetime.utcnow()
    payload = {
        "event_timestamp": td,
        "driver_id": 1001,
        "conv_rate": 0.3,
        "acc_rate": 0.4,
        "avg_daily_trips": 200,
    }
    df = pd.DataFrame([payload])
    df.to_parquet("./data/driver_stats_new.parquet", allow_truncated_timestamps=True)


def main():
    generate_new_data()


if __name__ == "__main__":
    main()
