import pandas as pd
import datetime


def generate_entity_data() -> None:
    td = datetime.datetime.utcnow()
    payload = {
        "event_timestamp": td,
        "created": td,
        "driver_id": 1001,
        "ssn": "123-45-6789",
        "dl": "some-dl-number",
    }
    df = pd.DataFrame([payload])
    outpath = "./feature_repo/data/driver_entity_table.parquet"
    df.to_parquet(outpath, allow_truncated_timestamps=True)
    print(f"data written to {outpath}")


if __name__ == "__main__":
    generate_entity_data()
